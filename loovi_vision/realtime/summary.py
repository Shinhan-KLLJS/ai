from datetime import datetime, timezone

from loovi_vision.enrich.session_summary import demographics_of
from loovi_vision.analysis.gaze_sessions import DWELL_BUCKETS, dwell_bucket


class SummaryAggregator:
    # 서버로 나가는 유일한 메시지(`summary`)를 만든다. 기본 5초마다 그 "구간(window)"의
    # OTS/LTS 카운트 + 성별/연령 분포 + Attention(시청시간)을 한 번에 담는다.
    #
    # ▷ 값 성격: 전부 "그 구간값"이다(누적 아님). 서버는 여러 구간을 합쳐 임의 시간대로 롤업한다.
    #
    # ▷ 구간값 만드는 방식 (a: 재집계 후 diff)
    #    1) 매 주기 registry 로 "세션 시작~현재 누적"을 다시 계산한다(_cumulative).
    #    2) 직전 주기 누적(self.prev)과 빼서 "이번 구간에 늘어난 만큼"을 얻는다(_window).
    #    3) 이번 누적을 self.prev 로 저장해 다음 구간 diff 기준으로 쓴다.
    #
    # ▷ 정확도
    #    - 카운트(ots/lts)·시청시간 합(dwell_sum)은 값이 단조 증가라 diff 가 정확하다.
    #    - 성별/연령·시청분포 버킷은 사람이 버킷을 옮기면(나이 재판정, 1~2초→2초+ 등) 음수가 날 수
    #      있어 0으로 clamp 한다(데모급 근사). 서버는 카운트/시간 합을 신뢰값으로 쓴다.
    #
    # ▷ Attention 을 poses 파일이 아니라 registry(facing_sec)로 계산하는 이유
    #    한 사람의 총 시청시간 = 그 track 의 누적 facing_sec 이므로 registry 만으로 충분하다.
    #    덕분에 매 주기 poses 파일을 다시 읽지 않아 가볍다.
    def __init__(self, settings, registry):
        self.settings = settings
        self.registry = registry
        self.prev = self._zero()   # 직전 주기까지의 누적값(diff 기준). 최초엔 모두 0.

    def build(self, seq, interval_sec):
        # 한 구간의 summary 메시지를 만든다. interval_sec = 직전 전송 이후 실제 경과 초(보통 5).
        cur = self._cumulative()             # 현재까지 누적
        win = self._window(self.prev, cur)   # 이번 구간 = 현재 − 직전
        self.prev = cur                      # 다음 구간 diff 기준 갱신
        return {
            "device_id": self.settings.rt_device_id,
            "board_id": self.settings.board_id,
            "seq": seq,                            # 메시지 순번(1부터). 순서 보장·중복 제거용
            "timestamp": self._timestamp(),        # window 종료 시각(UTC). window=[종료−interval_sec, 종료]
            "interval_sec": round(interval_sec, 3),
            "ots_count": win["ots"],               # 이 구간 신규 통행(확정 track)
            "lts_count": win["lts"],               # 이 구간 신규 응시자(누적 응시>=1초)
            "ots_demographics": win["ots_demo"],   # 얼굴 잡힌 통행자 성별 + 성별별 연령
            "lts_demographics": win["lts_demo"],   # 응시자 성별 + 성별별 연령
            "attention": {
                "avg_dwell_sec": win["avg_dwell"],   # 이 구간 신규 응시자 인원 기준 평균(Σ응시÷인원)
                "dwell_sum_sec": win["dwell_sum"],   # 이 구간 시청시간 합. 서버 롤업(Σ합÷Σ인원)의 재료
                "dwell_distribution": win["dwell_dist"],  # 1~2초/2초+ 응시자 수
            },
        }

    def _cumulative(self):
        # registry 를 훑어 "세션 시작~현재"까지의 누적 상태를 만든다.
        min_hits = self.settings.track_min_hits       # OTS 확정 기준(최소 관측 프레임 수)
        lts_min = self.settings.gaze_lts_min_sec       # LTS 기준(누적 응시 초)
        # OTS: 짧게 스친 노이즈를 빼기 위해 min_hits 이상 관측된 "확정 track"만 센다.
        confirmed = [s for s in self.registry.all() if s.frames_seen >= min_hits]
        # 성별/연령은 얼굴이 잡힌 사람만 추정 가능 → attended(얼굴 1회 이상 검출)로 한 번 더 거른다.
        attended = [s for s in confirmed if s.attended]
        # LTS(응시자): 누적 응시시간(facing_sec)이 임계 이상인 track.
        gazers = [s for s in self.registry.all() if s.facing_sec >= lts_min]
        # 시청시간 분포: 각 응시자의 총 시청시간(facing_sec)을 1~2초/2초+ 로 분류해 인원을 센다.
        dwell_dist = {b: 0 for b in DWELL_BUCKETS}
        for s in gazers:
            dwell_dist[dwell_bucket(s.facing_sec, self.settings.gaze_grade_dwell_sec)] += 1
        return {
            "ots": len(confirmed),
            "lts": len(gazers),
            "ots_demo": demographics_of(attended),
            "lts_demo": demographics_of(gazers),
            "dwell_sum": sum(s.facing_sec for s in gazers),  # 응시자 총 시청시간 합(인원 평균의 분자)
            "dwell_dist": dwell_dist,
        }

    def _window(self, prev, cur):
        # 누적(cur) − 직전 누적(prev) = 이번 구간값. 음수는 0으로 clamp.
        lts = max(0, cur["lts"] - prev["lts"])                       # 이번 구간 신규 응시자 수
        dwell_sum = max(0.0, cur["dwell_sum"] - prev["dwell_sum"])   # 이번 구간 늘어난 시청시간 합
        return {
            "ots": max(0, cur["ots"] - prev["ots"]),
            "lts": lts,
            "ots_demo": self._demo_diff(prev["ots_demo"], cur["ots_demo"]),
            "lts_demo": self._demo_diff(prev["lts_demo"], cur["lts_demo"]),
            # 이번 구간 평균: 신규 응시자(lts)가 없으면 0으로 둔다. 단 그런 구간에도 dwell_sum 은
            # 그대로 실어 보내, 서버 롤업(Σdwell_sum ÷ Σlts)에서 "계속 보던 사람"의 시청시간이
            # 유실되지 않게 한다. ← dwell_sum_sec 를 별도로 보내는 이유.
            "avg_dwell": round(dwell_sum / lts, 3) if lts else 0.0,
            "dwell_sum": round(dwell_sum, 3),
            # 버킷별 diff(clamp). 사람이 1~2초→2초+ 로 옮기면 낮은 버킷에서 음수가 나므로 0으로 막는다.
            "dwell_dist": {b: max(0, cur["dwell_dist"][b] - prev["dwell_dist"][b]) for b in DWELL_BUCKETS},
        }

    def _demo_diff(self, prev, cur):
        # 성별/연령 분포를 버킷별로 diff. 나이·성별 재판정으로 버킷이 바뀌면 음수가 날 수 있어 clamp 0.
        def diff(key):
            return {k: max(0, cur[key][k] - prev[key][k]) for k in cur[key]}
        return {"gender": diff("gender"), "male_age": diff("male_age"), "female_age": diff("female_age")}

    def _zero(self):
        # 최초 구간의 diff 기준(모든 누적값 0). 첫 build 는 "0~현재"가 곧 첫 구간이 된다.
        return {"ots": 0, "lts": 0, "ots_demo": demographics_of([]), "lts_demo": demographics_of([]),
                "dwell_sum": 0.0, "dwell_dist": {b: 0 for b in DWELL_BUCKETS}}

    def _timestamp(self):
        # 메시지 timestamp 는 UTC ISO8601(Z) 고정. 지역시간 변환은 서버/클라이언트 몫.
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
