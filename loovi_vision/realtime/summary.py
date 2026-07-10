from datetime import datetime, timezone

from loovi_vision.enrich.session_summary import demographics_of
from loovi_vision.analysis.gaze_sessions import DWELL_BUCKETS, dwell_bucket


class SummaryAggregator:
    # 서버로 나가는 유일한 메시지(`summary`)를 만든다. 기본 5초마다 그 "구간(window)"의
    # OTS/LTS 카운트 + 성별/연령 분포 + Attention(시청시간 분포)을 한 번에 담는다.
    #
    # ▷ 값 성격: 전부 "그 구간값"이다(누적 아님). 서버는 여러 구간을 합쳐 임의 시간대로 롤업한다.
    #
    # ▷ 두 가지 계산 방식 (필드 성격에 따라 다름)
    #    - 카운트/합(ots_count·lts_count·dwell_sum): "세션 시작~현재 누적"을 재집계해 직전과 diff.
    #      값이 단조 증가라 diff 가 정확하다(_cumulative/_window).
    #    - 분포(성별/연령 demographics·시청시간 dwell_distribution): diff 로 만들면 사람이 칸을
    #      옮길 때(나이 재판정, 1~2초→2초+ 등) 지나온 칸마다 중복 계상된다(음수 금지 스키마 → clamp 로
    #      "빠짐"이 유실). 그래서 "종료(안 보인 지 exit_grace 초 / 세션 마감) 시점에 최종값으로 1회만"
    #      세는 방식을 쓴다(_finalized) → 각 사람 딱 한 번, 서버가 구간들을 더해도 중복이 없다.
    #      (참고: 사람이 끝나야 잡히므로 약간 지연된다. 기간 전체 합계는 정확.)
    def __init__(self, settings, registry):
        self.settings = settings
        self.registry = registry
        self.prev = self._zero()      # 카운트/합의 직전 누적값(diff 기준). 최초엔 모두 0.
        self.done_ots_ids = set()     # ots_demographics 에 1회 반영한(종료된) 통행자
        self.done_gazer_ids = set()   # lts_demographics·dwell_distribution 에 1회 반영한 응시자

    def build(self, seq, interval_sec, now_sec=None, final=False):
        # 한 구간의 summary 메시지를 만든다. interval_sec = 직전 전송 이후 실제 경과 초(보통 5).
        # now_sec = 현재 시각(elapsed 초, 종료 판정용). final=True 면 남은 사람을 모두 종료 처리(세션 마감).
        cur = self._cumulative()             # 카운트/합: 현재까지 누적
        win = self._window(self.prev, cur)   # 이번 구간 = 현재 − 직전
        self.prev = cur
        # 분포: 이 구간에 "종료된" 사람만 최종값으로 1회 계상(중복 없음).
        ots_done = self._finalized(now_sec, final, self._is_ots, self.done_ots_ids)
        gazers_done = self._finalized(now_sec, final, self._is_gazer, self.done_gazer_ids)
        dwell = {b: 0 for b in DWELL_BUCKETS}
        for s in gazers_done:
            dwell[dwell_bucket(s.facing_sec)] += 1
        return {
            "device_id": self.settings.rt_device_id,
            "board_id": self.settings.board_id,
            "seq": seq,                              # 메시지 순번(1부터). 순서 보장·중복 제거용
            "timestamp": self._timestamp(),          # window 종료 시각(UTC). window=[종료−interval_sec, 종료]
            "interval_sec": round(interval_sec, 3),
            "ots_count": win["ots"],                 # 이 구간 신규 통행(확정 track)
            "lts_count": win["lts"],                 # 이 구간 신규 응시자(누적 응시>=1초)
            "ots_demographics": demographics_of(ots_done),    # 이 구간에 종료된 통행자의 성별+연령(최종값)
            "lts_demographics": demographics_of(gazers_done), # 이 구간에 종료된 응시자의 성별+연령(최종값)
            "attention": {
                "avg_dwell_sec": win["avg_dwell"],   # 이 구간 신규 응시자 인원 기준 평균(Σ응시÷인원)
                "dwell_sum_sec": win["dwell_sum"],   # 이 구간 시청시간 합. 서버 롤업(Σ합÷Σ인원)의 재료
                "dwell_distribution": dwell,         # 이 구간에 종료된 응시자를 최종 버킷으로 1회 계상
            },
        }

    def _cumulative(self):
        # 카운트/합만 누적 계산한다(분포는 _finalized 로 별도 처리).
        min_hits = self.settings.track_min_hits       # OTS 확정 기준(최소 관측 프레임 수)
        lts_min = self.settings.gaze_lts_min_sec       # LTS 기준(누적 응시 초)
        confirmed = [s for s in self.registry.all() if s.frames_seen >= min_hits]
        gazers = [s for s in self.registry.all() if s.facing_sec >= lts_min]
        return {
            "ots": len(confirmed),
            "lts": len(gazers),
            "dwell_sum": sum(s.facing_sec for s in gazers),  # 응시자 총 시청시간 합(인원 평균의 분자)
        }

    def _window(self, prev, cur):
        # 누적(cur) − 직전 누적(prev) = 이번 구간값. 값이 단조 증가라 음수가 없다.
        lts = max(0, cur["lts"] - prev["lts"])                       # 이번 구간 신규 응시자 수
        dwell_sum = max(0.0, cur["dwell_sum"] - prev["dwell_sum"])   # 이번 구간 늘어난 시청시간 합
        return {
            "ots": max(0, cur["ots"] - prev["ots"]),
            "lts": lts,
            # 신규 응시자 없으면 평균 0. 단 dwell_sum 은 그대로 실어 서버 롤업(Σsum÷Σlts)에서
            # "계속 보던 사람"의 시청시간이 유실되지 않게 한다. ← dwell_sum_sec 를 별도로 보내는 이유.
            "avg_dwell": round(dwell_sum / lts, 3) if lts else 0.0,
            "dwell_sum": round(dwell_sum, 3),
        }

    def _is_ots(self, s):
        # OTS demographics 대상: 확정(min_hits+)되고 얼굴이 잡힌 통행자(성별/연령 추정 가능).
        return s.frames_seen >= self.settings.track_min_hits and s.face_visible

    def _is_gazer(self, s):
        # LTS 대상: 누적 응시시간이 임계(초) 이상인 응시자.
        return s.facing_sec >= self.settings.gaze_lts_min_sec

    def _finalized(self, now_sec, final, predicate, done_ids):
        # predicate 를 만족하는 track 중 이 구간에 "종료된" 것만 반환하고 done_ids 에 기록(중복 방지).
        #   종료 = final(세션 마감) 이거나, 마지막으로 보인 지 exit_grace 초 넘게 안 보임.
        #   한 번 센 track 은 재등장해도 다시 세지 않는다(분포 중복 방지).
        out = []
        grace = self.settings.rt_exit_grace_sec
        for s in self.registry.all():
            if s.track_id in done_ids or not predicate(s):
                continue
            stale = (s.last_seen_sec is not None and now_sec is not None
                     and (now_sec - s.last_seen_sec) > grace)
            if not (final or stale):            # 아직 보는 중 → 끝날 때까지 보류
                continue
            done_ids.add(s.track_id)
            out.append(s)
        return out

    def _zero(self):
        # 카운트/합의 최초 diff 기준(0). 첫 build 는 "0~현재"가 곧 첫 구간이 된다.
        return {"ots": 0, "lts": 0, "dwell_sum": 0.0}

    def _timestamp(self):
        # 메시지 timestamp 는 UTC ISO8601(Z) 고정. 지역시간 변환은 서버/클라이언트 몫.
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
