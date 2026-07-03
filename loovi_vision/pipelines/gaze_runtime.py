from loovi_vision.enrich.gaze import is_facing_smoothed
from loovi_vision.enrich.gaze_enricher import GazeEnricher
from loovi_vision.realtime.summary import SummaryAggregator
from loovi_vision.realtime.sqs_sender import SqsSender
from loovi_vision.analysis.gaze_sessions import analyze_gaze_sessions
from loovi_vision.pipelines.session_io import poses_path, path_text


class GazeRuntime:
    # 2차 런타임 묶음: head pose 보강(COLD) + 구간 summary(HOT) 비동기 SQS 전송 + 사후 분석.
    # 모든 시계열 시각은 elapsed 초(run 시작 기준)로 통일한다 (summary timestamp만 UTC).
    def __init__(self, settings, registry, run_id):
        self.settings = settings
        self.registry = registry
        self.poses_path = poses_path(settings, run_id)
        self.gaze = GazeEnricher(settings, registry, self.poses_path)
        self.summary = SummaryAggregator(settings, registry)   # 구간: 카운트+성별/연령+Attention
        self.sender = SqsSender(settings) if settings.rt_enable else None
        if self.sender:
            self.sender.start()
        self.last_summary = 0.0
        self.last_now = 0.0        # 마지막 tick 시각(종료 시 남은 부분 window 산출용)
        self.summary_seq = 0

    @property
    def on_face(self):
        # FaceEnricher.on_face 에 연결할 콜백.
        return self.gaze.observe_face

    @property
    def facing_now(self):
        return self.gaze.facing_now

    def tick(self, seen, now_sec):
        # 매 처리 프레임 호출. summary_interval(기본 5초)마다 구간 summary 전송.
        self.last_now = now_sec
        self._maybe_summary(now_sec)

    def _maybe_summary(self, now_sec):
        # summary_interval(기본 5초)이 지났을 때만 한 구간 summary 를 만들어 전송한다.
        # build 에 넘기는 두 번째 인자는 직전 전송 이후 실제 경과 초(interval_sec 로 기록됨).
        if now_sec - self.last_summary < self.settings.rt_summary_interval:
            return
        self.summary_seq += 1
        self._enqueue(self.summary.build(self.summary_seq, now_sec - self.last_summary))
        self.last_summary = now_sec

    def _enqueue(self, message):
        # sender 가 있을 때만(=realtime.enable) SQS 로 보낸다. 없으면 로컬 기록만 남는다.
        if self.sender:
            self.sender.enqueue(message)

    def current_gazers(self, now_sec):
        # 로컬 JSONL/overlay 용: 지금 이 순간 평활 응시 중인 인원(구간 누적 LTS 와는 다른, 순간값).
        return sum(1 for s in self.registry.all() if is_facing_smoothed(s, self.settings, now_sec))

    def gazing_ids(self, now_sec):
        # overlay 색 표시용: 현재 평활 응시 중인 track 집합.
        return {s.track_id for s in self.registry.all() if is_facing_smoothed(s, self.settings, now_sec)}

    def poses(self):
        # overlay 표시용: track별 최신 (yaw, pitch, roll) 원본 값 (3축 그리기 + 숫자).
        return self.gaze.last_pose

    def gaze_secs(self):
        # track_id별 누적 응시 시간(초) 조회. overlay 표시/외부 조회용.
        return {s.track_id: s.facing_sec for s in self.registry.all()}

    def lts_count(self):
        # LTS(라이브): 누적 응시 시간이 임계(초) 이상인 track 수 (실제 응시자).
        return sum(1 for s in self.registry.all()
                   if s.facing_sec >= self.settings.gaze_lts_min_sec)

    def row_extra(self, now_sec, ots):
        # 1초 JSONL 추가 필드: 평활 동시 응시 + OTS(통행) + LTS(응시자 누적).
        return {
            "concurrent_gazers": self.current_gazers(now_sec),
            "ots": ots,
            "lts": self.lts_count(),
        }

    def finalize(self):
        # 실행 종료 시: COLD raw 파일을 닫고, 남은 부분 window 를 마저 전송하고,
        # raw 에서 응시 구간/등급(Attention)을 사후 산출해 성능/전송 지표와 함께 반환한다.
        self.gaze.close()
        # 종료 시점에 아직 한 주기(기본 5초)가 안 찬 마지막 부분 window 도 버리지 않고 1건 전송한다
        # (측정 구간 종료 시점에 진행 중이던 응시도 누락 없이 반영).
        if self.sender and self.last_now > self.last_summary:
            self.summary_seq += 1
            self._enqueue(self.summary.build(self.summary_seq, self.last_now - self.last_summary))
        summary = analyze_gaze_sessions(self.poses_path, self.settings)
        summary["poses_path"] = path_text(self.poses_path)
        summary["headpose_avg_ms"] = self.gaze.avg_infer_ms()
        # OTS(통행, 확정 track) / LTS(누적 응시 >= 임계초). LTS는 COLD total_gaze_sec로 산출
        # → 임계값 바꿔 재분석해도 raw 재저장 없이 결과가 바뀐다(검증1).
        lts_min = self.settings.gaze_lts_min_sec
        summary["ots"] = sum(1 for s in self.registry.all() if s.frames_seen >= self.settings.track_min_hits)
        summary["lts"] = sum(1 for p in summary["per_track_gaze"] if p["total_gaze_sec"] >= lts_min)
        summary["lts_min_sec"] = lts_min
        if self.sender:
            self.sender.stop()
            summary.update(self.sender.metrics())
        return summary


def build_gaze_runtime(settings, enricher, run_id):
    # gaze.enable=false거나 face가 없으면 None (1차와 100% 동일). on_face 연결까지 처리.
    if not settings.gaze_enable:
        return None
    if not enricher:
        print("  WARNING: gaze.enable=true 이나 face.enable=false -> gaze 건너뜀 (얼굴 검출 필요)")
        return None
    runtime = GazeRuntime(settings, enricher.registry, run_id)
    enricher.on_face = runtime.on_face  # 1차 얼굴 crop을 head pose에 재사용
    return runtime


def attach_performance(manifest, proc_frames, wall_sec, max_dt, gaze_runtime):
    # 성능 로그(fps/headpose ms/전송 지표) + COLD 응시 분석 결과를 manifest에 병합한다.
    perf = {
        "avg_fps": round(proc_frames / wall_sec, 2),
        "min_fps": round(1.0 / max_dt, 2) if max_dt > 0 else 0.0,
        "processed_frames": proc_frames,
    }
    manifest["performance"] = perf
    if not gaze_runtime:
        return
    summary = gaze_runtime.finalize()
    perf["headpose_avg_ms"] = summary.pop("headpose_avg_ms", 0.0)
    for key in ("snapshots_sent", "send_failures", "snapshots_spilled",
                "buffer_max_depth", "last_send_latency_ms"):
        if key in summary:
            perf[key] = summary.pop(key)
    manifest.update(summary)
