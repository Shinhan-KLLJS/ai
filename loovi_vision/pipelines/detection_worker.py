import threading
import time

from loovi_vision.pipelines.frame_render import collect_overlay_state, make_stats
from loovi_vision.pipelines.perf_meter import PerfMeter


# 첫 검출이 끝나기 전 프레임에 쓸 빈 스냅샷: tracker/batch를 건드리지 않고 0값 HUD만 그린다.
_EMPTY_STATS = {"now_count": 0, "peak_persons": 0, "active_tracks": 0, "unique_total": 0, "frame_count": 0}
_EMPTY_OVERLAY = {"face_boxes": None, "face_labels": None, "attended_ids": None,
                  "gazing_ids": None, "gazers": None, "lts": None, "poses": None, "gaze_secs": None}


class DetectionWorker:
    """검출→추적→(얼굴/gaze)→집계 처리를 담당한다.

    라이브(웹캠) 모드에선 별도 스레드로 돌려 메인 루프(캡처·표시)와 분리하고,
    영상 파일/동기 모드에선 process()를 메인이 직접 호출한다.
    메인 스레드는 snapshot()이 주는 불변 dict만 읽어 렌더링하므로 가변 상태에
    동시 접근하지 않는다(락은 입력 슬롯/스냅샷 교체에만 사용).
    """

    def __init__(self, settings, detector, tracker, enricher, gaze_runtime, batch, on_flush):
        self.settings = settings
        self.detector = detector
        self.tracker = tracker
        self.enricher = enricher
        self.gaze_runtime = gaze_runtime
        self.batch = batch
        self._on_flush = on_flush              # (row_dict) -> None : JSONL append 콜백
        self._meter = PerfMeter("detect-worker", enabled=settings.perf_log)
        self._lock = threading.Lock()
        self._snapshot = {"detections": [], "det_to_track": {},
                          "overlay": _EMPTY_OVERLAY, "stats": _EMPTY_STATS}
        self._input = None                     # (frame, now_sec, frame_id) : 최신 입력만 유지
        self._new_input = threading.Event()
        self._idle_ms = 0.0                    # 다음 입력을 기다린 시간(=submit 대기). 크면 worker가 놀았다는 뜻
        self._running = False
        self._thread = None
        # 성능 지표(manifest 기록용). 워커 스레드(또는 동기 호출자)만 갱신한다.
        self.proc_frames = 0       # detect+track을 수행한 전체 프레임(=트래킹 갱신 빈도)
        self.enrich_frames = 0     # 얼굴/포즈 보강까지 수행한 프레임(=보강 샘플 빈도)
        self.max_dt = 0.0
        self._last_proc_sec = 0.0
        self._last_enrich_sec = -1.0e9   # 마지막 보강 시각(초). 첫 프레임은 무조건 보강되게 큰 음수로 시작.

    def process(self, frame, now_sec, frame_id):
        # 한 프레임 동기 처리: 검출/추적/보강/집계 후 렌더용 스냅샷을 발행한다.
        # 단계별 시간을 재서 어느 단계(검출/추적/얼굴/gaze)가 병목인지 실측한다.
        t0 = time.perf_counter()
        detections = self.detector.detect(frame)
        t1 = time.perf_counter()
        det_to_track = self.tracker.update(detections, frame)
        t2 = time.perf_counter()
        seen = faced = None
        # 트래킹 분리: detect+track은 매 프레임, 무거운 얼굴/포즈 보강은 enrich_interval_sec 간격으로만 수행한다.
        # (얼굴 비율/성별·나이·gaze는 본래 표본 통계라 샘플링해도 의미가 유지되고, 카운트/OTS는 매 프레임 트래커가 갱신한다.)
        if self.enricher and self._enrich_due(now_sec):
            seen, faced = self.enricher.process(frame, detections, det_to_track, frame_id, now_sec)
            self._last_enrich_sec = now_sec
            self.enrich_frames += 1
        t3 = time.perf_counter()
        if self.gaze_runtime:
            # summary(기본 5초) 타이밍만 매 프레임 확인(가벼움). 실제 pose 관측은 위 보강 프레임에서만 일어난다.
            self.gaze_runtime.tick(seen, now_sec)
        t4 = time.perf_counter()
        self.batch.add(frame_id, detections, self.tracker, det_to_track, seen, faced)
        if self.batch.should_flush():
            extra = self.gaze_runtime.row_extra(now_sec, self.tracker.total_unique) if self.gaze_runtime else None
            self._on_flush(self.batch.flush(extra))
        # 오버레이/HUD 스냅샷은 처리 직후 같은 스레드에서 만들어 일관성을 보장한다.
        overlay = collect_overlay_state(self.enricher, self.gaze_runtime, now_sec)
        stats = make_stats(detections, self.batch, self.tracker, self.enricher, overlay["gazers"], overlay["lts"])
        with self._lock:
            self._snapshot = {"detections": detections, "det_to_track": det_to_track,
                              "overlay": overlay, "stats": stats}
        t5 = time.perf_counter()
        # post = batch+flush+overlay+stats+snapshot, total = process 1회 전체, idle = 직전 입력 대기.
        self._meter.add({"detect": (t1 - t0) * 1e3, "track": (t2 - t1) * 1e3,
                         "face": (t3 - t2) * 1e3, "gaze": (t4 - t3) * 1e3,
                         "post": (t5 - t4) * 1e3, "total": (t5 - t0) * 1e3,
                         "idle": self._idle_ms}, len(detections))
        self.proc_frames += 1
        dt = now_sec - self._last_proc_sec
        self.max_dt = max(self.max_dt, dt)
        self._last_proc_sec = now_sec

    def _enrich_due(self, now_sec):
        # enrich_interval_sec<=0 이면 매 프레임 보강(기존 동작). >0 이면 그 간격이 지났을 때만 보강한다.
        interval = self.settings.enrich_interval_sec
        return interval <= 0 or (now_sec - self._last_enrich_sec) >= interval

    def snapshot(self):
        with self._lock:
            return self._snapshot

    def flush_final(self, now_sec):
        # 종료 직전 남은 부분 batch를 저장한다(동기/스레드 공통, 루프 종료 후 1회 호출).
        if self.batch.frame_count:
            extra = self.gaze_runtime.row_extra(now_sec, self.tracker.total_unique) if self.gaze_runtime else None
            self._on_flush(self.batch.flush(extra))

    # --- 라이브(스레드) 모드 전용 ---
    def submit(self, frame, now_sec, frame_id):
        # 최신 입력만 유지: 처리가 밀리면 중간 프레임은 건너뛰고 항상 최신을 처리한다.
        with self._lock:
            self._input = (frame, now_sec, frame_id)
        self._new_input.set()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        # 새 입력이 올 때마다 최신 프레임 한 장을 처리한다. 타임아웃은 종료 확인용.
        while self._running:
            t_wait = time.perf_counter()
            if not self._new_input.wait(0.5):
                continue
            with self._lock:
                self._new_input.clear()
                item = self._input
            if item is not None:
                # 대기 시간 기록: 즉시 처리(≈0)면 compute-bound, 크면 입력(submit)-bound.
                self._idle_ms = (time.perf_counter() - t_wait) * 1e3
                self.process(*item)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
