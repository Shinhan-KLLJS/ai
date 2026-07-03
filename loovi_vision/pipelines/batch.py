import time
from datetime import datetime


class PersonOnlyBatch:
    # batch_sec 동안 처리된 detection/tracking 결과를 하나의 JSONL row로 집계한다.
    # face_enabled=True면 그 window의 주목(얼굴 보임) 지표를 추가로 기록한다.
    def __init__(self, settings, run_id, run_started_at, face_enabled=False):
        self.settings = settings
        self.run_id = run_id
        self.run_started_at = run_started_at
        self.face_enabled = face_enabled
        self.reset()

    def reset(self):
        self.t0 = time.time()
        self.frame_count = 0
        self.frame_detections = 0
        self.peak_persons = 0
        self.conf_sum = 0.0
        self.conf_n = 0
        self.unique_total = 0
        self.active_tracks = 0
        self.window_seen = set()    # 이 window에 보인 track_id
        self.window_face = set()    # 이 window에 얼굴이 보인 track_id
        self.samples = []

    def add(self, frame_id, detections, tracker, det_to_track, seen=None, faced=None):
        # 여기서 frame_count는 전체 카메라 프레임이 아니라 추론을 수행한 처리 프레임 수다.
        self.frame_count += 1
        self.frame_detections += len(detections)
        self.peak_persons = max(self.peak_persons, len(detections))
        self.unique_total = tracker.total_unique
        self.active_tracks = len(tracker.tracks)
        for det in detections:
            self.conf_sum += float(det.get("confidence", 0.0))
            self.conf_n += 1
        if seen:
            self.window_seen.update(seen)
        if faced:
            self.window_face.update(faced)
        if self.settings.save_frame_samples:
            # 검증 세션에서는 bbox/track_id를 row 안에 남겨 프레임 단위 분석에 쓴다.
            self.samples.append({
                "frame_id": frame_id,
                "person_count": len(detections),
                "active_tracks": self.active_tracks,
                "unique_total": self.unique_total,
                "detections": [
                    {
                        "bbox": list(map(int, det["bbox"])),
                        "confidence": round(float(det.get("confidence", 0.0)), 4),
                        "track_id": det_to_track.get(i),
                    }
                    for i, det in enumerate(detections)
                ],
            })

    def should_flush(self):
        return time.time() - self.t0 >= self.settings.batch_sec

    def flush(self, extra=None):
        # 현재 batch를 payload로 변환한 뒤 내부 누적값을 다음 window용으로 초기화한다.
        # extra: gaze 활성 시 concurrent_gazers 등 추가 필드 (없으면 1차와 동일).
        now = time.time()
        payload = {
            "run_id": self.run_id,
            "board_id": self.settings.board_id,
            "experiment": self.settings.experiment_name,
            "mode": "person_only",
            "window_start": datetime.fromtimestamp(self.t0).strftime("%Y-%m-%d %H:%M:%S"),
            "window_end": datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_start_sec": round(self.t0 - self.run_started_at, 3),
            "elapsed_end_sec": round(now - self.run_started_at, 3),
            "frame_count": self.frame_count,
            "frame_detections": self.frame_detections,
            "avg_persons_per_frame": round(self.frame_detections / self.frame_count, 2) if self.frame_count else 0.0,
            "peak_persons": self.peak_persons,
            "avg_confidence": round(self.conf_sum / self.conf_n, 4) if self.conf_n else None,
            "unique_total": self.unique_total,
            "active_tracks": self.active_tracks,
            "person_model": str(self.settings.person_onnx),
            "person_conf_min": self.settings.person_conf_min,
            "iou_threshold": self.settings.iou_threshold,
            "tracker_backend": self.settings.tracker_backend,
            "track_min_hits": self.settings.track_min_hits,
        }
        if self.face_enabled:
            # 통행(분모)=window_seen, 주목(분자)=window_face.
            seen = len(self.window_seen)
            faced = len(self.window_face)
            payload["persons_with_face"] = faced
            payload["face_visible_ratio"] = round(faced / seen, 4) if seen else 0.0
        if extra:
            payload.update(extra)
        if self.settings.save_frame_samples:
            payload["samples"] = self.samples
        self.reset()
        return payload
