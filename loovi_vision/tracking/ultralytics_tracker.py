import os
from types import SimpleNamespace

# ultralytics 가 ReID 모델(ONNX)을 로드할 때 requirement 자동설치로 CPU용 onnxruntime 를 깔아
# 기존 onnxruntime-gpu 와 충돌(→ GPU 비활성)하는 것을 막는다. ultralytics import 이전에 설정해야 한다.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from .base import TrackerDetections, TrackStore


class UltralyticsTracker(TrackStore):
    # Ultralytics의 BoT-SORT/ByteTrack을 현재 detector 출력에 연결하는 adapter.
    def __init__(self, settings):
        super().__init__(settings)
        # 내부 임계값은 하드코딩하지 않고 settings(config)에서 주입한다.
        args = SimpleNamespace(
            track_high_thresh=settings.track_high_thresh,
            track_low_thresh=settings.track_low_thresh,
            new_track_thresh=settings.new_track_thresh,
            track_buffer=settings.track_buffer,
            match_thresh=settings.track_match_thresh,
            fuse_score=True,
            gmc_method=settings.tracker_gmc_method,
            proximity_thresh=settings.track_proximity_thresh,
            appearance_thresh=settings.track_appearance_thresh,
            # body Re-ID: 켜면 외형 임베딩으로 끊긴 track 재결합을 시도(중복 통행 방지).
            # 우리는 커스텀 ONNX detector라 "auto"(YOLO 내부 특징)를 못 쓰고 전용 ReID 모델 파일이 필요하다.
            with_reid=settings.track_with_reid,
            model=(settings.track_reid_model if settings.track_with_reid else "auto"),
        )
        if settings.tracker_backend == "bytetrack":
            from ultralytics.trackers.byte_tracker import BYTETracker

            self.backend_name = "ByteTrack"
            self.backend = BYTETracker(args)
        else:
            from ultralytics.trackers.bot_sort import BOTSORT

            self.backend_name = "BoT-SORT"
            self.backend = BOTSORT(args)
        reid = f" +ReID({settings.track_reid_model})" if settings.track_with_reid else ""
        print(f"  OK Tracker [{self.backend_name}]{reid}")

    def update(self, detections, frame=None):
        # tracker 출력 row에서 track_id와 원본 detection index를 다시 연결한다.
        results = TrackerDetections.from_detections(detections)
        rows = self.backend.update(results, img=frame)
        active_ids = set()
        det_to_track = {}

        for row in rows:
            if len(row) < 8:
                continue
            x1, y1, x2, y2 = map(int, row[:4])
            track_id = int(row[4])
            det_idx = int(row[7])
            if det_idx < 0 or det_idx >= len(detections):
                continue

            active_ids.add(track_id)
            det_to_track[det_idx] = track_id
            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    "box": (x1, y1, x2, y2),
                    "missing": 0,
                    "hits": 1,
                    "counted": False,
                }
            else:
                track = self.tracks[track_id]
                track["box"] = self.smooth_box(track["box"], (x1, y1, x2, y2))
                track["missing"] = 0
                track["hits"] = track.get("hits", 1) + 1
            self.confirm_if_ready(self.tracks[track_id])

        # 이번 프레임에서 매칭되지 않은 track은 missing을 올리고 오래 사라지면 제거한다.
        for track_id, track in self.tracks.items():
            if track_id not in active_ids:
                track["missing"] = track.get("missing", 0) + 1
        self.prune_missing(self.settings.track_max_missing)

        return det_to_track
