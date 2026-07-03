from types import SimpleNamespace

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
            with_reid=False,
            model="auto",
        )
        if settings.tracker_backend == "bytetrack":
            from ultralytics.trackers.byte_tracker import BYTETracker

            self.backend_name = "ByteTrack"
            self.backend = BYTETracker(args)
        else:
            from ultralytics.trackers.bot_sort import BOTSORT

            self.backend_name = "BoT-SORT"
            self.backend = BOTSORT(args)
        print(f"  OK Tracker [{self.backend_name}]")

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
