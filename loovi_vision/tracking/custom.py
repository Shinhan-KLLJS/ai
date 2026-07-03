from .base import TrackStore


class CustomTracker(TrackStore):
    # 외부 tracker 초기화 실패 시 쓰는 단순 거리 기반 fallback tracker.
    def __init__(self, settings):
        super().__init__(settings)
        self.next_id = 1
        print("  OK Tracker [custom]")

    @staticmethod
    def _center(box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    @staticmethod
    def _distance(a, b):
        ac, bc = CustomTracker._center(a), CustomTracker._center(b)
        return ((ac[0] - bc[0]) ** 2 + (ac[1] - bc[1]) ** 2) ** 0.5

    def update(self, detections, frame=None):
        # 기존 track과 새 detection bbox 중심점 거리가 가장 가까운 쌍을 매칭한다.
        boxes = []
        for det in detections:
            x, y, w, h = det["bbox"]
            boxes.append((x, y, x + w, y + h))

        max_dist = self.settings.custom_match_max_dist   # 매칭 허용 최대 거리(px)
        det_to_track = {}
        used = set()
        for track_id, track in list(self.tracks.items()):
            best_idx, best_dist = -1, float("inf")
            for idx, box in enumerate(boxes):
                if idx in used:
                    continue
                dist = self._distance(track["box"], box)
                if dist < best_dist:
                    best_idx, best_dist = idx, dist
            if best_idx >= 0 and best_dist <= max_dist:
                used.add(best_idx)
                det_to_track[best_idx] = track_id
                track["box"] = self.smooth_box(track["box"], boxes[best_idx])
                track["missing"] = 0
                track["hits"] = track.get("hits", 1) + 1
                self.confirm_if_ready(track)
            else:
                track["missing"] += 1

        # 어떤 기존 track에도 매칭되지 않은 detection은 새 track으로 시작한다.
        for idx, box in enumerate(boxes):
            if idx in used:
                continue
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = {
                "box": box,
                "missing": 0,
                "hits": 1,
                "counted": False,
            }
            det_to_track[idx] = track_id
            self.confirm_if_ready(self.tracks[track_id])

        self.prune_missing(self.settings.track_max_missing)
        return det_to_track
