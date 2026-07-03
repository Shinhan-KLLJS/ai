import numpy as np


class TrackerDetections:
    # Ultralytics tracker가 기대하는 results-like 객체의 최소 인터페이스.
    def __init__(self, xywh, conf, cls, xyxy=None):
        self.xywh = xywh.astype(np.float32)
        if xyxy is None:
            xyxy = np.empty((0, 4), dtype=np.float32)
            if len(self.xywh):
                xyxy = self.xywh.copy()
                xyxy[:, 0] = self.xywh[:, 0] - self.xywh[:, 2] / 2
                xyxy[:, 1] = self.xywh[:, 1] - self.xywh[:, 3] / 2
                xyxy[:, 2] = self.xywh[:, 0] + self.xywh[:, 2] / 2
                xyxy[:, 3] = self.xywh[:, 1] + self.xywh[:, 3] / 2
        self.xyxy = xyxy.astype(np.float32)
        self.conf = conf.astype(np.float32)
        self.cls = cls.astype(np.float32)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return TrackerDetections(self.xywh[idx], self.conf[idx], self.cls[idx], self.xyxy[idx])

    @staticmethod
    def from_detections(detections):
        # pipeline의 [x, y, w, h] bbox를 tracker가 쓰는 center xywh 형식으로 변환한다.
        xywh, conf, cls = [], [], []
        for det in detections:
            x, y, w, h = det["bbox"]
            xywh.append([x + w / 2, y + h / 2, w, h])
            conf.append(det.get("confidence", 1.0))
            cls.append(0.0)
        if not xywh:
            return TrackerDetections(
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        return TrackerDetections(
            np.asarray(xywh, dtype=np.float32),
            np.asarray(conf, dtype=np.float32),
            np.asarray(cls, dtype=np.float32),
        )


class TrackStore:
    # 모든 tracker backend가 공유하는 track 저장소와 unique count 로직.
    def __init__(self, settings):
        self.settings = settings
        self.tracks = {}
        self.total_unique = 0

    def smooth_box(self, old_box, new_box):
        # 화면 overlay가 떨리지 않도록 이전 bbox와 새 bbox를 지수평활한다.
        a = self.settings.track_box_smooth_alpha
        return tuple(int(round(a * old_box[i] + (1 - a) * new_box[i])) for i in range(4))

    def confirm_if_ready(self, track):
        # 일정 hit 수 이상 관측된 track만 고유 방문자로 확정한다.
        if track.get("counted"):
            return
        if track.get("hits", 0) < self.settings.track_min_hits:
            return
        track["counted"] = True
        self.total_unique += 1

    def prune_missing(self, max_missing):
        # 오래 매칭되지 않은 track을 제거한다(공통 로직).
        for track_id, track in list(self.tracks.items()):
            if track["missing"] > max_missing:
                del self.tracks[track_id]
