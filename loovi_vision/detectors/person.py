from pathlib import Path

import cv2
import numpy as np

from loovi_vision.runtime import onnx_providers, provider_label


class PersonDetector:
    # YOLO 계열 ONNX 모델을 직접 실행해 person bbox만 반환하는 detector.
    def __init__(self, settings):
        import onnxruntime as ort

        self.settings = settings
        self.model_path = Path(settings.person_onnx)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=onnx_providers(settings.enable_cuda),
        )
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.input_hw = (int(shape[2]), int(shape[3]))
        print(f"  OK PersonDetector [{self.model_path.stem} {self.input_hw[0]}px] [{provider_label(self.session)}]")

    def _letterbox(self, frame):
        # 원본 비율을 유지한 채 모델 입력 정사각형으로 맞추고 padding 정보를 보존한다.
        size = self.input_hw[0]
        h, w = frame.shape[:2]
        scale = size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (nw, nh))
        pad_h, pad_w = size - nh, size - nw
        top, left = pad_h // 2, pad_w // 2
        padded = cv2.copyMakeBorder(
            resized,
            top,
            pad_h - top,
            left,
            pad_w - left,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return padded, scale, top, left

    def _nms(self, boxes, scores):
        # confidence가 높은 bbox부터 겹치는 bbox를 제거한다.
        if not len(boxes):
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size:
            i = order[0]
            keep.append(i)
            ix1 = np.maximum(x1[i], x1[order[1:]])
            iy1 = np.maximum(y1[i], y1[order[1:]])
            ix2 = np.minimum(x2[i], x2[order[1:]])
            iy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
            order = order[np.where(iou <= self.settings.iou_threshold)[0] + 1]
        return keep

    def detect(self, frame):
        # ONNX 출력 좌표를 원본 프레임 좌표계로 복원한 뒤 사람 bbox 목록을 만든다.
        h, w = frame.shape[:2]
        padded, scale, pad_top, pad_left = self._letterbox(frame)
        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]

        # 현재 export된 YOLO11 ONNX는 [cx, cy, w, h, score...] 형태로 사용한다.
        out = self.session.run(None, {self.input_name: blob})[0]
        preds = out[0].T
        scores = preds[:, 4]
        mask = scores >= self.settings.person_conf_min
        preds = preds[mask]
        scores = scores[mask]
        if not len(preds):
            return []

        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = np.clip((cx - bw / 2 - pad_left) / scale, 0, w)
        y1 = np.clip((cy - bh / 2 - pad_top) / scale, 0, h)
        x2 = np.clip((cx + bw / 2 - pad_left) / scale, 0, w)
        y2 = np.clip((cy + bh / 2 - pad_top) / scale, 0, h)
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        frame_area = float(w * h)
        results = []
        for idx in self._nms(boxes, scores):
            x1i, y1i, x2i, y2i = map(int, boxes[idx])
            wi, hi = x2i - x1i, y2i - y1i
            # 너무 작거나 화면 대부분을 덮는 bbox는 현장 오탐으로 보고 제거한다.
            if hi < self.settings.person_min_height:
                continue
            area_ratio = (wi * hi) / frame_area
            if area_ratio < self.settings.person_min_area_ratio:
                continue
            if area_ratio > self.settings.person_max_area_ratio:
                continue
            results.append({
                "bbox": (x1i, y1i, wi, hi),
                "confidence": float(scores[idx]),
            })
        return results
