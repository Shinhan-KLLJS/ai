"""
AdScope v7 — Phase A: Person bbox 기반 트래킹

v6 대비 변경:
  - PersonDetector 추가: YOLOv8n (COCO) class=0 person 감지
  - UniquePersonTracker 입력을 face bbox → person bbox로 교체
    → 전신/상체 bbox는 고개 방향과 무관하게 안정적
  - YOLOv8n-face는 보조 역할 (시선 판단 + 성별/연령 추정)
  - face → person 연결: face 중심점이 person bbox 내부에 있으면 연결
  - 화면에 person bbox(큰 박스) + face bbox(작은 박스) 이중 표시
"""

import sys
import cv2
import numpy as np
import json
import time
import math
import zipfile
import io
import urllib.request
from datetime import datetime
from pathlib import Path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ① 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Config:
    # 판정 기준
    YAW_THRESHOLD    = 30
    PITCH_THRESHOLD  = 25
    FACE_CONF_MIN    = 0.45
    PERSON_CONF_MIN  = 0.40
    IOU_THRESHOLD    = 0.45

    # 처리 주기
    PROCESS_EVERY_N  = 2

    # 카메라
    CAMERA_ID        = 0
    FRAME_W          = 1280
    FRAME_H          = 720

    # 배치 저장
    BATCH_SEC        = 15

    # 모델 경로
    MODEL_DIR        = Path("models")
    PERSON_ONNX      = Path("models") / "yolov8n.onnx"
    YOLO_ONNX        = Path("models") / "yolov8n-face.onnx"
    POSE_ONNX        = Path("models") / "sixdrepnet.onnx"
    GENDER_AGE_ONNX  = Path("models") / "genderage.onnx"

    # 모델 최소 크기 (bytes)
    PERSON_MIN_SIZE     = 5_000_000
    YOLO_MIN_SIZE       = 3_000_000
    POSE_MIN_SIZE       = 100_000
    GENDER_AGE_MIN_SIZE = 500_000

    # 감지 최소 크기
    PERSON_MIN_HEIGHT = 40    # px — 이 높이 미만 person bbox는 무시
    MIN_FACE_SIZE     = 8     # px — 이 크기 미만 face bbox는 무시

    # YOLO 입력 해상도
    YOLO_INPUT_SIZE  = 640    # person 감지는 640으로 충분 (전신 크기)
    FACE_INPUT_SIZE  = 960    # face 감지는 고해상도 유지

    # 성별/연령
    GENDER_AGE_MIN_FACE = 25
    GENDER_AGE_REFRESH  = 30


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ② 모델 파일 확인 + 다운로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOWNLOAD_URLS = {
    "person": [
        "https://huggingface.co/Ultralytics/assets/resolve/main/yolov8n.onnx",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx",
    ],
    "yolo": [
        "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.onnx",
        "https://huggingface.co/Ultralytics/assets/resolve/main/yolov8n-face.onnx",
    ],
    "pose": [
        "https://github.com/thohemp/6DRepNet360/releases/download/v1.0.0/sixdrepnet360_Mobilenet_nobn_new.onnx",
    ],
    "gender_age": [
        "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/buffalo_l/genderage.onnx",
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    ],
}

def _download_file(url, dest_path, desc):
    print(f"  Downloading {desc}... ", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"done ({dest_path.stat().st_size // 1024}KB)")
        return True
    except Exception as e:
        print(f"failed ({e})")
        if dest_path.exists():
            dest_path.unlink(missing_ok=True)
        return False

def _try_extract_from_zip(zip_url, inner_filename, dest_path):
    print(f"  Downloading ZIP... ", end="", flush=True)
    try:
        data = urllib.request.urlopen(zip_url, timeout=120).read()
        print("done")
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                if Path(name).name.lower() == inner_filename.lower():
                    raw = z.read(name)
                    dest_path.write_bytes(raw)
                    print(f"  Extracted: {name} ({len(raw)//1024}KB)")
                    return True
        print(f"  Not found in ZIP: {inner_filename}")
    except Exception as e:
        print(f"  ZIP failed: {e}")
    return False

def _try_export_yolov8n():
    """ultralytics 설치된 경우 yolov8n.onnx 직접 익스포트."""
    try:
        from ultralytics import YOLO
        print("  Exporting yolov8n.onnx via ultralytics...")
        m = YOLO("yolov8n.pt")
        m.export(format="onnx", opset=12, simplify=True)
        exported = Path("yolov8n.onnx")
        if exported.exists():
            dest = Config.PERSON_ONNX
            dest.parent.mkdir(exist_ok=True)
            exported.rename(dest)
            print(f"  Exported -> {dest} ({dest.stat().st_size // 1024}KB)")
            return True
    except Exception as e:
        print(f"  Export failed: {e}")
    return False

def check_and_download_models():
    Config.MODEL_DIR.mkdir(exist_ok=True)
    results = {"person": False, "yolo": False, "pose": False, "gender_age": False}

    specs = [
        ("person",     Config.PERSON_ONNX,     Config.PERSON_MIN_SIZE,     "YOLOv8n person ONNX"),
        ("yolo",       Config.YOLO_ONNX,        Config.YOLO_MIN_SIZE,       "YOLOv8n-face ONNX"),
        ("pose",       Config.POSE_ONNX,         Config.POSE_MIN_SIZE,       "6DRepNet360 ONNX"),
        ("gender_age", Config.GENDER_AGE_ONNX,   Config.GENDER_AGE_MIN_SIZE, "InsightFace genderage ONNX"),
    ]

    for key, path, min_size, desc in specs:
        if path.exists() and path.stat().st_size >= min_size:
            print(f"  OK {desc} ({path.stat().st_size // 1024}KB)")
            results[key] = True
            continue

        downloaded = False
        for url in DOWNLOAD_URLS[key]:
            if url.endswith(".zip"):
                if _try_extract_from_zip(url, path.name, path):
                    if path.exists() and path.stat().st_size >= min_size:
                        results[key] = True
                        downloaded = True
                        break
            else:
                if _download_file(url, path, desc):
                    if path.exists() and path.stat().st_size >= min_size:
                        results[key] = True
                        downloaded = True
                        break
                    if path.exists():
                        path.unlink(missing_ok=True)

        # person 모델은 ultralytics export로 fallback
        if not downloaded and key == "person":
            if _try_export_yolov8n():
                if path.exists() and path.stat().st_size >= min_size:
                    results[key] = True
                    downloaded = True

        if not downloaded:
            print(f"  WARNING: {desc} not available")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ③ YOLOv8n Person 감지기 (Phase A 핵심)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PersonDetector:
    """
    YOLOv8n COCO class=0 (person) 전용 감지기.
    출력 shape: (1, 84, 8400) — [cx,cy,w,h, 80-class-scores...]
    person score = out[0, 4, :]
    """

    def __init__(self):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            str(Config.PERSON_ONNX), sess_options=opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name
        inp_shape = self.sess.get_inputs()[0].shape  # e.g. [1,3,640,640]
        self.input_hw = (int(inp_shape[2]), int(inp_shape[3]))  # (H, W)
        used = self.sess.get_providers()[0].replace("ExecutionProvider", "")
        print(f"  OK PersonDetector [YOLOv8n {self.input_hw[0]}px] [{used}]")

    def _letterbox(self, frame):
        size = self.input_hw[0]
        h, w = frame.shape[:2]
        scale = size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (nw, nh))
        pad_h, pad_w = size - nh, size - nw
        top, left = pad_h // 2, pad_w // 2
        padded = cv2.copyMakeBorder(
            resized, top, pad_h - top, left, pad_w - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded, scale, top, left

    def _nms(self, boxes, scores):
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
            order = order[np.where(iou <= Config.IOU_THRESHOLD)[0] + 1]
        return keep

    def detect(self, frame):
        h, w = frame.shape[:2]
        padded, scale, pt, pl = self._letterbox(frame)
        blob = padded.astype(np.float32) / 255.
        blob = blob.transpose(2, 0, 1)[np.newaxis]

        out = self.sess.run(None, {self.input_name: blob})[0]
        # out: (1, 84, 8400) → transpose → (8400, 84)
        preds = out[0].T

        # COCO class 0 = person (index 4 = first class score)
        person_scores = preds[:, 4]
        mask = person_scores >= Config.PERSON_CONF_MIN
        preds = preds[mask]
        person_scores = person_scores[mask]

        if not len(preds):
            return []

        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1o = np.clip((cx - bw / 2 - pl) / scale, 0, w)
        y1o = np.clip((cy - bh / 2 - pt) / scale, 0, h)
        x2o = np.clip((cx + bw / 2 - pl) / scale, 0, w)
        y2o = np.clip((cy + bh / 2 - pt) / scale, 0, h)
        boxes = np.stack([x1o, y1o, x2o, y2o], axis=1)

        keep = self._nms(boxes, person_scores)

        results = []
        for idx in keep:
            x1i, y1i, x2i, y2i = map(int, boxes[idx])
            bw_i, bh_i = x2i - x1i, y2i - y1i
            if bh_i < Config.PERSON_MIN_HEIGHT:
                continue
            results.append({
                "bbox":       (x1i, y1i, bw_i, bh_i),
                "confidence": float(person_scores[idx]),
            })
        return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ④ YOLOv8-face 감지기 (보조 — 시선·성별/연령용)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class YOLOFaceDetector:
    def __init__(self):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            str(Config.YOLO_ONNX), sess_options=opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name
        used = self.sess.get_providers()[0].replace("ExecutionProvider", "")
        print(f"  OK YOLOv8n-face [{used}] (size={Config.FACE_INPUT_SIZE})")

    def _letterbox(self, frame):
        size = Config.FACE_INPUT_SIZE
        h, w = frame.shape[:2]
        scale = size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (nw, nh))
        pad_h, pad_w = size - nh, size - nw
        top, left = pad_h // 2, pad_w // 2
        padded = cv2.copyMakeBorder(
            resized, top, pad_h - top, left, pad_w - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded, scale, top, left

    def _nms(self, boxes, scores):
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
            order = order[np.where(iou <= Config.IOU_THRESHOLD)[0] + 1]
        return keep

    def detect(self, frame):
        h, w = frame.shape[:2]
        padded, scale, pt, pl = self._letterbox(frame)
        blob = padded.astype(np.float32) / 255.
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        out = self.sess.run(None, {self.input_name: blob})[0]
        preds = out[0].T
        mask = preds[:, 4] >= Config.FACE_CONF_MIN
        preds = preds[mask]
        if not len(preds):
            return []

        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1o = np.clip((cx - bw / 2 - pl) / scale, 0, w)
        y1o = np.clip((cy - bh / 2 - pt) / scale, 0, h)
        x2o = np.clip((cx + bw / 2 - pl) / scale, 0, w)
        y2o = np.clip((cy + bh / 2 - pt) / scale, 0, h)
        boxes = np.stack([x1o, y1o, x2o, y2o], axis=1)
        scores = preds[:, 4]
        keep = self._nms(boxes, scores)

        results = []
        for idx in keep:
            x1i, y1i, x2i, y2i = map(int, boxes[idx])
            x1i, y1i = max(0, x1i), max(0, y1i)
            x2i, y2i = min(w, x2i), min(h, y2i)
            bw_i, bh_i = x2i - x1i, y2i - y1i
            if bw_i < Config.MIN_FACE_SIZE or bh_i < Config.MIN_FACE_SIZE:
                continue
            p = 6
            crop = frame[max(0, y1i - p):min(h, y2i + p),
                         max(0, x1i - p):min(w, x2i + p)]
            results.append({
                "bbox":       (x1i, y1i, bw_i, bh_i),
                "confidence": float(scores[idx]),
                "face_crop":  crop,
            })
        return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⑤ Face → Person 연결
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def associate_faces_to_persons(person_dets, face_dets, face_poses):
    """
    각 face의 중심점이 어느 person bbox 내부에 있는지 확인해 연결.

    반환:
      person_poses   : list[(yaw,pitch,roll,look)] — person_dets 길이와 동일
      person_to_face : dict {person_idx: face_idx}
    """
    default_pose = (0., 0., 0., False)
    person_poses   = [default_pose] * len(person_dets)
    person_to_face = {}

    if not face_dets or not person_dets:
        return person_poses, person_to_face

    for fi, fdet in enumerate(face_dets):
        fx, fy, fw, fh = fdet["bbox"]
        fcx = fx + fw / 2
        fcy = fy + fh / 2

        for pi, pdet in enumerate(person_dets):
            px, py, pw, ph = pdet["bbox"]
            if px <= fcx <= px + pw and py <= fcy <= py + ph:
                # 이미 이 person에 연결된 face가 있으면 confidence 높은 것 유지
                prev_fi = person_to_face.get(pi)
                if prev_fi is None or fdet["confidence"] > face_dets[prev_fi]["confidence"]:
                    person_to_face[pi] = fi
                    person_poses[pi]   = face_poses[fi]
                break  # face는 가장 잘 맞는 person 하나에만 연결

    return person_poses, person_to_face


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⑥ 6DRepNet Head Pose
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SixDRepNetPose:
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            str(Config.POSE_ONNX), sess_options=opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name
        out_shape = self.sess.get_outputs()[0].shape
        print(f"  OK SixDRepNet (output: {out_shape})")

    def _preprocess(self, face_img):
        resized = cv2.resize(face_img, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normed = (rgb - self.MEAN) / self.STD
        return normed.transpose(2, 0, 1)[np.newaxis]

    def _rot_to_euler(self, R):
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            pitch = math.atan2(-R[2, 0], sy)
            yaw   = math.atan2(R[1, 0] / math.cos(pitch), R[0, 0] / math.cos(pitch))
            roll  = math.atan2(R[2, 1], R[2, 2])
        else:
            pitch = math.atan2(-R[2, 0], sy)
            yaw, roll = 0., math.atan2(-R[1, 2], R[1, 1])
        return (round(math.degrees(yaw), 1),
                round(math.degrees(pitch), 1),
                round(math.degrees(roll), 1))

    def estimate(self, face_crop, bbox=None, frame_shape=None):
        if face_crop is None or face_crop.size == 0:
            return 0., 0., 0.
        try:
            out = self.sess.run(None, {self.input_name: self._preprocess(face_crop)})[0]
            return self._rot_to_euler(out[0])
        except Exception:
            return 0., 0., 0.


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⑦ 폴백 Head Pose (solvePnP)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FallbackPose:
    FACE_3D = np.array([
        [0.,    0.,    0.  ],
        [0.,  -330., -65.  ],
        [-225., 170., -135.],
        [225.,  170., -135.],
        [-150., -150., -125.],
        [150., -150., -125.],
    ], dtype=np.float64)

    def __init__(self):
        print("  WARNING: Using solvePnP fallback (less accurate)")

    def estimate(self, face_crop, bbox=None, frame_shape=None):
        if bbox is None or frame_shape is None:
            return 0., 0., 0.
        x, y, w, h = bbox
        fh, fw = frame_shape[:2]
        landmarks_2d = np.array([
            [x + w * 0.50, y + h * 0.40],
            [x + w * 0.50, y + h * 0.88],
            [x + w * 0.22, y + h * 0.28],
            [x + w * 0.78, y + h * 0.28],
            [x + w * 0.35, y + h * 0.72],
            [x + w * 0.65, y + h * 0.72],
        ], dtype=np.float64)
        focal = fw
        cam   = np.array([[focal, 0, fw / 2], [0, focal, fh / 2], [0, 0, 1]], dtype=np.float64)
        dist  = np.zeros((4, 1))
        ok, rvec, _ = cv2.solvePnP(self.FACE_3D, landmarks_2d, cam, dist,
                                    flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return 0., 0., 0.
        rmat, _ = cv2.Rodrigues(rvec)
        pitch = math.degrees(math.atan2(-rmat[2, 0],
                                         math.sqrt(rmat[2, 1] ** 2 + rmat[2, 2] ** 2)))
        yaw   = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
        roll  = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
        return round(yaw, 1), round(pitch, 1), round(roll, 1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⑧ 성별/연령 추정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class GenderAgeEstimator:
    """
    antelopev2 genderage.onnx (96x96)
    출력: [male_logit, female_logit, age/100]
    """

    def __init__(self):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            str(Config.GENDER_AGE_ONNX), sess_options=opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name
        inp   = self.sess.get_inputs()[0]
        out   = self.sess.get_outputs()[0]
        shape = inp.shape
        self.input_size = int(shape[3]) if shape[3] not in (None, "None") else 112
        used = self.sess.get_providers()[0].replace("ExecutionProvider", "")
        print(f"  OK GenderAge [{used}] ({self.input_size}x{self.input_size}, out:{out.shape})")

    def _preprocess(self, face_img):
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size))
        normed = (resized.astype(np.float32) - 127.5) / 127.5
        return normed.transpose(2, 0, 1)[np.newaxis]

    def estimate(self, face_crop):
        if face_crop is None or face_crop.size == 0:
            return "?", None
        h, w = face_crop.shape[:2]
        if w < Config.GENDER_AGE_MIN_FACE or h < Config.GENDER_AGE_MIN_FACE:
            return "?", None
        try:
            blob = self._preprocess(face_crop)
            pred = self.sess.run(None, {self.input_name: blob})[0][0]
            if len(pred) >= 3:
                gender = "M" if float(pred[0]) > float(pred[1]) else "F"
                age    = max(1, min(99, int(round(float(pred[2]) * 100))))
            elif len(pred) == 2:
                gender = "M" if float(pred[0]) > float(pred[1]) else "F"
                age    = None
            else:
                return "?", None
            return gender, age
        except Exception:
            return "?", None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⑨ 주목 판정 + Attention Score
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AttentionEngine:
    def is_looking(self, yaw, pitch):
        return (abs(yaw)   < Config.YAW_THRESHOLD and
                abs(pitch) < Config.PITCH_THRESHOLD)

    def score(self, yaw, pitch):
        if not self.is_looking(yaw, pitch):
            return 0.0
        ys = max(0., 1 - abs(yaw)   / Config.YAW_THRESHOLD)
        ps = max(0., 1 - abs(pitch) / Config.PITCH_THRESHOLD)
        return round(ys * ps * 100, 1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⑩ 고유 인원 트래커
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class UniquePersonTracker:
    """
    IoU + 중심점 거리 기반 트래킹.
    v7: 입력이 person bbox (전신) → 훨씬 안정적.
    """
    IOU_THRESH        = 0.30
    MAX_MISSING       = 45    # ~3초 (30fps / PROCESS_EVERY_N=2 기준)
    CENTROID_FALLBACK = 200   # person bbox는 face보다 크므로 fallback 거리도 키움

    def __init__(self):
        self.tracks        = {}
        self.next_id       = 1
        self.total_unique  = 0
        self.looked_unique = 0

    def attention_rate(self):
        if self.total_unique == 0:
            return 0.
        return round(self.looked_unique / self.total_unique * 100, 1)

    @staticmethod
    def _iou(b1, b2):
        ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter + 1e-8)

    @staticmethod
    def _centroid_dist(b1, b2):
        return math.sqrt(((b1[0]+b1[2])/2 - (b2[0]+b2[2])/2)**2 +
                         ((b1[1]+b1[3])/2 - (b2[1]+b2[3])/2)**2)

    def update(self, detections, pose_results, frame_n=0):
        """
        detections: person_dets (bbox = x,y,w,h)
        반환: (new_count, det_to_track {det_idx: track_id})
        """
        boxes = []
        for det in detections:
            x, y, w, h = det["bbox"]
            boxes.append((x, y, x + w, y + h))

        matched_tracks = set()
        matched_boxes  = set()
        det_to_track   = {}

        for tid, track in self.tracks.items():
            best_iou, best_i = 0, -1
            best_dist, best_dist_i = float("inf"), -1
            for i, box in enumerate(boxes):
                if i in matched_boxes:
                    continue
                s = self._iou(track["box"], box)
                d = self._centroid_dist(track["box"], box)
                if s > best_iou:
                    best_iou, best_i = s, i
                if d < best_dist:
                    best_dist, best_dist_i = d, i

            if best_iou >= self.IOU_THRESH and best_i >= 0:
                pass
            elif best_dist <= self.CENTROID_FALLBACK and best_dist_i >= 0:
                best_i = best_dist_i

            if best_i >= 0 and (best_iou >= self.IOU_THRESH or
                                 best_dist <= self.CENTROID_FALLBACK):
                track["box"]     = boxes[best_i]
                track["missing"] = 0
                if (not track["looked"] and
                        best_i < len(pose_results) and
                        pose_results[best_i][3]):
                    track["looked"] = True
                    self.looked_unique += 1
                matched_tracks.add(tid)
                matched_boxes.add(best_i)
                det_to_track[best_i] = tid
            else:
                track["missing"] += 1

        new_count = 0
        for i, box in enumerate(boxes):
            if i not in matched_boxes:
                looking = pose_results[i][3] if i < len(pose_results) else False
                self.tracks[self.next_id] = {
                    "box":      box,
                    "missing":  0,
                    "looked":   looking,
                    "gender":   None,
                    "age":      None,
                    "ga_frame": -999,
                }
                det_to_track[i] = self.next_id
                if looking:
                    self.looked_unique += 1
                self.next_id     += 1
                self.total_unique += 1
                new_count         += 1

        # 오래 사라진 트랙 제거
        dead = [tid for tid, t in self.tracks.items() if t["missing"] > self.MAX_MISSING]
        for tid in dead:
            del self.tracks[tid]

        return new_count, det_to_track


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⑪ 배치 집계
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class BatchAggregator:
    def __init__(self, board_id):
        self.board_id = board_id
        self.reset()

    def reset(self):
        self.t0             = time.time()
        self.total          = 0
        self.looking        = 0
        self.score_sum      = 0.
        self.score_n        = 0
        self.peak           = 0
        self.frame_count    = 0
        self.unique_total   = 0
        self.unique_looking = 0
        self.demographics_seen = set()
        self.demographics      = []

    def add(self, n_total, n_looking, scores, tracker=None):
        self.total      += n_total
        self.looking    += n_looking
        self.peak        = max(self.peak, n_total)
        self.score_sum  += sum(scores)
        self.score_n    += len(scores)
        self.frame_count += 1
        if tracker:
            self.unique_total   = tracker.total_unique
            self.unique_looking = tracker.looked_unique
            for tid, track in tracker.tracks.items():
                if tid not in self.demographics_seen and track.get("gender") not in (None, "?"):
                    self.demographics_seen.add(tid)
                    self.demographics.append({
                        "gender": track["gender"],
                        "age":    track.get("age"),
                    })

    def should_flush(self):
        return time.time() - self.t0 >= Config.BATCH_SEC

    def flush(self):
        attn       = round(self.looking / self.total * 100, 1) if self.total else 0.
        avg_score  = round(self.score_sum / self.score_n, 1)   if self.score_n else 0.
        unique_attn = (round(self.unique_looking / self.unique_total * 100, 1)
                       if self.unique_total > 0 else 0.)

        male_count   = sum(1 for d in self.demographics if d["gender"] == "M")
        female_count = sum(1 for d in self.demographics if d["gender"] == "F")
        ages  = [d["age"] for d in self.demographics if d["age"] is not None]
        avg_age = round(sum(ages) / len(ages), 1) if ages else None
        age_dist = {"10s": 0, "20s": 0, "30s": 0, "40s": 0, "50plus": 0}
        for a in ages:
            if   a < 20: age_dist["10s"]    += 1
            elif a < 30: age_dist["20s"]    += 1
            elif a < 40: age_dist["30s"]    += 1
            elif a < 50: age_dist["40s"]    += 1
            else:        age_dist["50plus"] += 1

        p = {
            "board_id":              self.board_id,
            "window_start":          datetime.fromtimestamp(self.t0).strftime("%Y-%m-%d %H:%M:%S"),
            "window_end":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "unique_total":          self.unique_total,
            "unique_looked":         self.unique_looking,
            "unique_attention_rate": unique_attn,
            "unique_male":           male_count,
            "unique_female":         female_count,
            "avg_age":               avg_age,
            "age_distribution":      age_dist,
            "frame_detections":      self.total,
            "frame_looking":         self.looking,
            "frame_attention_rate":  attn,
            "avg_attention_score":   avg_score,
            "peak_persons":          self.peak,
            "frame_count":           self.frame_count,
        }
        self.reset()
        return p


def save_to_db(payload):
    print("\n" + "=" * 62)
    print(f"  [{payload['window_start']}] Batch summary")
    skip = {"window_start", "window_end", "board_id", "age_distribution"}
    for k, v in payload.items():
        if k not in skip:
            print(f"     {k:<28}: {v}")
    if payload.get("age_distribution"):
        print(f"     {'age_distribution':<28}: {payload['age_distribution']}")
    print("=" * 62)
    with open("data_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⑫ 시각화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def draw_axis(frame, bbox, yaw, pitch, roll):
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2
    sz = max(int(w * 0.25), 20)  # person bbox 기준으로 크기 조정
    yr, pr, rr = math.radians(yaw), math.radians(pitch), math.radians(roll)
    cv2.arrowedLine(frame, (cx, cy),
                    (cx + int(sz * math.cos(yr) * math.cos(pr)),
                     cy - int(sz * math.sin(pr))),
                    (0, 0, 255), 2, tipLength=0.3)
    cv2.arrowedLine(frame, (cx, cy),
                    (cx + int(sz * 0.55 * math.cos(yr + math.pi / 2)),
                     cy + int(sz * 0.55 * math.sin(rr))),
                    (0, 200, 0), 1, tipLength=0.3)
    cv2.arrowedLine(frame, (cx, cy),
                    (cx - int(sz * 0.55 * math.sin(yr) * math.sin(pr)),
                     cy - int(sz * 0.55 * math.cos(pr))),
                    (255, 100, 0), 1, tipLength=0.3)


def draw(frame, person_dets, face_dets, person_poses, person_scores,
         stats, track_infos=None, person_to_face=None):
    hf, wf = frame.shape[:2]
    person_to_face = person_to_face or {}

    # ── Person bbox (주 박스) ──
    for i, (pdet, pose) in enumerate(zip(person_dets, person_poses)):
        x, y, w, h = pdet["bbox"]
        yaw, pitch, roll, look = pose
        sc    = person_scores[i] if i < len(person_scores) else 0.
        color = (0, min(255, int(sc * 2.55 + 80)), 150) if look else (110, 70, 200)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # 시선 화살표: person bbox 상단 중앙에 표시
        if i in person_to_face:
            draw_axis(frame, (x, y, w, h), yaw, pitch, roll)

        # 레이블
        ti = (track_infos[i] if track_infos and i < len(track_infos) else {}) or {}
        ga_str = ""
        if ti.get("gender") not in (None, "?"):
            age_part = f"/{ti['age']}" if ti.get("age") is not None else ""
            ga_str = f"  {ti['gender']}{age_part}"

        lines = [
            f"{'LOOK' if look else 'PASS'} {sc:.0f}pt{ga_str}",
            f"yaw:{yaw:+.0f} pitch:{pitch:+.0f}",
        ]
        for j, txt in enumerate(lines):
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            by = y - (j + 1) * (th + 7)
            if by < 0:
                continue
            cv2.rectangle(frame, (x, by - 2), (x + tw + 4, by + th + 2), color, -1)
            cv2.putText(frame, txt, (x + 2, by + th),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)

    # ── Face bbox (보조 박스 — 얇은 테두리) ──
    for fdet in face_dets:
        fx, fy, fw, fh = fdet["bbox"]
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (200, 200, 60), 1)

    # ── 반투명 통계 패널 ──
    ov = frame.copy()
    cv2.rectangle(ov, (8, 8), (360, 215), (10, 10, 22), -1)
    cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (8, 8), (360, 215), (80, 60, 155), 1)

    # 축 범례
    for li, (ltxt, lcol) in enumerate([("Z(nose)", (0, 0, 255)),
                                        ("X(yaw)",  (0, 200, 0)),
                                        ("Y(pitch)", (255, 100, 0))]):
        cv2.circle(frame, (wf - 120, 15 + li * 18), 4, lcol, -1)
        cv2.putText(frame, ltxt, (wf - 112, 19 + li * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, lcol, 1)

    m_cnt = stats.get("male_count", 0)
    f_cnt = stats.get("female_count", 0)
    avg_a = stats.get("avg_age")
    age_str  = f"{avg_a:.0f}y" if avg_a is not None else "--y"
    demo_str = f"M:{m_cnt} F:{f_cnt}  AvgAge:{age_str}"

    lines_panel = [
        (f"AdScope v7  [Person+Face]",                        (200, 175, 255)),
        (f"Now    : {stats['total']:>3} active",              (225, 225, 255)),
        (f"Looking: {stats['looking']:>3} (now)",             (80, 255, 150)),
        (f"-----------------------------",                     (60, 60, 80)),
        (f"Unique : {stats['unique_total']:>3} total",        (255, 220, 100)),
        (f"Looked : {stats['unique_looked']:>3}",             (255, 180, 80)),
        (f"Attn%  : {stats['unique_attn']:>5.1f}%",          (255, 200, 60)),
        (f"-----------------------------",                     (60, 60, 80)),
        (demo_str,                                             (160, 220, 255)),
        (f"Batch  : {stats['elapsed']:>4.0f}s / {Config.BATCH_SEC}s", (150, 155, 190)),
    ]
    for idx, (txt, col) in enumerate(lines_panel):
        cv2.putText(frame, txt, (16, 28 + idx * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1)

    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, ts, (wf - 215, hf - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 200, 70), 1)
    return frame


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⑬ 실시간 모드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_live(board_id="board_gangnam_01"):
    print(f"\n{'=' * 62}")
    print(f"  AdScope v7  |  Person+Face tracking")
    print(f"  Person model: {Config.PERSON_ONNX.name}")
    print(f"  Quit: q")
    print(f"{'=' * 62}\n")

    print("[ Model check ]")
    dl = check_and_download_models()
    print()

    # Person 감지기
    if dl["person"]:
        try:
            person_det = PersonDetector()
        except Exception as e:
            print(f"  PersonDetector load failed: {e}")
            print("  Falling back to face-only mode (v6 behavior)")
            person_det = None
    else:
        print("  yolov8n.onnx not found — face-only fallback")
        person_det = None

    # Face 감지기
    try:
        face_det = YOLOFaceDetector() if dl["yolo"] else None
    except Exception as e:
        print(f"  Face detector load failed: {e}")
        face_det = None

    if person_det is None and face_det is None:
        print("  ERROR: No detector available. Exiting.")
        return

    # Head Pose
    try:
        if not dl["pose"]:
            raise FileNotFoundError("pose model missing")
        pose_est = SixDRepNetPose()
    except Exception as e:
        print(f"  SixDRepNet failed ({e}), using solvePnP fallback")
        pose_est = FallbackPose()

    # 성별/연령
    ga_est = None
    if dl["gender_age"]:
        try:
            ga_est = GenderAgeEstimator()
        except Exception as e:
            print(f"  GenderAge load failed: {e}")
    if ga_est is None:
        print("  Gender/Age estimation disabled")

    engine     = AttentionEngine()
    aggregator = BatchAggregator(board_id)
    tracker    = UniquePersonTracker()

    cap = cv2.VideoCapture(Config.CAMERA_ID)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open camera {Config.CAMERA_ID}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_H)
    print(f"\n  Camera ready\n")

    frame_n = 0
    # 마지막 처리 결과 캐시 (매 프레임 재사용)
    person_dets   = []
    face_dets_cur = []
    person_poses  = []
    person_scores = []
    track_infos   = []
    person_to_face_map = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1
        now = time.time()

        if frame_n % Config.PROCESS_EVERY_N == 0:
            # ── 1) Person 감지 (주 트래킹) ──
            if person_det is not None:
                person_dets = person_det.detect(frame)
            else:
                # person 모델 없으면 face bbox를 person으로 사용 (v6 호환)
                person_dets = face_det.detect(frame) if face_det else []

            # ── 2) Face 감지 (보조 — 시선·성별/연령) ──
            face_dets_cur = face_det.detect(frame) if face_det else []

            # ── 3) Face → person 연결 + 시선 추정 ──
            face_poses_raw = []
            for fdet in face_dets_cur:
                yaw, pitch, roll = pose_est.estimate(
                    fdet.get("face_crop"), fdet["bbox"], frame.shape)
                looking = engine.is_looking(yaw, pitch)
                face_poses_raw.append((yaw, pitch, roll, looking))

            person_poses, person_to_face_map = associate_faces_to_persons(
                person_dets, face_dets_cur, face_poses_raw)

            # ── 4) Attention Score (per person) ──
            person_scores = []
            for pose in person_poses:
                yaw, pitch, roll, look = pose
                person_scores.append(engine.score(yaw, pitch) if look else 0.)

            # ── 5) 트래킹 + 집계 ──
            n_look = sum(1 for _, _, _, lk in person_poses if lk)
            _, det_to_track = tracker.update(person_dets, person_poses, frame_n)
            aggregator.add(len(person_dets), n_look, person_scores, tracker)

            # ── 6) 성별/연령 추정 (트랙 단위 캐싱) ──
            track_infos = []
            for pi in range(len(person_dets)):
                ti = {}
                if ga_est and pi in det_to_track:
                    tid   = det_to_track[pi]
                    track = tracker.tracks.get(tid, {})
                    need_refresh = (
                        track.get("gender") is None or
                        (frame_n - track.get("ga_frame", -999)) >= Config.GENDER_AGE_REFRESH
                    )
                    if need_refresh:
                        face_crop = None
                        fi = person_to_face_map.get(pi)
                        if fi is not None:
                            face_crop = face_dets_cur[fi].get("face_crop")
                        g, a = ga_est.estimate(face_crop)
                        if g != "?":
                            track["gender"]   = g
                            track["age"]      = a
                            track["ga_frame"] = frame_n
                    ti = {"gender": track.get("gender"), "age": track.get("age")}
                track_infos.append(ti)

        # ── 통계 패널 데이터 ──
        n_tot  = len(person_dets)
        n_look = sum(1 for _, _, _, lk in person_poses if lk)
        m_cnt  = sum(1 for t in tracker.tracks.values() if t.get("gender") == "M")
        f_cnt  = sum(1 for t in tracker.tracks.values() if t.get("gender") == "F")
        ages   = [t["age"] for t in tracker.tracks.values() if t.get("age") is not None]
        avg_a  = round(sum(ages) / len(ages), 1) if ages else None

        stats = {
            "total":         n_tot,
            "looking":       n_look,
            "elapsed":       now - aggregator.t0,
            "unique_total":  tracker.total_unique,
            "unique_looked": tracker.looked_unique,
            "unique_attn":   tracker.attention_rate(),
            "male_count":    m_cnt,
            "female_count":  f_cnt,
            "avg_age":       avg_a,
        }

        frame = draw(frame, person_dets, face_dets_cur, person_poses,
                     person_scores, stats, track_infos, person_to_face_map)
        cv2.imshow("AdScope v7  (q: quit)", frame)

        if aggregator.should_flush():
            save_to_db(aggregator.flush())

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    p = aggregator.flush()
    if p["frame_count"] > 0:
        save_to_db(p)
    cap.release()
    cv2.destroyAllWindows()
    print("\nDone. data_log.jsonl saved.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    run_live()
