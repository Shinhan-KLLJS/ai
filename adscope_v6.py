"""
AdScope v6 — 성별/연령 추정 + 장거리 감지 강화

v5 대비 신규 기능:
  1. 성별/연령 추정 (GenderAgeEstimator — InsightFace genderage.onnx, 112×112)
     - 트랙 ID 단위로 캐싱 (GENDER_AGE_REFRESH 프레임마다 갱신)
     - DB에 unique_male / unique_female / avg_age / age_distribution 추가
  2. 장거리 감지 강화
     - YOLO 입력 해상도 640 → 960 (Config.YOLO_INPUT_SIZE)
       960: 약 1.33× 소형 얼굴 해상도 향상, 약 2× CPU 시간
     - SAHI-lite: 좌우 2-타일 분할 감지 (Config.SAHI_ENABLE=True 시 활성)
       타일(704px wide) at 960 → 1.78× 해상도 향상 vs 640 full-frame
     - 최소 얼굴 크기 15px → 8px
     - 감지 신뢰도 임계값 0.50 → 0.45

모델 수동 다운로드:
  [YOLOv8n-face]
    https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.onnx
    → C:\\adscope\\models\\yolov8n-face.onnx

  [6DRepNet360]
    https://github.com/thohemp/6DRepNet360/releases/download/v1.0.0/sixdrepnet360_Mobilenet_nobn_new.onnx
    → C:\\adscope\\models\\sixdrepnet.onnx

  [InsightFace genderage — buffalo_l 팩에 포함]
    buffalo_l.zip (약 320MB):
      https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
    → 압축 해제 → buffalo_l/genderage.onnx → C:\\adscope\\models\\genderage.onnx
    (스크립트 실행 시 직접 다운로드도 시도함)
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
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import os


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ① 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Config:
    # 판정 기준
    YAW_THRESHOLD    = 30
    PITCH_THRESHOLD  = 25
    CONFIDENCE_MIN   = 0.45    # v5: 0.50 → v6: 0.45 (원거리 소형 얼굴 포함)
    IOU_THRESHOLD    = 0.45

    # 처리 주기
    PROCESS_EVERY_N  = 2       # N프레임마다 AI 처리

    # 카메라
    CAMERA_ID        = 0
    FRAME_W          = 1280
    FRAME_H          = 720

    # 배치 저장
    BATCH_SEC        = 15

    # 모델 경로
    MODEL_DIR        = Path("models")
    YOLO_ONNX        = Path("models") / "yolov8n-face.onnx"
    POSE_ONNX        = Path("models") / "sixdrepnet.onnx"
    GENDER_AGE_ONNX  = Path("models") / "genderage.onnx"

    # 모델 최소 크기
    YOLO_MIN_SIZE       = 3_000_000
    POSE_MIN_SIZE       = 100_000
    GENDER_AGE_MIN_SIZE = 500_000

    # ── 장거리 감지 (v6 신규) ──
    YOLO_INPUT_SIZE  = 960     # 640=빠름, 960=균형(권장), 1280=고정밀
    MIN_FACE_SIZE    = 8       # px — v5: 15 → v6: 8 (원거리 소형 얼굴)
    SAHI_ENABLE      = False   # True: 2-타일 분할 감지 (GPU 환경 권장)
                               # CPU에서도 동작하나 약 2× 느려짐

    # ── 성별/연령 (v6 신규) ──
    GENDER_AGE_MIN_FACE = 25   # 이 크기(px) 미만 crop은 추정 스킵 (신뢰도 낮음)
    GENDER_AGE_REFRESH  = 30   # 트랙당 N프레임마다 재추정


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ② 모델 파일 확인 + 다운로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOWNLOAD_URLS = {
    "yolo": [
        "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.onnx",
        "https://huggingface.co/Ultralytics/assets/resolve/main/yolov8n-face.onnx",
    ],
    "pose": [
        "https://github.com/thohemp/6DRepNet360/releases/download/v1.0.0/sixdrepnet360_Mobilenet_nobn_new.onnx",
    ],
    "gender_age": [
        # buffalo_l 팩에서 genderage.onnx만 직접 받기 시도
        "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/buffalo_l/genderage.onnx",
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    ],
}

def _download_file(url, dest_path, desc):
    print(f"  ⬇  {desc} 다운로드... ", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"완료 ({dest_path.stat().st_size // 1024}KB)")
        return True
    except Exception as e:
        print(f"실패 ({e})")
        if dest_path.exists():
            dest_path.unlink(missing_ok=True)
        return False

def _try_extract_from_zip(zip_url, inner_filename, dest_path):
    """buffalo_l.zip 등에서 특정 파일만 추출."""
    print(f"  ⬇  ZIP 다운로드 중 ({zip_url}) — 잠시 기다려주세요...", flush=True)
    try:
        data = urllib.request.urlopen(zip_url, timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                if Path(name).name.lower() == inner_filename.lower():
                    raw = z.read(name)
                    dest_path.write_bytes(raw)
                    print(f"  ✅ 추출 완료: {name} ({len(raw)//1024}KB) → {dest_path}")
                    return True
        print(f"  ⚠  ZIP 내에 {inner_filename} 없음")
    except Exception as e:
        print(f"  ⚠  ZIP 처리 실패: {e}")
    return False

def check_and_download_models():
    Config.MODEL_DIR.mkdir(exist_ok=True)
    results = {"yolo": False, "pose": False, "gender_age": False}

    specs = [
        ("yolo",       Config.YOLO_ONNX,       Config.YOLO_MIN_SIZE,       "YOLOv8n-face ONNX"),
        ("pose",       Config.POSE_ONNX,        Config.POSE_MIN_SIZE,       "6DRepNet360 ONNX"),
        ("gender_age", Config.GENDER_AGE_ONNX,  Config.GENDER_AGE_MIN_SIZE, "InsightFace genderage ONNX"),
    ]

    for key, path, min_size, desc in specs:
        if path.exists() and path.stat().st_size >= min_size:
            print(f"  ✅ {desc} — 캐시 사용 ({path.stat().st_size // 1024}KB)")
            results[key] = True
            continue

        downloaded = False
        for url in DOWNLOAD_URLS[key]:
            if url.endswith(".zip"):
                # ZIP에서 추출
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

        if not downloaded:
            if key == "gender_age":
                print(f"\n  ⚠  {desc} 다운로드 실패 — 성별/연령 추정 비활성화")
                print(f"     수동 다운로드: buffalo_l.zip 내 genderage.onnx")
                print(f"     → {path}\n")
            else:
                print(f"\n  ⚠  {desc} 다운로드 실패")
                for u in DOWNLOAD_URLS[key]:
                    print(f"     URL: {u}")
                print()

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ③ YOLOv8-face ONNX 감지기 (장거리 강화 버전)
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
        sahi = "SAHI-on" if Config.SAHI_ENABLE else f"size={Config.YOLO_INPUT_SIZE}"
        print(f"  ✅ YOLOv8n-face 로드 완료 [{used}] ({sahi})")

    def _letterbox(self, frame, size):
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

    def _raw_detect(self, crop, offset_x=0, offset_y=0):
        """ONNX 추론만 실행, NMS 없이 (boxes, scores) 반환."""
        h, w = crop.shape[:2]
        size = Config.YOLO_INPUT_SIZE
        padded, scale, pt, pl = self._letterbox(crop, size)
        blob = padded.astype(np.float32) / 255.
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        out = self.sess.run(None, {self.input_name: blob})[0]
        preds = out[0].T
        mask = preds[:, 4] >= Config.CONFIDENCE_MIN
        preds = preds[mask]
        if not len(preds):
            return np.zeros((0, 4)), np.zeros(0)

        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1o = np.clip((cx - bw / 2 - pl) / scale, 0, w) + offset_x
        y1o = np.clip((cy - bh / 2 - pt) / scale, 0, h) + offset_y
        x2o = np.clip((cx + bw / 2 - pl) / scale, 0, w) + offset_x
        y2o = np.clip((cy + bh / 2 - pt) / scale, 0, h) + offset_y
        boxes = np.stack([x1o, y1o, x2o, y2o], axis=1)
        return boxes, preds[:, 4]

    def _to_detections(self, frame, boxes, scores):
        """(boxes, scores) → detection dict 리스트 변환."""
        h, w = frame.shape[:2]
        results = []
        for box, score in zip(boxes, scores):
            x1i, y1i, x2i, y2i = map(int, box)
            x1i, y1i = max(0, x1i), max(0, y1i)
            x2i, y2i = min(w, x2i), min(h, y2i)
            bw_i, bh_i = x2i - x1i, y2i - y1i
            if bw_i < Config.MIN_FACE_SIZE or bh_i < Config.MIN_FACE_SIZE:
                continue
            p = 6
            crop = frame[max(0, y1i - p):min(h, y2i + p),
                         max(0, x1i - p):min(w, x2i + p)]
            results.append({
                "bbox": (x1i, y1i, bw_i, bh_i),
                "confidence": float(score),
                "face_crop": crop,
            })
        return results

    def detect(self, frame):
        """표준 전체 프레임 감지."""
        boxes, scores = self._raw_detect(frame)
        if not len(boxes):
            return []
        keep = self._nms(boxes, scores)
        return self._to_detections(frame, boxes[keep], scores[keep])

    def detect_sahi(self, frame):
        """
        SAHI-lite: 좌우 2-타일 분할 + 전체 프레임 병합.
        타일(704px 폭) at 960 → ~1.78× 해상도 vs 640 full-frame.
        NMS는 모든 타일 합산 후 한 번만 실행.
        """
        h, w = frame.shape[:2]
        overlap = int(w * 0.10)   # 10% 오버랩으로 경계 얼굴 누락 방지
        mid = w // 2

        # 좌측 타일: 0 ~ mid+overlap
        b0, s0 = self._raw_detect(frame[:, :mid + overlap], offset_x=0)
        # 우측 타일: mid-overlap ~ w (offset 보정)
        b1, s1 = self._raw_detect(frame[:, mid - overlap:], offset_x=mid - overlap)

        all_b = [b for b in [b0, b1] if len(b)]
        all_s = [s for s in [s0, s1] if len(s)]
        if not all_b:
            return []

        merged_b = np.concatenate(all_b)
        merged_s = np.concatenate(all_s)
        keep = self._nms(merged_b, merged_s)
        return self._to_detections(frame, merged_b[keep], merged_s[keep])

    def run(self, frame):
        """SAHI_ENABLE 설정에 따라 적절한 감지 방식 자동 선택."""
        if Config.SAHI_ENABLE:
            return self.detect_sahi(frame)
        return self.detect(frame)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ④ DNN ResNet-SSD 폴백 감지기
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DNNFaceDetector:
    PROTOTXT   = Path("models") / "deploy.prototxt"
    CAFFEMODEL = Path("models") / "res10_300x300_ssd_iter_140000.caffemodel"
    PROTOTXT_URL   = ("https://raw.githubusercontent.com/opencv/opencv/master/"
                      "samples/dnn/face_detector/deploy.prototxt")
    CAFFEMODEL_URL = ("https://github.com/opencv/opencv_3rdparty/raw/"
                      "dnn_samples_face_detector_20170830/"
                      "res10_300x300_ssd_iter_140000.caffemodel")

    def __init__(self):
        Path("models").mkdir(exist_ok=True)
        for path, url, min_sz, desc in [
            (self.PROTOTXT,   self.PROTOTXT_URL,   1000,      "prototxt"),
            (self.CAFFEMODEL, self.CAFFEMODEL_URL,  9_000_000, "caffemodel"),
        ]:
            if not (path.exists() and path.stat().st_size >= min_sz):
                _download_file(url, path, f"DNN {desc}")
        self.net = cv2.dnn.readNetFromCaffe(str(self.PROTOTXT), str(self.CAFFEMODEL))
        print("  ✅ DNN ResNet-SSD 로드 완료 (폴백)")

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104., 177., 123.), False, False)
        self.net.setInput(blob)
        dets = self.net.forward()
        results = []
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < Config.CONFIDENCE_MIN:
                continue
            x1 = max(0, int(dets[0, 0, i, 3] * w))
            y1 = max(0, int(dets[0, 0, i, 4] * h))
            x2 = min(w, int(dets[0, 0, i, 5] * w))
            y2 = min(h, int(dets[0, 0, i, 6] * h))
            bw, bh = x2 - x1, y2 - y1
            if bw < Config.MIN_FACE_SIZE or bh < Config.MIN_FACE_SIZE:
                continue
            p = 6
            crop = frame[max(0, y1 - p):min(h, y2 + p),
                         max(0, x1 - p):min(w, x2 + p)]
            results.append({"bbox": (x1, y1, bw, bh), "confidence": conf, "face_crop": crop})
        return results

    def run(self, frame):
        return self.detect(frame)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⑤ 6DRepNet Head Pose
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
        print(f"  ✅ SixDRepNet 로드 완료 (출력: {out_shape})")

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
# ⑥ 폴백 Head Pose (solvePnP)
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
        print("  ⚠  Head Pose 폴백 (solvePnP) — 6DRepNet보다 정확도 낮음")

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
# ⑦ 성별/연령 추정 (신규)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class GenderAgeEstimator:
    """
    InsightFace antelopev2 팩의 genderage.onnx 사용 (96×96 입력).
    출력: (1, 3) — [male_logit, female_logit, age/100]
    주의: buffalo_l은 순서가 반대 [female, male, age]
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
        inp  = self.sess.get_inputs()[0]
        out  = self.sess.get_outputs()[0]
        # 입력 크기 자동 감지 (buffalo_l=112, antelopev2=96)
        shape = inp.shape  # e.g. [None, 3, 96, 96] or [1, 3, 112, 112]
        self.input_size = int(shape[3]) if shape[3] not in (None, "None") else 112
        used = self.sess.get_providers()[0].replace("ExecutionProvider", "")
        print(f"  ✅ GenderAge 로드 완료 [{used}] "
              f"(입력:{self.input_size}×{self.input_size}, 출력:{out.shape})")

    def _preprocess(self, face_img):
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size))
        normed = (resized.astype(np.float32) - 127.5) / 127.5   # → [-1, 1]
        return normed.transpose(2, 0, 1)[np.newaxis]             # (1,3,H,W)

    def estimate(self, face_crop):
        """
        반환: (gender, age)
          gender: "M" | "F" | "?"
          age:    int (1~99) | None
        """
        if face_crop is None or face_crop.size == 0:
            return "?", None
        h, w = face_crop.shape[:2]
        if w < Config.GENDER_AGE_MIN_FACE or h < Config.GENDER_AGE_MIN_FACE:
            return "?", None
        try:
            blob = self._preprocess(face_crop)
            pred = self.sess.run(None, {self.input_name: blob})[0][0]  # (3,) 또는 (2,)
            if len(pred) >= 3:
                # antelopev2: pred[0]=male_logit, pred[1]=female_logit
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
# ⑧ 주목 판정 + Attention Score
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
# ⑨ 고유 인원 트래커 (성별/연령 캐싱 포함)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class UniquePersonTracker:
    """
    IoU + 중심점 거리 기반 고유 인원 트래킹.
    v6 추가: 트랙별 gender/age 캐싱, update()가 det_index→track_id 매핑 반환.
    """
    IOU_THRESH        = 0.25   # 0.30→0.25: 고개 돌림 시 박스 변형에 더 관대
    MAX_MISSING       = 60     # 20→60 (~4초): 짧은 시선 이탈·고개 돌림 후 복귀 허용
    CENTROID_FALLBACK = 160    # IoU 기준 미달 시 중심점 거리(px)로 fallback 매칭

    def __init__(self):
        self.tracks        = {}   # track_id → {...}
        self.next_id       = 1
        self.total_unique  = 0
        self.looked_unique = 0

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
        반환: (new_count, det_to_track)
          det_to_track: {detection_index: track_id}
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
            best_dist = float("inf")
            for i, box in enumerate(boxes):
                if i in matched_boxes:
                    continue
                s = self._iou(track["box"], box)
                d = self._centroid_dist(track["box"], box)
                if s > best_iou:
                    best_iou, best_i = s, i
                if d < best_dist:
                    best_dist = d
                    best_dist_i = i

            # IoU 기준 우선, 미달 시 중심점 거리 fallback
            if best_iou >= self.IOU_THRESH and best_i >= 0:
                pass  # 아래 공통 처리
            elif best_dist <= self.CENTROID_FALLBACK and best_dist_i >= 0:
                best_i = best_dist_i  # 중심점 기준으로 같은 사람 판정

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
                    "gender":   None,   # "M" | "F" | "?" | None(미추정)
                    "age":      None,
                    "ga_frame": -999,   # 마지막 gender/age 추정 프레임
                }
                self.total_unique += 1
                if looking:
                    self.looked_unique += 1
                det_to_track[i] = self.next_id
                self.next_id += 1
                new_count += 1

        self.tracks = {k: v for k, v in self.tracks.items()
                       if v["missing"] <= self.MAX_MISSING}
        return new_count, det_to_track

    def attention_rate(self):
        if self.total_unique == 0:
            return 0.0
        return round(self.looked_unique / self.total_unique * 100, 1)

    def active_count(self):
        return len(self.tracks)

    def reset(self):
        self.tracks        = {}
        self.next_id       = 1
        self.total_unique  = 0
        self.looked_unique = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⑩ 1분 배치 집계 (성별/연령 통계 포함)
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
        self.demographics_seen = set()    # track_id 중복 방지
        self.demographics      = []       # [{"gender":..., "age":...}, ...]

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
            # 확정된 성별/연령 즉시 기록 (트랙 퇴장 전에도 캡처)
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
        attn      = round(self.looking / self.total * 100, 1) if self.total else 0.
        avg_score = round(self.score_sum / self.score_n, 1)   if self.score_n else 0.
        unique_attn = (round(self.unique_looking / self.unique_total * 100, 1)
                       if self.unique_total > 0 else 0.)

        # 성별/연령 집계
        male_count   = sum(1 for d in self.demographics if d["gender"] == "M")
        female_count = sum(1 for d in self.demographics if d["gender"] == "F")
        ages = [d["age"] for d in self.demographics if d["age"] is not None]
        avg_age = round(sum(ages) / len(ages), 1) if ages else None
        age_dist = {"10s": 0, "20s": 0, "30s": 0, "40s": 0, "50plus": 0}
        for a in ages:
            if   a < 20: age_dist["10s"]    += 1
            elif a < 30: age_dist["20s"]    += 1
            elif a < 40: age_dist["30s"]    += 1
            elif a < 50: age_dist["40s"]    += 1
            else:        age_dist["50plus"] += 1

        p = {
            "board_id":               self.board_id,
            "window_start":           datetime.fromtimestamp(self.t0).strftime("%Y-%m-%d %H:%M:%S"),
            "window_end":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # ── 고유 인원 기반 (메인 KPI) ──
            "unique_total":           self.unique_total,
            "unique_looked":          self.unique_looking,
            "unique_attention_rate":  unique_attn,
            # ── 성별/연령 (v6 신규) ──
            "unique_male":            male_count,
            "unique_female":          female_count,
            "avg_age":                avg_age,
            "age_distribution":       age_dist,
            # ── 프레임 누적 (보조 지표) ──
            "frame_detections":       self.total,
            "frame_looking":          self.looking,
            "frame_attention_rate":   attn,
            "avg_attention_score":    avg_score,
            "peak_persons":           self.peak,
            "frame_count":            self.frame_count,
        }
        self.reset()
        return p


def save_to_db(payload):
    print("\n" + "━" * 62)
    print(f"  📊 [{payload['window_start']}] 1분 집계")
    skip = {"window_start", "window_end", "board_id", "age_distribution"}
    for k, v in payload.items():
        if k not in skip:
            print(f"     {k:<28}: {v}")
    if payload.get("age_distribution"):
        print(f"     {'age_distribution':<28}: {payload['age_distribution']}")
    print("━" * 62)
    with open("data_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⑪ 시각화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def draw_axis(frame, bbox, yaw, pitch, roll):
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2
    sz = max(int(w * 0.55), 20)
    yr, pr, rr = math.radians(yaw), math.radians(pitch), math.radians(roll)

    # Z축 — 코가 향하는 방향 (광고판 주목 핵심)
    cv2.arrowedLine(frame, (cx, cy),
                    (cx + int(sz * math.cos(yr) * math.cos(pr)),
                     cy - int(sz * math.sin(pr))),
                    (0, 0, 255), 2, tipLength=0.3)
    # X축 — 좌우 yaw
    cv2.arrowedLine(frame, (cx, cy),
                    (cx + int(sz * 0.55 * math.cos(yr + math.pi / 2)),
                     cy + int(sz * 0.55 * math.sin(rr))),
                    (0, 200, 0), 1, tipLength=0.3)
    # Y축 — 상하 pitch
    cv2.arrowedLine(frame, (cx, cy),
                    (cx - int(sz * 0.55 * math.sin(yr) * math.sin(pr)),
                     cy - int(sz * 0.55 * math.cos(pr))),
                    (255, 100, 0), 1, tipLength=0.3)


def draw(frame, detections, pose_results, scores, stats, detector_name, track_infos=None):
    hf, wf = frame.shape[:2]

    for i, (det, (yaw, pitch, roll, look)) in enumerate(zip(detections, pose_results)):
        x, y, w, h = det["bbox"]
        conf  = det["confidence"]
        sc    = scores[i] if i < len(scores) else 0.
        color = (0, min(255, int(sc * 2.55 + 80)), 150) if look else (110, 70, 200)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        draw_axis(frame, (x, y, w, h), yaw, pitch, roll)

        # 신뢰도 바
        cv2.rectangle(frame, (x, y + h + 2), (x + int(w * conf), y + h + 6), color, -1)
        cv2.rectangle(frame, (x, y + h + 2), (x + w, y + h + 6), color, 1)

        # 레이블 줄 구성
        ti = (track_infos[i] if track_infos and i < len(track_infos) else {}) or {}
        ga_str = ""
        if ti.get("gender") not in (None, "?"):
            age_part = f"/{ti['age']}" if ti.get("age") is not None else ""
            ga_str = f"  {ti['gender']}{age_part}"

        lines = [
            f"{'LOOK' if look else 'PASS'} {sc:.0f}pt  {conf:.0%}",
            f"yaw:{yaw:+.0f} pitch:{pitch:+.0f}{ga_str}",
        ]
        for j, txt in enumerate(lines):
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            by = y - (j + 1) * (th + 7)
            if by < 0:
                continue
            cv2.rectangle(frame, (x, by - 2), (x + tw + 4, by + th + 2), color, -1)
            cv2.putText(frame, txt, (x + 2, by + th),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)

    # 반투명 패널
    ov = frame.copy()
    cv2.rectangle(ov, (8, 8), (345, 215), (10, 10, 22), -1)
    cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (8, 8), (345, 215), (80, 60, 155), 1)

    # Axis legend
    for li, (ltxt, lcol) in enumerate([("Z(nose)", (0, 0, 255)),
                                        ("X(yaw)",  (0, 200, 0)),
                                        ("Y(pitch)",(255, 100, 0))]):
        cv2.circle(frame, (wf - 120, 15 + li * 18), 4, lcol, -1)
        cv2.putText(frame, ltxt, (wf - 112, 19 + li * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, lcol, 1)

    # Stats panel
    m_cnt = stats.get("male_count", 0)
    f_cnt = stats.get("female_count", 0)
    avg_a = stats.get("avg_age")
    age_str  = f"{avg_a:.0f}y" if avg_a is not None else "--y"
    demo_str = f"M:{m_cnt} F:{f_cnt}  AvgAge:{age_str}"

    lines_panel = [
        (f"AdScope v6  [{detector_name}]",                       (200, 175, 255)),
        (f"Now    : {stats['total']:>3} active",                  (225, 225, 255)),
        (f"Looking: {stats['looking']:>3} (now)",                 (80, 255, 150)),
        (f"-----------------------------",                        (60, 60, 80)),
        (f"Unique : {stats['unique_total']:>3} total",            (255, 220, 100)),
        (f"Looked : {stats['unique_looked']:>3}",                 (255, 180, 80)),
        (f"Attn%  : {stats['unique_attn']:>5.1f}%",              (255, 200, 60)),
        (f"-----------------------------",                        (60, 60, 80)),
        (demo_str,                                                (160, 220, 255)),
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
# ⑫ 실시간 모드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_live(board_id="board_gangnam_01"):
    print(f"\n{'═' * 62}")
    print(f"  👁  AdScope v6 — 실시간 모드")
    print(f"  YOLO 입력 크기: {Config.YOLO_INPUT_SIZE}  |  SAHI: {'ON' if Config.SAHI_ENABLE else 'OFF'}")
    print(f"  종료: q 키")
    print(f"{'═' * 62}\n")

    print("[ 모델 파일 확인 ]")
    dl = check_and_download_models()
    print()

    # ── 감지기 선택 ──
    detector_name = "YOLOv8"
    try:
        detector = YOLOFaceDetector() if dl["yolo"] else (_ for _ in ()).throw(Exception())
    except Exception:
        print("  → DNN ResNet-SSD 폴백 사용")
        detector_name = "DNN-SSD"
        detector = DNNFaceDetector()

    # ── Head Pose 선택 ──
    pose_name = "6DRepNet"
    try:
        if not dl["pose"]:
            raise FileNotFoundError(f"모델 없음: {Config.POSE_ONNX}")
        pose_est = SixDRepNetPose()
    except Exception as e:
        print(f"  → SixDRepNet 로드 실패: {e}")
        print("  → solvePnP 폴백 사용")
        pose_name = "solvePnP"
        pose_est = FallbackPose()

    # ── 성별/연령 추정기 (옵션) ──
    ga_est = None
    if dl["gender_age"]:
        try:
            ga_est = GenderAgeEstimator()
        except Exception as e:
            print(f"  ⚠  GenderAge 로드 실패: {e}")
    if ga_est is None:
        print("  ⚠  성별/연령 추정 비활성 (genderage.onnx 없음)")

    engine     = AttentionEngine()
    aggregator = BatchAggregator(board_id)
    tracker    = UniquePersonTracker()

    cap = cv2.VideoCapture(Config.CAMERA_ID)
    if not cap.isOpened():
        print(f"❌ 카메라 {Config.CAMERA_ID}번 열기 실패")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_H)
    print(f"\n  ✅ 카메라 연결 완료\n")

    frame_n = 0
    detections, pose_results, scores = [], [], []
    track_infos = []
    conf_sum, conf_n = 0., 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1
        now = time.time()

        if frame_n % Config.PROCESS_EVERY_N == 0:
            detections   = detector.run(frame)
            pose_results = []
            scores       = []
            conf_sum, conf_n = 0., 0

            for det in detections:
                yaw, pitch, roll = pose_est.estimate(
                    det.get("face_crop"), det["bbox"], frame.shape)
                looking = engine.is_looking(yaw, pitch)
                sc      = engine.score(yaw, pitch)
                pose_results.append((yaw, pitch, roll, looking))
                scores.append(sc)
                conf_sum += det["confidence"]
                conf_n   += 1

            n_look = sum(1 for _, _, _, lk in pose_results if lk)
            _, det_to_track = tracker.update(detections, pose_results, frame_n)
            aggregator.add(len(detections), n_look, scores, tracker)

            # ── 성별/연령 추정 (트랙 단위 캐싱) ──
            track_infos = []
            for i, det in enumerate(detections):
                ti = {}
                if ga_est and i in det_to_track:
                    tid   = det_to_track[i]
                    track = tracker.tracks.get(tid, {})
                    need_refresh = (
                        track.get("gender") is None or
                        (frame_n - track.get("ga_frame", -999)) >= Config.GENDER_AGE_REFRESH
                    )
                    if need_refresh:
                        g, a = ga_est.estimate(det.get("face_crop"))
                        if g != "?":
                            track["gender"]   = g
                            track["age"]      = a
                            track["ga_frame"] = frame_n
                    ti = {"gender": track.get("gender"), "age": track.get("age")}
                track_infos.append(ti)

        # 현재 프레임 통계
        n_look = sum(1 for _, _, _, lk in pose_results if lk)
        n_tot  = len(detections)

        # 성별/연령 집계 (패널 표시용)
        m_cnt = sum(1 for t in tracker.tracks.values() if t.get("gender") == "M")
        f_cnt = sum(1 for t in tracker.tracks.values() if t.get("gender") == "F")
        ages  = [t["age"] for t in tracker.tracks.values()
                 if t.get("age") is not None]
        avg_a = round(sum(ages) / len(ages), 1) if ages else None

        stats = {
            "total":          n_tot,
            "looking":        n_look,
            "elapsed":        now - aggregator.t0,
            "unique_total":   tracker.total_unique,
            "unique_looked":  tracker.looked_unique,
            "unique_attn":    tracker.attention_rate(),
            "male_count":     m_cnt,
            "female_count":   f_cnt,
            "avg_age":        avg_a,
        }
        frame = draw(frame, detections, pose_results, scores, stats,
                     f"{detector_name}+{pose_name}", track_infos)
        cv2.imshow("AdScope v6  (q: 종료)", frame)

        if aggregator.should_flush():
            save_to_db(aggregator.flush())

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    p = aggregator.flush()
    if p["frame_count"] > 0:
        save_to_db(p)
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ 종료. data_log.jsonl 저장 완료.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⑬ 테스트 모드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_test():
    import random
    print(f"\n{'═' * 62}")
    print(f"  🧪 AdScope v6 — 기능 검증")
    print(f"{'═' * 62}\n")

    engine = AttentionEngine()
    pose   = FallbackPose()
    frame_shape = (720, 1280, 3)

    print("① Head Pose + 주목 판정\n")
    cases = [
        ("정면 중앙",       (560, 280, 120, 140), True),
        ("왼쪽 끝",         ( 50, 280,  70, 130), False),
        ("오른쪽 끝",       (1100, 280,  70, 130), False),
        ("원거리 소형 얼굴", (600, 300,  40,  50), True),
        ("측면 찌그러짐",   (560, 280,  60, 140), False),
    ]
    print(f"  {'상황':<20} {'yaw':>8} {'pitch':>8} {'판정':>10}  {'점수':>6}  정확")
    print("  " + "─" * 62)
    correct = 0
    for name, bbox, expected in cases:
        yaw, pitch, _ = pose.estimate(None, bbox, frame_shape)
        looking = engine.is_looking(yaw, pitch)
        sc = engine.score(yaw, pitch)
        ok = looking == expected
        correct += int(ok)
        print(f"  {name:<20} {yaw:>+7.1f}° {pitch:>+7.1f}°  "
              f"{'LOOKING' if looking else 'PASSING':>10}  {sc:>5.1f}  {'✅' if ok else '❌'}")
    print(f"\n  정확도: {correct}/{len(cases)} ({correct / len(cases) * 100:.0f}%)")

    print(f"\n② UniquePersonTracker + 성별/연령 캐싱 시뮬레이션\n")
    tracker = UniquePersonTracker()
    random.seed(42)
    print(f"  {'프레임':>4}  {'감지':>4}  {'신규':>4}  {'활성':>4}  {'누적':>5}  "
          f"{'본人':>4}  {'주목률':>7}  {'M/F':>5}")
    print("  " + "─" * 60)
    for fn in range(1, 25):
        n = random.randint(1, 5)
        dets  = [{"bbox": (100 + i * 130, 100, 75, 100)} for i in range(n)]
        poses = [(random.gauss(0, 22), random.gauss(0, 12), 0.,
                  random.random() < 0.40) for _ in range(n)]
        new, d2t = tracker.update(dets, poses, fn)
        # 시뮬레이션: 30% 확률로 성별/연령 세팅
        for i, det in enumerate(dets):
            if i in d2t and random.random() < 0.30:
                tid = d2t[i]
                if tid in tracker.tracks:
                    tracker.tracks[tid]["gender"]   = random.choice(["M", "F"])
                    tracker.tracks[tid]["age"]      = random.randint(18, 55)
                    tracker.tracks[tid]["ga_frame"] = fn
        m = sum(1 for t in tracker.tracks.values() if t.get("gender") == "M")
        f = sum(1 for t in tracker.tracks.values() if t.get("gender") == "F")
        if fn <= 8 or fn % 4 == 0:
            print(f"  {fn:>4}  {n:>4}  {new:>4}  {tracker.active_count():>4}  "
                  f"{tracker.total_unique:>5}  {tracker.looked_unique:>4}  "
                  f"{tracker.attention_rate():>6.1f}%  {m}M/{f}F")

    print(f"\n  ✅ 고유 인원 {tracker.total_unique}명, 주목률 {tracker.attention_rate()}%")

    print(f"\n③ BatchAggregator 성별/연령 집계 시뮬\n")
    agg = BatchAggregator("board_gangnam_01")
    random.seed(7)
    for _ in range(200):
        n = random.randint(2, 6)
        dets  = [{"bbox": (560 + random.randint(-100, 100),
                           280 + random.randint(-50, 50),
                           random.randint(60, 120),
                           random.randint(80, 140)),
                  "confidence": random.uniform(0.55, 0.97),
                  "face_crop": None} for _ in range(n)]
        prs = []
        scs = []
        for d in dets:
            y2, p2, _ = pose.estimate(None, d["bbox"], frame_shape)
            lk = engine.is_looking(y2, p2)
            scs.append(engine.score(y2, p2))
            prs.append((y2, p2, 0., lk))
        n_look = sum(1 for _, _, _, lk in prs if lk)
        agg.add(n, n_look, scs, tracker)
    agg.t0 -= Config.BATCH_SEC
    payload = agg.flush()
    print("  DB payload (주요 항목):")
    keys = ["unique_total", "unique_looked", "unique_attention_rate",
            "unique_male", "unique_female", "avg_age", "age_distribution",
            "frame_attention_rate", "avg_attention_score"]
    for k in keys:
        print(f"    {k:<28}: {payload.get(k)}")

    print(f"\n{'═' * 62}")
    print("  ✅ 검증 완료!")
    print("  웹캠 실행: 파일 맨 아래 RUN_MODE = 'live' 로 변경")
    print(f"{'═' * 62}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실행 모드
#   'test' : 카메라 없이 기능 검증
#   'live' : 웹캠 실시간 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RUN_MODE = "live"

if __name__ == "__main__":
    if RUN_MODE == "test":
        run_test()
    else:
        run_live(board_id="board_gangnam_01")
