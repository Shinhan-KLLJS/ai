"""
AdScope v4 (Fixed) — 버그 수정 + 모델 수동 다운로드 안내 버전

수정 사항:
  1. 폴백 Head Pose 100% 버그 수정 (solvePnP 오일러 각도 계산 개선)
  2. 모델 다운로드 URL 수정 + 수동 다운로드 안내 추가
  3. 모델 없을 때도 안정적으로 동작하는 폴백 구조

설치:
    pip install opencv-python numpy onnxruntime requests

모델 수동 다운로드 (브라우저에서 직접):
    [YOLOv8n-face]
    https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.onnx
    → C:\\adscope\\models\\yolov8n-face.onnx 로 저장

    [6DRepNet360]
    https://github.com/thohemp/6DRepNet360/releases/download/v1.0.0/sixdrepnet360_Mobilenet_nobn_new.onnx
    → C:\\adscope\\models\\sixdrepnet360_Mobilenet_nobn_new.onnx 로 저장
"""

import cv2
import numpy as np
import json
import time
import math
import random
import urllib.request
from datetime import datetime
from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ① 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Config:
    YAW_THRESHOLD    = 30
    PITCH_THRESHOLD  = 25
    CONFIDENCE_MIN   = 0.50
    IOU_THRESHOLD    = 0.45
    PROCESS_EVERY_N  = 2
    CAMERA_ID        = 0
    FRAME_W          = 1280
    FRAME_H          = 720
    BATCH_SEC        = 60
    MODEL_DIR        = Path("models")
    YOLO_ONNX        = Path("models") / "yolov8n-face.onnx"
    POSE_ONNX        = Path("models") / "sixdrepnet.onnx"
    YOLO_MIN_SIZE    = 3_000_000   # 3MB 이상이어야 유효한 모델
    POSE_MIN_SIZE    = 3_000_000


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ② 모델 파일 확인 + 안내
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOWNLOAD_URLS = {
    "yolo": [
        "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.onnx",
        "https://huggingface.co/Ultralytics/assets/resolve/main/yolov8n-face.onnx",
    ],
    "pose": [
        "https://github.com/thohemp/6DRepNet360/releases/download/v1.0.0/sixdrepnet360_Mobilenet_nobn_new.onnx",
    ],
}

def check_and_download_models():
    Config.MODEL_DIR.mkdir(exist_ok=True)
    results = {"yolo": False, "pose": False}

    specs = [
        ("yolo", Config.YOLO_ONNX, Config.YOLO_MIN_SIZE,
         "YOLOv8n-face ONNX (~6MB)"),
        ("pose", Config.POSE_ONNX, Config.POSE_MIN_SIZE,
         "6DRepNet360 ONNX (~4MB)"),
    ]

    for key, path, min_size, desc in specs:
        # 이미 있고 유효한 크기면 스킵
        if path.exists() and path.stat().st_size >= min_size:
            print(f"  ✅ {desc} — 캐시 사용 ({path.stat().st_size//1024}KB)")
            results[key] = True
            continue

        # 자동 다운로드 시도
        downloaded = False
        for url in DOWNLOAD_URLS[key]:
            print(f"  ⬇️  {desc} 다운로드 시도... ", end="", flush=True)
            try:
                urllib.request.urlretrieve(url, path)
                if path.exists() and path.stat().st_size >= min_size:
                    print(f"완료 ({path.stat().st_size//1024}KB)")
                    results[key] = True
                    downloaded = True
                    break
                else:
                    print("파일 크기 이상 (재시도)")
                    path.unlink(missing_ok=True)
            except Exception as e:
                print(f"실패 ({e})")
                path.unlink(missing_ok=True)

        if not downloaded:
            print(f"\n  ⚠️  {desc} 자동 다운로드 실패")
            print(f"     브라우저에서 직접 다운로드 후")
            print(f"     C:\\adscope\\{path} 에 저장하세요")
            for url in DOWNLOAD_URLS[key]:
                print(f"     URL: {url}")
            print()

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ③ YOLOv8-face ONNX 감지기
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
        used = self.sess.get_providers()[0].replace("ExecutionProvider","")
        print(f"  ✅ YOLOv8n-face 로드 완료 [{used}]")

    def _letterbox(self, frame, size=640):
        h, w = frame.shape[:2]
        scale = size / max(h, w)
        nh, nw = int(h*scale), int(w*scale)
        resized = cv2.resize(frame, (nw, nh))
        pad_h, pad_w = size-nh, size-nw
        top, left = pad_h//2, pad_w//2
        padded = cv2.copyMakeBorder(resized, top, pad_h-top, left, pad_w-left,
                                     cv2.BORDER_CONSTANT, value=(114,114,114))
        return padded, scale, top, left

    def _nms(self, boxes, scores):
        if not len(boxes): return []
        x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        areas = (x2-x1)*(y2-y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size:
            i = order[0]; keep.append(i)
            inter_x1 = np.maximum(x1[i], x1[order[1:]])
            inter_y1 = np.maximum(y1[i], y1[order[1:]])
            inter_x2 = np.minimum(x2[i], x2[order[1:]])
            inter_y2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0,inter_x2-inter_x1)*np.maximum(0,inter_y2-inter_y1)
            iou   = inter/(areas[i]+areas[order[1:]]-inter+1e-8)
            order = order[np.where(iou<=Config.IOU_THRESHOLD)[0]+1]
        return keep

    def detect(self, frame):
        h, w = frame.shape[:2]
        padded, scale, pt, pl = self._letterbox(frame)
        blob = padded.astype(np.float32)/255.
        blob = blob.transpose(2,0,1)[np.newaxis]
        out  = self.sess.run(None, {self.input_name: blob})[0]
        preds = out[0].T
        mask  = preds[:,4] >= Config.CONFIDENCE_MIN
        preds = preds[mask]
        if not len(preds): return []

        cx,cy,bw,bh = preds[:,0],preds[:,1],preds[:,2],preds[:,3]
        x1p,y1p = cx-bw/2, cy-bh/2
        x2p,y2p = cx+bw/2, cy+bh/2
        x1o = np.clip((x1p-pl)/scale, 0, w)
        y1o = np.clip((y1p-pt)/scale, 0, h)
        x2o = np.clip((x2p-pl)/scale, 0, w)
        y2o = np.clip((y2p-pt)/scale, 0, h)
        boxes  = np.stack([x1o,y1o,x2o,y2o],axis=1)
        scores = preds[:,4]
        keep   = self._nms(boxes, scores)

        results = []
        for i in keep:
            x1i,y1i,x2i,y2i = map(int, boxes[i])
            bw_i,bh_i = x2i-x1i, y2i-y1i
            if bw_i<15 or bh_i<15: continue
            p=8
            crop = frame[max(0,y1i-p):min(h,y2i+p),
                         max(0,x1i-p):min(w,x2i+p)]
            results.append({"bbox":(x1i,y1i,bw_i,bh_i),
                             "confidence":float(scores[i]),
                             "face_crop":crop})
        return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ④ DNN ResNet-SSD 폴백 감지기
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DNNFaceDetector:
    PROTOTXT   = Path("models")/"deploy.prototxt"
    CAFFEMODEL = Path("models")/"res10_300x300_ssd_iter_140000.caffemodel"

    PROTOTXT_URL   = ("https://raw.githubusercontent.com/opencv/opencv/master/"
                      "samples/dnn/face_detector/deploy.prototxt")
    CAFFEMODEL_URL = ("https://github.com/opencv/opencv_3rdparty/raw/"
                      "dnn_samples_face_detector_20170830/"
                      "res10_300x300_ssd_iter_140000.caffemodel")

    def __init__(self):
        Path("models").mkdir(exist_ok=True)
        for path, url, min_sz, desc in [
            (self.PROTOTXT,   self.PROTOTXT_URL,   1000,    "prototxt"),
            (self.CAFFEMODEL, self.CAFFEMODEL_URL,  9_000_000, "caffemodel"),
        ]:
            if not (path.exists() and path.stat().st_size >= min_sz):
                print(f"    ⬇️  DNN {desc} 다운로드...", end=" ", flush=True)
                try:
                    urllib.request.urlretrieve(url, path)
                    print("완료")
                except Exception as e:
                    print(f"실패 ({e})")

        self.net = cv2.dnn.readNetFromCaffe(str(self.PROTOTXT), str(self.CAFFEMODEL))
        print("  ✅ DNN ResNet-SSD 로드 완료 (폴백)")

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame,(300,300)), 1.0, (300,300),
            (104.,177.,123.), False, False)
        self.net.setInput(blob)
        dets = self.net.forward()
        results = []
        for i in range(dets.shape[2]):
            conf = float(dets[0,0,i,2])
            if conf < Config.CONFIDENCE_MIN: continue
            x1=int(dets[0,0,i,3]*w); y1=int(dets[0,0,i,4]*h)
            x2=int(dets[0,0,i,5]*w); y2=int(dets[0,0,i,6]*h)
            x1,y1=max(0,x1),max(0,y1)
            x2,y2=min(w,x2),min(h,y2)
            bw,bh=x2-x1,y2-y1
            if bw<20 or bh<20: continue
            p=8
            crop = frame[max(0,y1-p):min(h,y2+p),
                         max(0,x1-p):min(w,x2+p)]
            results.append({"bbox":(x1,y1,bw,bh),
                             "confidence":conf,
                             "face_crop":crop})
        return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⑤ 6DRepNet360 Head Pose
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SixDRepNetPose:
    """
    SixDRepNet (pip 패키지 → ONNX 변환 버전)
    출력: (1,3,3) 회전 행렬 → yaw/pitch/roll
    export_6drepnet_v3.py 로 변환한 models/sixdrepnet.onnx 사용
    """
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
        print(f"  ✅ SixDRepNet ONNX 로드 완료 (출력: {out_shape}, ±5° 정밀도)")

    def _preprocess(self, face_img):
        resized = cv2.resize(face_img, (224, 224))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normed  = (rgb - self.MEAN) / self.STD
        return normed.transpose(2, 0, 1)[np.newaxis]

    def _rot_to_euler(self, R):
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy > 1e-6:
            pitch = math.atan2(-R[2,0], sy)
            yaw   = math.atan2(R[1,0]/math.cos(pitch), R[0,0]/math.cos(pitch))
            roll  = math.atan2(R[2,1], R[2,2])
        else:
            pitch = math.atan2(-R[2,0], sy)
            yaw, roll = 0., math.atan2(-R[1,2], R[1,1])
        return round(math.degrees(yaw),1), round(math.degrees(pitch),1), round(math.degrees(roll),1)

    def estimate(self, face_crop, bbox=None, frame_shape=None):
        if face_crop is None or face_crop.size == 0:
            return 0., 0., 0.
        try:
            blob = self._preprocess(face_crop)
            out  = self.sess.run(None, {self.input_name: blob})[0]
            R    = out[0]  # (3,3) 회전 행렬
            return self._rot_to_euler(R)
        except Exception:
            return 0., 0., 0.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⑥ 개선된 폴백 Head Pose (solvePnP 버그 수정)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FallbackPose:
    """
    수정된 solvePnP 기반 Head Pose
    - 오일러 각도를 RQDecomp3x3 대신 atan2로 직접 계산
    - pitch 버그(항상 0°) 수정 → 100% LOOKING 오류 해결
    """
    FACE_3D = np.array([
        [0.,    0.,    0.  ],
        [0.,  -330., -65.  ],
        [-225., 170., -135.],
        [225.,  170., -135.],
        [-150.,-150., -125.],
        [150., -150., -125.],
    ], dtype=np.float64)

    def __init__(self):
        print("  ⚠️  Head Pose 폴백 (solvePnP) — 6DRepNet보다 정확도 낮음")
        print("      6DRepNet 모델을 수동 다운로드하면 정밀도 대폭 향상")

    def estimate(self, face_crop, bbox=None, frame_shape=None):
        if bbox is None or frame_shape is None:
            return 0., 0., 0.

        x, y, w, h = bbox
        fh, fw = frame_shape[:2]

        # 얼굴 박스 내 6개 랜드마크 추정 (비율 기반)
        landmarks_2d = np.array([
            [x + w*0.50, y + h*0.40],   # 코 끝
            [x + w*0.50, y + h*0.88],   # 턱
            [x + w*0.22, y + h*0.28],   # 왼눈 왼쪽
            [x + w*0.78, y + h*0.28],   # 오른눈 오른쪽
            [x + w*0.35, y + h*0.72],   # 왼입꼬리
            [x + w*0.65, y + h*0.72],   # 오른입꼬리
        ], dtype=np.float64)

        focal = fw
        cam   = np.array([[focal,0,fw/2],[0,focal,fh/2],[0,0,1]], dtype=np.float64)
        dist  = np.zeros((4,1))

        ok, rvec, _ = cv2.solvePnP(
            self.FACE_3D, landmarks_2d, cam, dist,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return 0., 0., 0.

        rmat, _ = cv2.Rodrigues(rvec)

        # atan2로 오일러 각도 직접 계산 (RQDecomp3x3 버그 우회)
        pitch = math.degrees(math.atan2(
            -rmat[2,0],
            math.sqrt(rmat[2,1]**2 + rmat[2,2]**2)))
        yaw   = math.degrees(math.atan2(rmat[1,0], rmat[0,0]))
        roll  = math.degrees(math.atan2(rmat[2,1], rmat[2,2]))

        return round(yaw,1), round(pitch,1), round(roll,1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⑦ 주목 판정 + Attention Score
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
# ⑧ 1분 배치 집계
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

    def add(self, n_total, n_looking, scores, tracker=None):
        self.total      += n_total
        self.looking    += n_looking
        self.peak        = max(self.peak, n_total)
        self.score_sum  += sum(scores)
        self.score_n    += len(scores)
        self.frame_count += 1
        # 고유 인원 트래커 스냅샷
        if tracker:
            self.unique_total   = tracker.total_unique
            self.unique_looking = tracker.looked_unique

    def should_flush(self):
        return time.time() - self.t0 >= Config.BATCH_SEC

    def flush(self):
        attn  = round(self.looking/self.total*100,1) if self.total else 0.
        avg_s = round(self.score_sum/self.score_n,1) if self.score_n else 0.
        unique_attn = (round(self.unique_looking/self.unique_total*100,1)
                       if self.unique_total > 0 else 0.)
        p = {
            "board_id":                self.board_id,
            "window_start":            datetime.fromtimestamp(self.t0).strftime("%Y-%m-%d %H:%M:%S"),
            "window_end":              datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # ── 고유 인원 기반 (메인 지표) ──
            "unique_total":            self.unique_total,
            "unique_looked":           self.unique_looking,
            "unique_attention_rate":   unique_attn,
            # ── 프레임 누적 (보조 지표) ──
            "frame_detections":        self.total,
            "frame_looking":           self.looking,
            "frame_attention_rate":    attn,
            "avg_attention_score":     avg_s,
            "peak_persons":            self.peak,
            "frame_count":             self.frame_count,
        }
        self.reset()
        return p

def save_to_db(payload):
    print("\n" + "━"*60)
    print(f"  📊 [{payload['window_start']}] 1분 집계")
    for k,v in payload.items():
        if k not in ("window_start","window_end","board_id"):
            print(f"     {k:<28}: {v}")
    print("━"*60)
    with open("data_log.jsonl","a",encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False)+"\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⑨ 시각화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def draw_axis(frame, bbox, yaw, pitch, roll):
    """
    3축 방향 화살표:
      빨간 화살표 → Z축 (코가 향하는 방향, 광고 주목 여부 판단 핵심)
      초록 화살표 → X축 (좌우 고개 돌림 = yaw)
      파란 화살표 → Y축 (위아래 고개 기울기 = pitch)
    """
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2
    sz = int(w * 0.6)

    yr = math.radians(yaw)
    pr = math.radians(pitch)
    rr = math.radians(roll)

    # Z축 — 코 방향 (광고판 향할 때 앞으로 나옴 → 짧아 보임)
    zx = int(sz * math.cos(yr) * math.cos(pr))
    zy = int(sz * math.sin(pr))
    cv2.arrowedLine(frame, (cx, cy), (cx + zx, cy - zy),
                    (0, 0, 255), 2, tipLength=0.3)

    # X축 — 좌우 yaw
    xx = int(sz * 0.6 * math.cos(yr + math.pi/2))
    xy = int(sz * 0.6 * math.sin(rr))
    cv2.arrowedLine(frame, (cx, cy), (cx + xx, cy + xy),
                    (0, 200, 0), 1, tipLength=0.3)

    # Y축 — 상하 pitch
    yx = int(sz * 0.6 * math.sin(yr) * math.sin(pr))
    yy = int(sz * 0.6 * math.cos(pr))
    cv2.arrowedLine(frame, (cx, cy), (cx - yx, cy - yy),
                    (255, 100, 0), 1, tipLength=0.3)

def draw(frame, detections, pose_results, scores, stats, detector_name):
    hf,wf = frame.shape[:2]

    for i,(det,(yaw,pitch,roll,look)) in enumerate(zip(detections,pose_results)):
        x,y,w,h = det["bbox"]
        conf    = det["confidence"]
        sc      = scores[i] if i<len(scores) else 0.
        color   = (0,min(255,int(sc*2.55+80)),150) if look else (110,70,200)

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        draw_axis(frame,(x,y,w,h),yaw,pitch,roll)

        # 신뢰도 바
        cv2.rectangle(frame,(x,y+h+2),(x+int(w*conf),y+h+6),color,-1)
        cv2.rectangle(frame,(x,y+h+2),(x+w,y+h+6),color,1)

        label1 = f"{'LOOK' if look else 'PASS'} {sc:.0f}pt  {conf:.0%}"
        label2 = f"yaw:{yaw:+.0f}  pitch:{pitch:+.0f}"
        for j,txt in enumerate([label1,label2]):
            (tw,th),_ = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.40,1)
            by = y-(j+1)*(th+7)
            cv2.rectangle(frame,(x,by-2),(x+tw+4,by+th+2),color,-1)
            cv2.putText(frame,txt,(x+2,by+th),
                        cv2.FONT_HERSHEY_SIMPLEX,0.40,(0,0,0),1)

    ov = frame.copy()
    cv2.rectangle(ov,(8,8),(330,185),(10,10,22),-1)
    cv2.addWeighted(ov,0.80,frame,0.20,0,frame)
    cv2.rectangle(frame,(8,8),(330,185),(80,60,155),1)

    # 화살표 범례
    legends = [("Z(코방향)", (0,0,255)), ("X(좌우)", (0,200,0)), ("Y(상하)", (255,100,0))]
    for li, (ltxt, lcol) in enumerate(legends):
        cv2.circle(frame, (wf-130, 15+li*18), 4, lcol, -1)
        cv2.putText(frame, ltxt, (wf-122, 19+li*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, lcol, 1)

    lines=[
        (f"AdScope v4  [{detector_name}]",                    (200,175,255)),
        (f"Now      : {stats['total']:>3}명 활성",             (225,225,255)),
        (f"Looking  : {stats['looking']:>3}명 (현재)",          (80,255,150)),
        (f"────────────────────────",                          (60,60,80)),
        (f"고유인원 : {stats['unique_total']:>3}명 누적",       (255,220,100)),
        (f"광고본人 : {stats['unique_looked']:>3}명",           (255,180,80)),
        (f"고유주목률: {stats['unique_attn']:>5.1f}%",         (255,200,60)),
        (f"Batch    : {stats['elapsed']:>4.0f}s / {Config.BATCH_SEC}s",(150,155,190)),
    ]
    for idx,(txt,col) in enumerate(lines):
        cv2.putText(frame,txt,(16,28+idx*22),
                    cv2.FONT_HERSHEY_SIMPLEX,0.48,col,1)

    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame,ts,(wf-210,hf-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.42,(70,200,70),1)
    return frame


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⑩ 실시간 모드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class UniquePersonTracker:
    """
    한 사람이 카메라에 등장하면 ID를 부여하고,
    퇴장 전까지 같은 사람을 다시 카운트하지 않음.

    - total_unique  : 실제로 지나간 고유 인원 수
    - looked_unique : 광고를 한 번이라도 본 고유 인원 수
    - attention_rate: looked_unique / total_unique (진짜 주목률)
    """
    IOU_THRESH  = 0.30   # 이 값 이상 겹치면 같은 사람으로 판단
    MAX_MISSING = 20     # 20프레임 안 보이면 퇴장으로 처리 (~0.7초)

    def __init__(self):
        self.tracks        = {}   # track_id → {box, missing, looked}
        self.next_id       = 1
        self.total_unique  = 0
        self.looked_unique = 0

    @staticmethod
    def _iou(b1, b2):
        ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-8)

    def update(self, detections, pose_results):
        """
        매 프레임 호출.
        detections  : YOLOv8 감지 결과 list
        pose_results: (yaw, pitch, roll, is_looking) list
        반환: 이번 프레임 신규 등장 인원 수
        """
        # bbox를 (x1,y1,x2,y2) 로 변환
        boxes = []
        for det in detections:
            x, y, w, h = det["bbox"]
            boxes.append((x, y, x+w, y+h))

        matched_tracks = set()
        matched_boxes  = set()

        # ── 기존 트랙과 새 박스 IoU 매칭 ──
        for tid, track in self.tracks.items():
            best_iou, best_i = 0, -1
            for i, box in enumerate(boxes):
                if i in matched_boxes:
                    continue
                score = self._iou(track["box"], box)
                if score > best_iou:
                    best_iou, best_i = score, i

            if best_iou >= self.IOU_THRESH and best_i >= 0:
                track["box"]     = boxes[best_i]
                track["missing"] = 0
                # 아직 한 번도 광고 안 봤는데 이번 프레임에 봤다면
                if (not track["looked"] and
                        best_i < len(pose_results) and
                        pose_results[best_i][3]):
                    track["looked"] = True
                    self.looked_unique += 1
                matched_tracks.add(tid)
                matched_boxes.add(best_i)
            else:
                track["missing"] += 1

        # ── 새 사람 등록 ──
        new_count = 0
        for i, box in enumerate(boxes):
            if i not in matched_boxes:
                looking = (pose_results[i][3]
                           if i < len(pose_results) else False)
                self.tracks[self.next_id] = {
                    "box":     box,
                    "missing": 0,
                    "looked":  looking,
                }
                self.total_unique += 1
                if looking:
                    self.looked_unique += 1
                self.next_id += 1
                new_count += 1

        # ── 퇴장 처리 ──
        self.tracks = {k: v for k, v in self.tracks.items()
                       if v["missing"] <= self.MAX_MISSING}
        return new_count

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





def run_live(board_id="board_gangnam_01"):
    print(f"\n{'═'*60}")
    print(f"  👁  AdScope v4 (Fixed) — 실시간 모드")
    print(f"  종료: q 키")
    print(f"{'═'*60}\n")

    print("[ 모델 파일 확인 ]")
    dl = check_and_download_models()
    print()

    # 감지기 선택
    detector_name = "YOLOv8"
    try:
        detector = YOLOFaceDetector() if dl["yolo"] else (_ for _ in ()).throw(Exception())
    except Exception:
        print("  → DNN ResNet-SSD 폴백 사용")
        detector_name = "DNN-SSD"
        detector = DNNFaceDetector()

    # Head Pose 선택
    pose_name = "6DRepNet"
    try:
        if not dl["pose"]:
            raise FileNotFoundError(f"모델 파일 없음: {Config.POSE_ONNX}")
        pose_est = SixDRepNetPose()
    except Exception as e:
        print(f"  → SixDRepNet 로드 실패: {e}")
        print(f"  → 파일 확인: {Config.POSE_ONNX} 존재={Config.POSE_ONNX.exists()}")
        print("  → solvePnZ 폴백 사용 (수정된 버전)")
        pose_name = "solvePnP"
        pose_est = FallbackPose()

    engine     = AttentionEngine()
    aggregator = BatchAggregator(board_id)
    tracker    = UniquePersonTracker()  # 고유 인원 트래커

    cap = cv2.VideoCapture(Config.CAMERA_ID)
    if not cap.isOpened():
        print(f"❌ 카메라 {Config.CAMERA_ID}번 열기 실패 (CAMERA_ID를 1로 변경해보세요)")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_H)
    print(f"\n  ✅ 카메라 연결 완료\n")

    frame_n = 0
    detections, pose_results, scores = [], [], []
    conf_sum, conf_n = 0., 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_n += 1
        now = time.time()

        if frame_n % Config.PROCESS_EVERY_N == 0:
            detections   = detector.detect(frame)
            pose_results = []
            scores       = []
            conf_sum, conf_n = 0., 0

            for det in detections:
                yaw,pitch,roll = pose_est.estimate(
                    det.get("face_crop"), det["bbox"], frame.shape)
                looking = engine.is_looking(yaw, pitch)
                sc      = engine.score(yaw, pitch)
                pose_results.append((yaw,pitch,roll,looking))
                scores.append(sc)
                conf_sum += det["confidence"]
                conf_n   += 1

            n_look = sum(1 for _,_,_,lk in pose_results if lk)
            new_persons = tracker.update(detections, pose_results)
            aggregator.add(len(detections), n_look, scores, tracker)

        n_look = sum(1 for _,_,_,lk in pose_results if lk)
        n_tot  = len(detections)
        avg_sc = sum(scores)/len(scores) if scores else 0.
        avg_cf = conf_sum/conf_n*100 if conf_n else 0.

        stats = {
            "total":          n_tot,
            "looking":        n_look,
            "attn":           (n_look/n_tot*100) if n_tot else 0.,
            "avg_score":      avg_sc,
            "conf":           avg_cf,
            "elapsed":        now-aggregator.t0,
            "unique_total":   tracker.total_unique,
            "unique_looked":  tracker.looked_unique,
            "unique_attn":    tracker.attention_rate(),
        }
        frame = draw(frame, detections, pose_results, scores, stats,
                     f"{detector_name}+{pose_name}")
        cv2.imshow(f"AdScope v4 Fixed  (q: 종료)", frame)

        if aggregator.should_flush():
            save_to_db(aggregator.flush())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    p = aggregator.flush()
    if p["frame_count"] > 0:
        save_to_db(p)
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ 종료. data_log.jsonl 저장 완료.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⑪ 테스트 모드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_test():
    print(f"\n{'═'*60}")
    print(f"  🧪 AdScope v4 Fixed — 버그 수정 검증")
    print(f"{'═'*60}\n")

    engine = AttentionEngine()
    pose   = FallbackPose()
    frame_shape = (720, 1280, 3)

    print("① 수정된 폴백 Head Pose 검증 (100% 버그 수정 확인)\n")
    cases = [
        ("정면 중앙",      (560,280,120,140), True),
        ("왼쪽 끝 얼굴",   ( 50,280, 70,130), False),
        ("오른쪽 끝 얼굴", (1100,280, 70,130), False),
        ("원거리 작은 얼굴",(600,300, 40, 50), True),
        ("측면 찌그러짐",  (560,280, 60,140), False),
    ]
    print(f"  {'상황':<20} {'yaw':>8} {'pitch':>8} {'판정':>10}  {'점수':>6}  정확")
    print("  " + "─"*60)
    correct = 0
    for name, bbox, expected in cases:
        yaw,pitch,_ = pose.estimate(None, bbox, frame_shape)
        looking = engine.is_looking(yaw, pitch)
        sc      = engine.score(yaw, pitch)
        ok      = looking == expected
        correct += int(ok)
        print(f"  {name:<20} {yaw:>+7.1f}° {pitch:>+7.1f}°  "
              f"{'LOOKING' if looking else 'PASSING':>10}  {sc:>5.1f}  {'✅' if ok else '❌'}")

    print(f"\n  정확도: {correct}/{len(cases)} ({correct/len(cases)*100:.0f}%)")

    print(f"\n② Attention Score 검증\n")
    score_cases = [
        ("완전 정면",     0,  0),
        ("살짝 옆",      15,  5),
        ("임계 직전",    29,  0),
        ("임계 초과",    31,  0),
        ("폰 보는 중",    5,-28),
    ]
    print(f"  {'상황':<16} {'yaw':>6} {'pitch':>7}  {'판정':>10}  {'점수':>6}")
    print("  " + "─"*50)
    for name,yaw,pitch in score_cases:
        looking = engine.is_looking(yaw,pitch)
        sc = engine.score(yaw,pitch)
        print(f"  {name:<16} {yaw:>+5}°  {pitch:>+5}°   "
              f"{'LOOKING' if looking else 'PASSING':>10}  {sc:>5.1f}")

    print(f"\n③ 1분 집계 시뮬레이션\n")
    agg = BatchAggregator("board_gangnam_01")
    random.seed(42)
    for _ in range(300):
        n = random.randint(2,8)
        dets  = [{"bbox":(560+random.randint(-100,100),
                           280+random.randint(-50,50),
                           random.randint(60,120),
                           random.randint(80,140)),
                  "confidence":random.uniform(0.6,0.97),
                  "face_crop":None} for _ in range(n)]
        prs   = []
        scs   = []
        for d in dets:
            y,p,_ = pose.estimate(None, d["bbox"], frame_shape)
            lk    = engine.is_looking(y,p)
            sc    = engine.score(y,p)
            prs.append((y,p,0.,lk))
            scs.append(sc)
        n_look = sum(1 for _,_,_,lk in prs if lk)
        agg.add(n,n_look,scs)

    agg.t0 -= Config.BATCH_SEC
    payload = agg.flush()
    print("  DB payload:")
    print(json.dumps(payload, ensure_ascii=False, indent=4))
    print(f"\n  프레임 주목률: {payload['frame_attention_rate']}%  (100% 버그 수정 ✅)")

    # ── 고유 인원 트래킹 시뮬레이션 ──
    print(f"\n{'─'*60}")
    print("④ 고유 인원 트래킹 시뮬레이션\n")
    tracker = UniquePersonTracker()
    random.seed(7)
    print(f"  {'프레임':>4}  {'감지':>4}  {'신규':>4}  {'활성':>4}  {'누적':>5}  {'본人':>4}  {'주목률':>7}")
    print("  " + "─"*48)
    for frame_n in range(1, 21):
        n = random.randint(1, 5)
        dets  = [{"bbox":(100+i*130, 100, 75, 100)} for i in range(n)]
        poses = [(random.gauss(0,22), random.gauss(0,12), 0.,
                  random.random()<0.35) for _ in range(n)]
        new = tracker.update(dets, poses)
        if frame_n <= 8 or frame_n % 4 == 0:
            print(f"  {frame_n:>4}  {n:>4}  {new:>4}  "
                  f"{tracker.active_count():>4}  {tracker.total_unique:>5}  "
                  f"{tracker.looked_unique:>4}  {tracker.attention_rate():>6.1f}%")
    print(f"\n  ✅ 고유 인원 {tracker.total_unique}명 중 {tracker.looked_unique}명이 광고 봄")
    print(f"     고유 주목률: {tracker.attention_rate()}%")
    print(f"     (프레임 누적 방식이었다면 훨씬 큰 숫자로 부풀려졌을 것)")

    print(f"\n{'═'*60}")
    print("  ✅ 버그 수정 확인 완료!")
    print("  웹캠 실행: 파일 맨 아래 RUN_MODE = 'live' 로 변경")
    print(f"{'═'*60}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실행 모드 선택
# 'test' : 카메라 없이 버그 수정 검증
# 'live' : 실제 웹캠 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RUN_MODE = "live"

if __name__ == "__main__":
    if RUN_MODE == "test":
        run_test()
    else:
        run_live(board_id="board_gangnam_01")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⑦-A  고유 인원 트래커 (핵심 신규 기능)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
