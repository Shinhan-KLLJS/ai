# AdScope

Camera-based Vision AI that measures offline advertising ROI in real time.

Detects unique visitors in front of a display, estimates attention rate, and logs demographics (gender, age) — all without storing any video footage.

---

## How It Works

```
Camera → Person Detection → Head Pose → Attention Engine → Unique Tracker → JSONL Log
                ↓
          Face Detection → Gender / Age Estimation
```

| Model | Role |
|---|---|
| YOLOv8n (COCO) | Person bounding box — primary tracking source |
| YOLOv8n-face | Face crop — gaze & gender/age input |
| SixDRepNet360 | Head pose (yaw/pitch/roll) → LOOK / PASS judgment |
| InsightFace genderage | Gender (M/F) + age estimation |

---

## Requirements

- Windows 10/11, Python **3.11** (3.12+ not supported due to torch DLL)
- Webcam

### Fixed dependency versions

```
onnxruntime==1.19.2
numpy==2.1.0
```

Other versions may cause DLL conflicts.

---

## Setup

```bash
# 1. Clone
git clone https://github.com/<your-username>/adscope.git
cd adscope

# 2. Create virtualenv with Python 3.11
py -3.11 -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install opencv-python onnxruntime==1.19.2 numpy==2.1.0 ultralytics
```

---

## Download Models

The large model binaries are excluded from git. Run the steps below once before first launch.

### SixDRepNet360 (head pose)

```bash
python export_6drepnet_v3.py
```

This exports `models/sixdrepnet.onnx` (138 KB structure file) and requires `models/sixdrepnet.onnx.data` (~150 MB weights). The weights are downloaded automatically on first run.

### YOLOv8n (person detection)

```bash
python -c "
from ultralytics import YOLO; import shutil, pathlib
YOLO('yolov8n.pt').export(format='onnx', opset=12, simplify=True)
shutil.move('yolov8n.onnx', 'models/yolov8n.onnx')
print('Done')
"
```

### YOLOv8n-face

Download manually and place at `models/yolov8n-face.onnx`:

```
https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.onnx
```

### InsightFace genderage (antelopev2)

Download manually and place at `models/genderage.onnx`:

```
https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/genderage.onnx
```

---

## Run

```bash
# Activate venv first
venv\Scripts\activate

# Launch (latest version)
python adscope_v7.py
```

Press **q** to quit. Results are appended to `data_log.jsonl`.

---

## Output — data_log.jsonl

Each line is one 15-second batch window (JSON):

| Field | Description |
|---|---|
| `unique_total` | Unique persons detected in the window |
| `unique_looked` | Unique persons who looked at the display |
| `unique_attention_rate` | `unique_looked / unique_total × 100` (%) |
| `unique_male` / `unique_female` | Gender breakdown |
| `avg_age` | Mean estimated age |
| `age_distribution` | Counts per decade (10s / 20s / 30s / 40s / 50plus) |
| `frame_attention_rate` | Frame-level attention rate (how long they looked) |
| `avg_attention_score` | Gaze quality score 0–100 (100 = direct frontal look) |
| `peak_persons` | Max simultaneous persons in a single frame |

See `DATA_LOG_SCHEMA.md` for full field reference.

---

## Project Structure

```
adscope/
├── adscope_v7.py          # Main — Person+Face dual tracking
├── adscope_v6.py          # Previous — Face-only tracking (reference)
├── export_6drepnet_v3.py  # SixDRepNet ONNX export script
├── DATA_LOG_SCHEMA.md     # data_log.jsonl field reference
├── CLAUDE_LOG.md          # Architecture decisions & change log
└── models/
    ├── yolov8n.onnx           # Person detector (gitignored, generate locally)
    ├── yolov8n-face.onnx      # Face detector (gitignored, download manually)
    ├── sixdrepnet.onnx        # Head pose structure file
    ├── sixdrepnet.onnx.data   # Head pose weights (gitignored)
    └── genderage.onnx         # Gender/age estimator
```

---

## Privacy

- No video is stored at any point
- Only aggregated numerical metrics are written to `data_log.jsonl`
- All inference runs locally on-device

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| A | Done | YOLOv8n person bbox tracking |
| B | Planned | OSNet ReID — appearance-based re-identification |
| C | Planned | Kalman filter position prediction |
| Backend | Planned | FastAPI + PostgreSQL/TimescaleDB |
| Dashboard | Planned | Next.js + WebSocket real-time view |
| Edge | Planned | RTSP IP camera, Jetson Nano deployment |
