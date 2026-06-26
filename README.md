# AdScope

카메라 기반 Vision AI로 오프라인 광고판의 효과를 실시간으로 측정하는 솔루션입니다.

광고판 앞을 지나는 고유 방문자를 감지하고, 주목률과 인구통계(성별, 연령)를 집계합니다. 영상은 일절 저장하지 않으며, 수치 데이터만 로컬에 기록합니다.

---

## 동작 원리

```
카메라 → 전신 감지 → 헤드 포즈 → 주목 판정 → 고유 인원 트래킹 → JSONL 저장
              ↓
         얼굴 감지 → 성별 / 연령 추정
```

| 모델 | 역할 |
|---|---|
| YOLOv8n (COCO) | 전신 바운딩 박스 — 주 트래킹 소스 |
| YOLOv8n-face | 얼굴 crop — 시선 판단 및 성별/연령 입력 |
| SixDRepNet360 | 헤드 포즈(yaw/pitch/roll) → LOOK / PASS 판정 |
| InsightFace genderage | 성별(M/F) + 연령 추정 |

---

## 요구 사항

- Windows 10/11, Python **3.11** (3.12 이상은 torch DLL 미지원)
- 웹캠

### 고정 패키지 버전

```
onnxruntime==1.19.2
numpy==2.1.0
```

다른 버전은 DLL 충돌이 발생할 수 있습니다.

---

## 설치

```bash
# 1. 클론
git clone https://github.com/Shinhan-KLLJS/ai.git
cd ai

# 2. Python 3.11 가상환경 생성
py -3.11 -m venv venv
venv\Scripts\activate

# 3. 패키지 설치
pip install opencv-python onnxruntime==1.19.2 numpy==2.1.0 ultralytics
```

---

## 모델 파일 준비

용량이 큰 모델 바이너리는 git에서 제외되어 있습니다. 최초 실행 전 아래 순서대로 준비하세요.

### SixDRepNet360 (헤드 포즈)

```bash
python export_6drepnet_v3.py
```

`models/sixdrepnet.onnx` (138KB 구조 파일)와 `models/sixdrepnet.onnx.data` (~150MB 가중치)가 자동 생성됩니다.

### YOLOv8n (전신 감지)

```bash
python -c "
from ultralytics import YOLO; import shutil, pathlib
YOLO('yolov8n.pt').export(format='onnx', opset=12, simplify=True)
shutil.move('yolov8n.onnx', 'models/yolov8n.onnx')
print('완료')
"
```

### YOLOv8n-face (얼굴 감지)

아래 링크에서 다운로드 후 `models/yolov8n-face.onnx`로 저장:

```
https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.onnx
```

### InsightFace genderage (성별/연령)

아래 링크에서 다운로드 후 `models/genderage.onnx`로 저장:

```
https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/genderage.onnx
```

---

## 실행

```bash
# 가상환경 활성화 후
venv\Scripts\activate

# 최신 버전 실행
python adscope_v7.py
```

종료는 카메라 화면에서 **q** 키를 누릅니다. 결과는 `data_log.jsonl`에 누적 저장됩니다.

---

## 출력 형식 — data_log.jsonl

15초 단위로 한 줄씩 JSON이 기록됩니다:

| 필드 | 설명 |
|---|---|
| `unique_total` | 윈도우 내 감지된 고유 인원 수 |
| `unique_looked` | 광고판을 바라본 고유 인원 수 |
| `unique_attention_rate` | 주목률 (%) = `unique_looked / unique_total × 100` |
| `unique_male` / `unique_female` | 성별 분류 |
| `avg_age` | 추정 평균 연령 |
| `age_distribution` | 연령대별 인원 (10대 / 20대 / 30대 / 40대 / 50대 이상) |
| `frame_attention_rate` | 프레임 기준 주목률 (얼마나 오래 봤는가) |
| `avg_attention_score` | 시선 품질 점수 0~100 (100 = 완전 정면 주목) |
| `peak_persons` | 단일 프레임 최대 동시 감지 인원 |

전체 필드 설명은 `DATA_LOG_SCHEMA.md`를 참고하세요.

---

## 프로젝트 구조

```
adscope/
├── adscope_v7.py          # 메인 — 전신+얼굴 이중 트래킹
├── adscope_v6.py          # 이전 버전 — 얼굴 전용 트래킹 (참고용)
├── export_6drepnet_v3.py  # SixDRepNet ONNX 익스포트 스크립트
├── DATA_LOG_SCHEMA.md     # data_log.jsonl 필드 상세 설명
├── CLAUDE_LOG.md          # 아키텍처 결정 및 변경 이력
└── models/
    ├── yolov8n.onnx           # 전신 감지 (gitignore — 로컬 생성)
    ├── yolov8n-face.onnx      # 얼굴 감지 (gitignore — 수동 다운로드)
    ├── sixdrepnet.onnx        # 헤드 포즈 구조 파일
    ├── sixdrepnet.onnx.data   # 헤드 포즈 가중치 (gitignore)
    └── genderage.onnx         # 성별/연령 추정 모델
```

---

## 개인정보 보호

- 영상은 어떠한 형태로도 저장되지 않습니다
- `data_log.jsonl`에는 집계된 수치 데이터만 기록됩니다
- 모든 추론은 로컬 디바이스에서 실행됩니다

---

## 개발 로드맵

| Phase | 상태 | 내용 |
|---|---|---|
| A | 완료 | YOLOv8n 전신 bbox 트래킹 |
| B | 예정 | OSNet ReID — 외형 기반 동일인 재인식 |
| C | 예정 | Kalman Filter 위치 예측 |
| 백엔드 | 예정 | FastAPI + PostgreSQL/TimescaleDB |
| 대시보드 | 예정 | Next.js + WebSocket 실시간 화면 |
| 엣지 배포 | 예정 | RTSP IP카메라, Jetson Nano 지원 |
