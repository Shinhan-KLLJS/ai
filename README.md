# Loovi

카메라 기반 Vision AI로 오프라인 광고판 앞의 사람을 감지하고 집계하는 프로젝트입니다.

새 구현은 `loovi_vision/` 패키지에서 관리합니다. 상세 설명은 [loovi_vision/README.md](loovi_vision/README.md)를 참고하세요.

현재 구현 범위:

```text
Camera
 → YOLO person detection
 → BoT-SORT / ByteTrack / custom tracking
 → person crop 기반 face detection + gender / age (insightface)
 → head pose 기반 gaze(주목) 판정
 → 1초 단위 JSONL 저장 + 5초 구간 summary (SQS 전송)
 → 오버레이 영상 저장 + 로컬 웹 리뷰
```

face / gaze / realtime 계층은 `configs/person_only.yaml`의 `enable` 플래그로 켜고 끕니다. 모두 `false`면 person-only 동작과 동일합니다.

---

## 실행

```powershell
venv\Scripts\activate
python main.py
```

동일한 실행을 모듈로 직접 호출할 수도 있습니다.

```powershell
python -m loovi_vision.pipelines.person_only --config loovi_vision\configs\person_only.yaml
```

---

## 설정

설정은 YAML로 관리합니다.

```text
loovi_vision/configs/person_only.yaml
```

주요 설정:

```yaml
models:
  person_onnx: models/yolo11l.onnx

detector:
  person_conf_min: 0.50
  iou_threshold: 0.45

tracker:
  backend: botsort   # botsort | bytetrack | custom
  min_hits: 3

runtime:
  batch_sec: 1
  record_video: true
  record_overlay: true
```

---

## 데이터

실행마다 하나의 `run_id`를 만들고, JSONL/영상/세션 메타파일을 같은 이름으로 저장합니다.

```text
data/jsonl/YYMMDD_HHMMSS_person_only.jsonl
data/videos/YYMMDD_HHMMSS_person_only.mp4
data/sessions/YYMMDD_HHMMSS_person_only.json
```

`runtime.batch_sec`를 `1`로 두면 1초 단위로 JSONL이 기록됩니다.

영상 저장은 기본으로 켜져 있습니다.

```yaml
runtime:
  record_video: true
  record_overlay: true   # true: bbox/ID가 그려진 영상, false: 원본 영상
  video_dir: data/videos
  session_dir: data/sessions
  video_fps: 30
```

로컬 리뷰 뷰어를 실행하면 브라우저에서 영상과 JSONL 추이 차트를 함께 볼 수 있습니다.

```powershell
python -m loovi_vision.review.server --port 8765
```

브라우저에서 `http://127.0.0.1:8765`에 접속합니다.

JSONL 예시:

```json
{
  "run_id": "260630_154210_person_only",
  "mode": "person_only",
  "elapsed_start_sec": 0.0,
  "elapsed_end_sec": 1.02,
  "frame_count": 75,
  "frame_detections": 140,
  "avg_persons_per_frame": 1.87,
  "peak_persons": 3,
  "unique_total": 2,
  "active_tracks": 2,
  "tracker_backend": "botsort"
}
```

---

## 구조

```text
loovi_vision/
  config.py / runtime.py    # 설정 로드, ONNX Runtime provider 구성
  detectors/                # person / face / head pose 검출
  tracking/                 # BoT-SORT · ByteTrack · custom tracker
  enrich/                   # face·gaze 보강, 세션 요약(session_summary)
  analysis/                 # 응시 세션 사후(COLD) 분석
  realtime/                 # 5초 구간 스냅샷 SQS 전송(summary)
  pipelines/                # person_only 메인 루프 + 프레임 렌더
  review/                   # 로컬 리뷰 서버(paths·state·handler·media)
  tools/                    # 포즈 캘리브레이션 · SQS 스모크 테스트
  configs/person_only.yaml

main.py                     # 진입점 (python main.py)
models/                     # ONNX 모델 (git 미추적)
data/                       # 실행 산출물 jsonl/videos/sessions (git 미추적)
```

---

## 모델 변환

새 YOLO `.pt` 모델을 받으면 ONNX로 변환한 뒤 YAML에 `.onnx` 경로를 넣습니다.

```powershell
python export_yolo_onnx.py models/yolo11l.pt
```
