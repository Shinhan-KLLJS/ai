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

실행 모드는 config 파일로 고릅니다.

- `configs/attention.yaml` — person + face + gaze + realtime 전체 파이프라인 (`python main.py`의 기본값)
- `configs/person_only.yaml` — 사람 검출/추적 + 로컬 기록만 하는 순수 baseline

두 파일은 face / gaze / realtime 계층의 `enable` 플래그만 다릅니다. 세부 계층을 개별로 켜고 끄려면 config의 `enable` 값을 직접 바꾸면 됩니다.

---

## 실행

```powershell
venv\Scripts\activate
python main.py                                                   # 전체 파이프라인(attention) 기본
python main.py --config loovi_vision\configs\person_only.yaml    # 순수 person-only
```

동일한 실행을 모듈로 직접 호출할 수도 있습니다.

```powershell
python -m loovi_vision.pipelines.person_only --config loovi_vision\configs\attention.yaml
```

---

## 설정

설정은 YAML로 관리합니다.

```text
loovi_vision/configs/attention.yaml     # 전체 파이프라인 (기본)
loovi_vision/configs/person_only.yaml   # 순수 person-only baseline
```

주요 설정:

```yaml
models:
  person_onnx: models/yolo11l.onnx

camera:
  id: 0              # id=0 내장 웹캠 / id=1 Iriun 등 가상 카메라
  width: 1920
  height: 1080
  fourcc: MJPG       # 픽셀 포맷 강제. 무압축(YUY2)이면 1080p가 대역폭 한계로 ~5fps로 떨어짐
  fps: 30            # 0이면 강제 안 함(카메라 기본값)

detector:
  person_conf_min: 0.50
  iou_threshold: 0.45

tracker:
  backend: botsort   # botsort | bytetrack | custom
  min_hits: 3

runtime:
  batch_sec: 1
  threaded_capture: true   # 웹캠 캡처를 별도 스레드로 분리(라이브 버벅임 제거)
  threaded_writer: true    # 영상 인코딩을 별도 스레드로 분리(표시 FPS 향상)
  perf_log: true           # 단계별 처리시간·FPS를 콘솔에 출력(진단)
  record_video: true
  record_overlay: true
```

---

## 라이브 파이프라인 (스레드 분리)

라이브 웹캠은 **검출 속도와 화면 표시를 분리**해 화면이 카메라 FPS로 부드럽게 흐르도록 처리합니다.
영상 파일 입력(`camera.video_path`)은 결정론적 재생을 위해 스레드 분리 없이 동기로 처리합니다.

- **캡처 스레드**(`threaded_capture`) — 카메라를 백그라운드에서 계속 읽어 항상 최신 프레임만 소비합니다. 무선 웹캠에서 버퍼에 프레임이 쌓여 지연이 누적되는 문제를 없앱니다.
- **검출 워커 스레드** — person/face/gaze 추론을 별도 스레드에서 최신 프레임으로 돌리고, 메인 루프는 캡처·표시에만 집중합니다. 오버레이 bbox는 마지막 검출 스냅샷이라 빠른 움직임엔 살짝 지연됩니다.
- **인코딩 스레드**(`threaded_writer`) — 영상 저장(mp4 인코딩)을 백그라운드로 옮겨 표시 FPS를 높입니다. 인코더가 뒤처지면 저장 프레임을 드롭합니다(라이브·집계엔 영향 없음).

내장 웹캠은 무압축(YUY2)으로 열리면 1080p가 대역폭 한계로 ~5fps까지 떨어지므로 `camera.fourcc: MJPG`로 압축 포맷을 강제합니다. `perf_log: true`면 단계별 처리시간과 워커·표시 FPS를 2초마다 콘솔에 출력해 병목 위치를 실측할 수 있습니다. 세부 설명은 [loovi_vision/README.md](loovi_vision/README.md#라이브-성능-스레드-파이프라인)를 참고하세요.

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
  pipelines/                # 검출 워커·라이브/동기 캡처 루프·프레임 렌더 (스레드 분리)
  review/                   # 로컬 리뷰 서버(paths·state·handler·media)
  tools/                    # 포즈 캘리브레이션 · SQS 스모크 테스트
  configs/attention.yaml    # 전체 파이프라인 (기본)
  configs/person_only.yaml  # 순수 person-only baseline

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
