# Loovi

단일 디렉토리로 관리하는 새 구현입니다. 기능을 하나씩 다시 추가합니다.

현재 구현 범위:

```text
person detection                               # OTS 후보 탐지 (YOLO)
person tracking                                # 동일인 중복 방지 (BoT-SORT)
person-only JSONL logging                      # 1초 단위 집계 row
face enrichment (통행/주목 분리 + 얼굴 기반 성별·연령)   # insightface buffalo_l
gaze / head pose (화면 향함 판정 → LTS)          # DMHead, facing proxy
attention (응시 구간·등급·평균 시청시간)            # gaze_sessions (COLD 사후 분석)
realtime SQS 전송 (summary 5초 구간)             # 수치만 전송, 영상 비전송
overlay video recording                        # 로컬 검증용 (서버 미전송)
local review viewer
```

측정 4계층 대응: **Traffic**(person) → **OTS**(통행/시야각) → **LTS**(head pose 응시)
→ **Attention**(응시 지속시간). 모든 영상 분석은 로컬에서 처리하고 **서버로는 수치만** 보냅니다.

## 실행

실행 모드는 config 로 고른다. `attention.yaml` 은 person+face+gaze+realtime 전체,
`person_only.yaml` 은 사람 검출/추적 + 로컬 기록만 하는 순수 baseline 이다.

```powershell
python -m loovi_vision.pipelines.person_only --config loovi_vision\configs\attention.yaml
```

또는 루트에서:

```powershell
python main.py                                                   # 전체 파이프라인(attention) 기본
python main.py --config loovi_vision\configs\person_only.yaml    # 순수 person-only
```

기본 입력은 실시간 웹캠이다. 통제된 데모(발표 안정성)를 위해 **사전 촬영 영상 파일**을
입력으로 쓰려면 `camera.video_path` 를 지정한다. 영상은 원래 속도로 재생(실시간 페이싱)되어
시청 시간 측정과 화면이 실시간과 일치한다.

```yaml
camera:
  video_path: data/clips/demo.mp4   # 빈 값이면 웹캠
```

## 라이브 성능 (스레드 파이프라인)

라이브 웹캠은 **검출 속도와 화면 표시를 분리**해 화면이 카메라 FPS 로 부드럽게 흐르게 한다.
영상 파일 입력은 결정론적 재생을 위해 이 분리 없이 동기로 처리한다(`capture_loops.sync_loop`).

- **캡처 스레드**(`threaded_capture`, `threaded_camera.py`) — 카메라를 백그라운드에서 쉬지 않고 읽어
  항상 최신 프레임 한 장만 유지한다. 무선 웹캠(Iriun 등)에서 내부 버퍼에 프레임이 쌓여 지연이
  누적되는 문제를 없앤다.
- **검출 워커 스레드**(`detection_worker.py`) — person/face/gaze 추론을 별도 스레드에서 최신 프레임으로
  돌리고(`capture_loops.live_loop`), 메인 루프는 캡처·표시에만 집중한다. 메인은 워커가 발행한 불변
  스냅샷만 읽어 렌더링하므로 락 경합이 없다. 오버레이 bbox 는 마지막 검출 스냅샷이라 빠르게 움직이는
  대상엔 살짝 지연되어 보인다.
- **인코딩 스레드**(`threaded_writer`, `threaded_writer.py`) — mp4 인코딩을 백그라운드로 옮겨 표시
  FPS 를 높인다. 인코더가 뒤처지면 저장 프레임을 드롭한다(라이브·집계엔 영향 없음).

### 카메라 픽셀 포맷

내장 USB 웹캠은 무압축(YUY2)으로 열리면 1080p 가 대역폭 한계로 ~5fps 까지 떨어진다.
`camera.fourcc: MJPG` 로 압축 포맷을 강제하면 프레임 공급이 회복된다. Iriun 같은 가상 카메라는
이미 압축 스트림이라, 문제가 생기면 `fourcc: ""`(빈 값)으로 강제를 해제한다. 실행 시 실제로
협상된 해상도/FPS/포맷을 콘솔에 찍어 확인할 수 있다.

```yaml
camera:
  id: 0            # id=0 내장 웹캠 / id=1 Iriun 등 가상 카메라
  width: 1920
  height: 1080
  fourcc: MJPG     # ""(빈 값)이면 강제 안 함(카메라 기본값)
  fps: 30          # 0이면 강제 안 함
```

### 성능 진단 로그

`runtime.perf_log: true`(`perf_meter.py`)면 검출/추적/얼굴/gaze 단계별 평균 처리시간(ms)과
워커·표시 FPS, 평균 인원을 2초마다 콘솔에 출력한다. 워커가 입력을 기다리는지(=표시 병목)
계산에 매여 있는지(=추론 병목)를 `idle` 값으로 실측해 병목 위치를 가른다.

### CUDA 워밍업

시작 시 사용하는 모든 모델(person + face + head pose)을 더미 입력으로 한 번씩 미리 돌려,
CUDA 커널 초기화/conv 알고리즘 선택 비용을 라이브 중이 아니라 시작 시점에 치른다. conv 탐색은
`HEURISTIC` 으로 지정해 첫 추론 지연 스파이크와 지연 변동을 줄인다(정확도 영향 없음).

## 구조

```text
loovi_vision/
  config.py            # YAML → Settings (카메라/런타임/검출/얼굴/gaze 옵션)
  runtime.py           # ONNX provider 구성 + CUDA conv 튜닝(HEURISTIC)
  tracking/
    base.py / factory.py / custom.py / ultralytics_tracker.py  # BoT-SORT·ByteTrack·custom
  detectors/
    person.py
    face.py            # insightface FaceAnalysis 래퍼 (검출 + 성별/연령)
    headpose.py        # DMHead ONNX 래퍼 (yaw/pitch/roll)
  enrich/
    track_state.py     # 사람별 기록장(통행/주목/best_face/응시 누적) + 레지스트리
    face_enricher.py   # track별 얼굴 검출 → 기록장 갱신 → 종료 시 성별/연령 1회
    gaze.py            # 화면 향함(facing) 판정 (즉석/평활)
    gaze_enricher.py   # 얼굴 crop → head pose → facing → COLD raw 기록 + 응시 누적
    session_summary.py # 세션 요약(주목률, 성별·연령 분포, per_track) + demographics_of
  analysis/
    gaze_sessions.py   # COLD raw → 응시 구간/등급/평균 시청시간 (Attention, 사후)
  realtime/
    summary.py         # 5초 구간: 카운트 + 성별/연령 + Attention (누적 diff)
    sqs_sender.py      # 비동기 SQS 전송(실패 시 백오프·디스크 스필)
  tools/
    calibrate_pose.py  # 수집된 raw pose 분포로 yaw/pitch 임계 캘리브레이션
    sqs_smoke_test.py  # 카메라 없이 SQS 연동만 검증(샘플 1건 전송)
  pipelines/
    person_only.py     # 실행 조립(설정·모델·카메라·루프) 진입 함수 run()
    capture_loops.py   # live_loop(스레드 검출) / sync_loop(영상 파일 동기)
    detection_worker.py# 검출→추적→얼굴/gaze→집계 워커(렌더용 스냅샷 발행)
    threaded_camera.py # 웹캠 캡처 스레드(최신 프레임만 유지)
    threaded_writer.py # 영상 인코딩 스레드(뒤처지면 프레임 드롭)
    perf_meter.py      # 단계별 처리시간·FPS 콘솔 계측
    batch.py           # 1초 단위 집계 row
    gaze_runtime.py    # 2차 런타임: summary(5초 구간) 집계·전송 묶음
    frame_render.py    # 스냅샷/상태 → 오버레이 프레임 조립
    overlay.py         # 사람/얼굴 박스 + head pose 오버레이
    session_io.py      # run_id/경로/manifest IO + 카메라·라이터 오픈
  configs/
    attention.yaml     # 전체 파이프라인 (기본)
    person_only.yaml   # 순수 person-only baseline
  review/
    server.py
```

## 데이터

실행마다 하나의 `run_id`를 만들고, JSONL/영상/세션 메타파일을 같은 이름으로 저장합니다.

```text
data/jsonl/YYMMDD_HHMMSS_person_only.jsonl
data/videos/YYMMDD_HHMMSS_person_only.mp4
data/sessions/YYMMDD_HHMMSS_person_only.json
```

기본 설정은 1초 단위 수집입니다.

```yaml
runtime:
  batch_sec: 1
```

영상 저장은 기본으로 켜져 있습니다.

```yaml
runtime:
  record_video: true
  record_overlay: true
  video_dir: data/videos
  session_dir: data/sessions
```

리뷰 뷰어는 로컬 웹 서버로 실행합니다.

```powershell
python -m loovi_vision.review.server --port 8765
```

브라우저에서 `http://127.0.0.1:8765`에 접속하면 세션 목록, 영상, 1초 단위 추이 차트, JSONL row 테이블을 함께 볼 수 있습니다.
얼굴 분석이 켜진 세션은 주목도 요약 카드(통행/주목/주목률)와 성별·연령 분포가 함께 표시됩니다.

## 얼굴 분석 (face enrichment)

`insightface` 의 `FaceAnalysis` (buffalo_l 팩) 하나로 얼굴 검출 + 성별 + 연령을 처리합니다.
모든 사람은 "통행"으로 집계(분모)하고, 얼굴이 한 번이라도 잡힌 사람만 "주목"으로 분류(분자)합니다.
얼굴이 끝까지 안 잡힌 사람은 버리지 않고 "성별·연령 미상(null)" 상태로 남깁니다.

설치(최초 실행 시 buffalo_l 모델 팩을 `~/.insightface/models/` 로 자동 다운로드):

```powershell
pip install -r requirements.txt
# insightface 가 의존성으로 CPU onnxruntime 을 끌어온다. GPU 충돌 방지를 위해 제거:
pip uninstall -y onnxruntime
# GPU provider 확인 (CUDAExecutionProvider 가 보여야 함):
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

> ⚠️ `onnxruntime`(CPU) 과 `onnxruntime-gpu` 를 **동시에 설치하면 충돌**한다.
> 둘 다 깔리면 `CUDAExecutionProvider` 를 요청해도 CPU 로 느리게 돌 수 있으니
> 반드시 하나만 남긴다 (GPU 환경: `onnxruntime-gpu` 만).

설정(`loovi_vision/configs/attention.yaml`):

```yaml
models:
  face_pack: buffalo_l
face:
  enable: true               # false 로 바꾸면 얼굴 분석 없이 person-only 동작
  conf_min: 0.5
  run_every_n_frames: 2      # track 당 얼굴 검출 주기 (작을수록 pose 샘플 증가)
  min_crop_size: 48          # 너무 작은 사람 crop 은 건너뜀
  det_size: [640, 640]       # [320,320] 로 낮추면 얼굴 검출이 빨라짐(먼 얼굴은 놓칠 수 있음)
  max_per_frame: 0           # 한 프레임에 얼굴 분석할 최대 인원(0=무제한). 사람 많을 때 3~4로 제한하면 표시 FPS 방어
```

산출물:

```text
JSONL(1초)  : persons_with_face, face_visible_ratio 추가
session json: total_unique, face_visible_count, face_visible_rate,
              gender_dist, age_dist, per_track
```

성능을 위해 얼굴 검출은 track 당 `run_every_n_frames` 마다, 성별·연령 판정은 사람당
best_face(= conf × 얼굴 면적 최대) 기준 세션 종료 시 1회만 수행합니다.
best_face crop 원본은 2차 head pose 입력 재사용을 위해 보관합니다.

사람이 많은 장면에서 얼굴 모델 호출이 폭주하면 표시 FPS 가 떨어집니다. `face.max_per_frame` 을
양수로 두면 그 프레임에서 crop 이 큰(=가까운) 상위 K명만 얼굴 분석합니다(나머지는 통행 집계엔 그대로 포함).

실행 시 얼굴 검출·성별/연령 세션이 **실제로 잡은 provider(CUDA/CPU)**를 콘솔에 찍습니다.
insightface 가 조용히 CPU 로 폴백하면 여기서 드러나며(느림 경고 출력), `onnxruntime-gpu` 설치·충돌을 점검하세요.

## 화면 응시 추적 (gaze, 2차) + Attention

head pose(DMHead)로 "화면(광고판)을 향했나"를 판정해 응시(LTS)를 추적하고, 응시가 지속된
시간(Attention)을 구간화합니다. 시선 추적이 아니라 head pose 기반 proxy 입니다.

- **HOT(실시간)**: 단일 `summary` 메시지를 비동기 SQS 로 전송한다 (수치만, 영상·좌표 미전송).
  기본 **5초마다** 그 구간(window)의 OTS/LTS 카운트 + 성별/연령 분포(OTS·LTS, 연령은 성별별 분리) +
  Attention(평균 시청시간·시청시간 합·분포)을 담는다. 매 주기 누적을 재집계해 직전과 diff 하는
  방식(`realtime/summary.py`). 값은 전부 **그 구간값**이라 서버가 여러 구간을 합쳐 시간대 롤업한다.
- **COLD(기록)**: 매 검출 프레임 raw(pose 각도 + 위치/크기/low_conf)를 `data/poses/*.jsonl` 에
  날것 그대로 저장. 구간화·등급·거리보정은 사후 분석(`analysis/gaze_sessions.py`).

전송 주기는 설정으로 조절한다(광고 노출 15~30초 → 5초=서브광고 해상도).

```yaml
realtime:
  summary_interval_sec: 5    # summary(구간별: 카운트 + 성별/연령 + Attention) 전송 주기(초)
```

Attention(응시 지속)은 연속 facing 구간을 `gap_tol_sec` 로 병합/분리해 세션을 만들고,
`grade_glance_sec` 미만 세션은 노이즈로 제외한다. **평균·분포는 세션이 아니라 "LTS 인원 기준"**
으로 확정한다: 한 사람의 여러 세션을 누적 응시시간(`total_gaze_sec`)으로 합쳐 1인 1대표값을 만들고,
LTS(누적 응시 >= `lts_min_sec`)인 사람만 평균/분포 대상에 넣는다. 즉 `평균 시청 = Σ(LTS 인원별 누적) ÷ LTS 인원 수`.

```yaml
gaze:
  gap_tol_sec: 0.5           # 이 이내 끊김은 같은 세션으로 이어붙임
  grade_glance_sec: 0.2      # 이 미만 세션은 노이즈로 제외 (최소 유효 시청)
  grade_view_sec: 1.0        # 세션 등급 경계(glance/view/dwell). 분포 버킷은 고정 정수초(2·3·4s)
  grade_dwell_sec: 2.0       # 이 이상 누적 응시 = "Attention(2초+ 응시자)"
  lts_min_sec: 1.0           # 누적 응시 이 초 이상이면 LTS(응시자)로 카운트
```

산출 지표: `avg_dwell_sec`(인원 기준 평균), `attention_count`(2초+ 응시자 수),
`dwell_distribution`(LTS 인원의 시청시간 분포: **1~2 / 2~3 / 3~4 / 4초 이상** 4구간).
LTS 필터(>= 1초)라 "1초 미만" 구간은 정의상 항상 0이므로 제외했다.

필요한 것:

```text
1) models/dmhead.onnx  : PINTO0309/DMHead 릴리스의 DMHead ONNX (pip 비제공, 직접 배치)
2) boto3               : SQS 전송용 (realtime.enable=false 면 불필요)
3) 캘리브레이션         : yaw_center/pitch_center/tol 실측 (아래 도구)
```

### SQS 환경 세팅

`summary` 메시지를 실제로 전송하려면 큐·리전·자격증명 3가지가 필요하다.

```text
1) 큐 URL     : realtime.sqs_queue_url (config) 또는 환경변수 LOOVI_SQS_QUEUE_URL
2) 리전       : realtime.sqs_region (예: ap-northeast-2) 또는 환경변수 AWS_REGION / AWS_DEFAULT_REGION
3) 자격증명   : boto3 표준 체인 — 환경변수(AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY) 또는 IAM Role / ~/.aws
```

카메라·모델 없이 연동만 먼저 검증하려면 스모크 테스트로 샘플 1건을 동기 전송한다
(성공 시 `MessageId`, 실패 시 원인 그대로 출력):

```bash
python -m loovi_vision.tools.sqs_smoke_test --config loovi_vision/configs/attention.yaml
```

캘리브레이션 보조 도구(수집된 raw 의 yaw/pitch 분포와 거리 구간별 분포 출력):

```powershell
python -m loovi_vision.tools.calibrate_pose            # data/poses 최신 파일 분석
```

사후 응시 구간/등급 재분석은 COLD raw 로 수행하므로, 임계값(`gaze.*`)을 바꿔도
raw 재수집 없이 결과만 달라집니다. SQS queue URL 은 `realtime.sqs_queue_url` 또는
환경변수 `LOOVI_SQS_QUEUE_URL` 로 주입하며, 미설정/전송 실패 시 `data/outbox/` 로 스필됩니다.

산출물:

```text
data/poses/*.jsonl   : 매 프레임 raw pose (COLD, 날것)
data/jsonl/*.jsonl   : 1초 집계 (1차 필드 + concurrent_gazers/ots/lts)
data/sessions/*.json : 세션 요약(응시 구간/등급, per_track_gaze, avg_dwell_sec) + 성능/전송 지표
data/outbox/*.json   : SQS 미전송 메시지 스필
SQS                  : summary(5초 구간, 카운트+성별·연령(성별별 연령)+Attention)
```

서버로 나가는 메시지 스키마(단일 `summary`, 기본 5초):

```jsonc
// summary (5초 구간) — 메시지는 이 한 종류뿐이라 type 필드 없음
{ "device_id":"...", "board_id":"...", "seq":1,
  "timestamp":"...", "interval_sec":5.0,
  "ots_count":8, "lts_count":3,
  "ots_demographics":{ "male":{"count":4,"age":{"20s":2,"...":0}},
                       "female":{"count":3,"age":{"20s":2,"...":0}} },
  "lts_demographics":{ "male":{"count":2,"age":{"20s":1,"...":0}},
                       "female":{"count":1,"age":{"20s":1,"...":0}} },
  "attention":{ "avg_dwell_sec":1.6, "dwell_sum_sec":4.8,
                "dwell_distribution":{"1_to_under_2s":2,"2_to_under_3s":1,"3_to_under_4s":0,"4s_and_over":0} } }
```

> 필드별 정의·서버 연동 가이드는 [docs/vision-summary-sqs-guide.md](../docs/vision-summary-sqs-guide.md),
> 검증용 JSON Schema는 [docs/vision-summary-schema.json](../docs/vision-summary-schema.json)를 정본으로 한다.

> 끄려면 `gaze.enable: false`(1차와 100% 동일) 또는 `realtime.enable: false`(로컬 기록만, SQS 없음).

## 다음 단계

1. 전신(person bbox) 기반 통행자 전체 성별·연령 (PAR 계열) — 현재는 얼굴 잡힌 통행자만
2. Attention 상한(cap)·이상치 제외 (정지 인물/마네킹/오검출로 인한 비현실적 장시간 시청)
3. tracking ID 재식별(ReID) — occlusion 으로 ID 끊길 때 응시 시간 분절/중복 완화
