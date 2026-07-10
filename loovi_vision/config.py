from pathlib import Path
import os

import cv2
import yaml


def load_env_file(path=".env"):
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def env_value(*names, default=None):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


# YAML 설정 파일을 읽어 dict로 반환한다.
def load_config(path):
    load_env_file()
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# "runtime.batch_sec"처럼 점으로 구분된 경로를 안전하게 읽는다.
def get(config, path, default=None):
    node = config
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


class Settings:
    # YAML dict를 런타임에서 쓰기 편한 타입과 기본값으로 정규화한다.
    def __init__(self, config):
        # 실험/저장 경로: 같은 run_id로 JSONL, 영상, 세션 메타파일을 묶는다.
        self.experiment_name = str(get(config, "experiment.name", "person_only"))
        self.board_id = str(env_value("board_id", "BOARD_ID", default=get(config, "experiment.board_id", "board_gangnam_01")))
        self.output_dir = Path(get(config, "experiment.output_dir", "data/jsonl"))
        self.output_suffix = str(get(config, "experiment.output_suffix", self.experiment_name))

        # 현재 파이프라인은 YOLO ONNX person detector 하나를 사용한다.
        self.person_onnx = Path(get(config, "models.person_onnx", "models/yolo11l.onnx"))
        # insightface 얼굴 분석 모델 팩 (buffalo_l = 검출 + 성별/연령).
        self.face_pack = str(get(config, "models.face_pack", "buffalo_l"))
        # head pose 전용 모델 (DMHead ONNX). gaze.enable=true일 때만 로드한다.
        self.headpose_onnx = Path(get(config, "models.headpose_onnx", "models/dmhead.onnx"))

        self.camera_id = int(get(config, "camera.id", 0))
        # 값이 있으면 웹캠 대신 사전 촬영 영상 파일을 입력으로 쓴다(통제된 데모용). 빈 값이면 웹캠.
        self.camera_video = str(get(config, "camera.video_path", "") or "")
        # Windows에서는 MSMF가 열리더라도 read에 실패할 수 있어 DirectShow를 고정한다.
        self.camera_backend = cv2.CAP_DSHOW
        self.frame_w = int(get(config, "camera.width", 1280))
        self.frame_h = int(get(config, "camera.height", 720))
        # 웹캠 픽셀 포맷/프레임레이트 강제. 내장 USB 웹캠은 무압축(YUY2)이면 1080p가 대역폭 한계로
        # ~5fps로 떨어지므로 MJPG 권장. fourcc="" 면 강제 안 함(카메라 기본값), fps<=0 이면 fps 강제 안 함.
        # Iriun 같은 가상 카메라는 이미 압축 스트림이라, 문제가 생기면 fourcc:"" 로 두면 된다.
        self.camera_fourcc = str(get(config, "camera.fourcc", "MJPG"))
        self.camera_fps = float(get(config, "camera.fps", 30.0))

        # 처리/저장 옵션: process_every_n은 추론 빈도, batch_sec은 집계 window다.
        self.enable_cuda = bool(get(config, "runtime.enable_cuda", True))
        # 웹캠 캡처를 별도 스레드로 분리(최신 프레임만 소비)해 라이브 화면 버벅임을 없앤다.
        # 영상 파일 입력에는 적용되지 않는다(프레임 순차 재생 유지).
        self.threaded_capture = bool(get(config, "runtime.threaded_capture", True))
        self.process_every_n = int(get(config, "runtime.process_every_n", 2))
        # 트래킹 분리: >0이면 detect+track은 매 프레임 돌리고 무거운 얼굴/포즈 보강만 이 간격(초)으로만 수행한다.
        # 트래커가 얼굴 비용에 발목잡히지 않아 프레임 간격이 좁아지고 OTS/카운트 신뢰가 회복된다. 0이면 매 프레임(기존 동작).
        self.enrich_interval_sec = float(get(config, "runtime.enrich_interval_sec", 0.0))
        self.batch_sec = int(get(config, "runtime.batch_sec", 15))
        self.show_window = bool(get(config, "runtime.show_window", True))
        self.save_frame_samples = bool(get(config, "runtime.save_frame_samples", False))
        self.record_video = bool(get(config, "runtime.record_video", False))
        self.record_overlay = bool(get(config, "runtime.record_overlay", True))
        # 인식결과 오버레이 영상과 별개로 원본(무보정) 영상도 함께 저장할지 여부.
        # true면 {run_id}_raw.mp4 로 원본을 추가 저장한다(오버레이 저장과 독립적으로 켤 수 있음).
        self.record_raw = bool(get(config, "runtime.record_raw", False))
        # 영상 인코딩을 별도 스레드로 분리해 메인 루프(=화면 표시) 속도를 높인다.
        self.threaded_writer = bool(get(config, "runtime.threaded_writer", True))
        # 워커 단계별 처리시간(검출/추적/얼굴/gaze ms)을 주기적으로 콘솔에 출력한다(성능 진단).
        self.perf_log = bool(get(config, "runtime.perf_log", True))
        self.video_dir = Path(get(config, "runtime.video_dir", "data/videos"))
        self.session_dir = Path(get(config, "runtime.session_dir", "data/sessions"))
        self.video_fps = float(get(config, "runtime.video_fps", 30.0))

        # detector threshold와 bbox 크기 필터. 현장 튜닝의 주요 대상이다.
        self.person_conf_min = float(get(config, "detector.person_conf_min", 0.50))
        self.iou_threshold = float(get(config, "detector.iou_threshold", 0.45))
        self.person_min_height = int(get(config, "detector.person_min_height", 40))
        self.person_min_area_ratio = float(get(config, "detector.person_min_area_ratio", 0.002))
        self.person_max_area_ratio = float(get(config, "detector.person_max_area_ratio", 0.70))

        # tracking backend와 unique count 안정화 파라미터.
        self.tracker_backend = str(get(config, "tracker.backend", "botsort")).lower()
        self.tracker_gmc_method = str(get(config, "tracker.gmc_method", "none"))
        self.track_min_hits = int(get(config, "tracker.min_hits", 3))
        self.track_box_smooth_alpha = float(get(config, "tracker.box_smooth_alpha", 0.65))
        # BoT-SORT/ByteTrack 내부 임계값. 현장 튜닝 대상이라 하드코딩 대신 config로 노출한다.
        self.track_high_thresh = float(get(config, "tracker.high_thresh", 0.25))
        self.track_low_thresh = float(get(config, "tracker.low_thresh", 0.10))
        self.new_track_thresh = float(get(config, "tracker.new_track_thresh", 0.25))
        self.track_buffer = int(get(config, "tracker.track_buffer", 30))
        self.track_match_thresh = float(get(config, "tracker.match_thresh", 0.80))
        self.track_proximity_thresh = float(get(config, "tracker.proximity_thresh", 0.50))
        self.track_appearance_thresh = float(get(config, "tracker.appearance_thresh", 0.80))
        # 매칭 실패 track을 몇 프레임 유지 후 제거할지(공통), custom tracker의 최대 매칭 거리(px).
        self.track_max_missing = int(get(config, "tracker.max_missing", 45))
        self.custom_match_max_dist = float(get(config, "tracker.custom_match_max_dist", 260.0))
        # body Re-ID(외형 임베딩): 켜면 끊긴 track 재결합으로 중복 통행 집계를 줄인다(BoT-SORT 전용).
        # appearance_thresh 는 with_reid=true 일 때만 실제로 쓰인다. 커스텀 ONNX detector라 전용 ReID 모델 필요.
        # 주의: detection 마다 외형 추론이 추가돼 느려진다(특히 CPU). 실험/현장 검증용.
        self.track_with_reid = bool(get(config, "tracker.with_reid", False))
        self.track_reid_model = str(get(config, "tracker.reid_model", "models/yolo26n-reid.onnx"))

        # face enrichment: enable=false면 기존 person_only와 100% 동일 동작.
        self.face_enable = bool(get(config, "face.enable", False))
        self.face_conf_min = float(get(config, "face.conf_min", 0.5))
        # track당 얼굴 검출 주기(처리 프레임 기준)와 너무 작은 crop 컷오프.
        self.face_run_every_n = int(get(config, "face.run_every_n_frames", 3))
        self.face_min_crop_size = int(get(config, "face.min_crop_size", 48))
        # 한 프레임에 얼굴 분석을 돌릴 최대 인원(가까운=큰 crop 우선). 0이면 무제한(기존 동작).
        # 사람이 많을 때 워커 iteration 폭주를 막아 표시 FPS를 지킨다.
        self.face_max_per_frame = int(get(config, "face.max_per_frame", 0))
        det_size = get(config, "face.det_size", [640, 640])
        self.face_det_size = (int(det_size[0]), int(det_size[1]))
        # 성별/연령 누적 시 얼굴 품질 가중치의 선명도(라플라시안 분산) 기준값.
        # 0(기본)이면 선명도 미적용(가중치=conf×area). >0이면 흐린 얼굴 표를 min(1, 선명도/기준)로 깎는다.
        # 매직 상수를 피하려 기본은 꺼두고, 평가셋으로 최적 기준을 정한 뒤 켜는 것을 권장한다.
        self.face_quality_sharp_ref = float(get(config, "face.quality_sharp_ref", 0.0))

        # gaze (2차): head pose 기반 화면 향함 판정. enable=false면 1차와 100% 동일.
        self.gaze_enable = bool(get(config, "gaze.enable", False))
        self.gaze_yaw_center = float(get(config, "gaze.yaw_center", 0.0))      # [캘리브레이션 실측]
        self.gaze_pitch_center = float(get(config, "gaze.pitch_center", 0.0))  # [캘리브레이션 실측]
        self.gaze_yaw_tol = float(get(config, "gaze.yaw_tol", 20.0))           # [시야각 기반]
        self.gaze_pitch_tol = float(get(config, "gaze.pitch_tol", 15.0))
        self.gaze_pose_min_face_px = int(get(config, "gaze.pose_min_face_px", 40))
        self.gaze_low_conf_policy = str(get(config, "gaze.low_conf_policy", "exclude"))
        # LTS(Likely To See): 누적 응시 시간이 이 초 이상인 track을 응시자로 본다.
        self.gaze_lts_min_sec = float(get(config, "gaze.lts_min_sec", 1.0))
        # 평활(HOT 전용): 최근 window 동안 facing 우세면 응시로 본다.
        # window 는 pose 샘플 주기의 2배 이상이어야 비율 판정이 의미를 갖는다. config_validate 가 검사한다.
        self.gaze_smooth_window_sec = float(get(config, "gaze.smooth_window_sec", 1.5))
        self.gaze_smooth_min_ratio = float(get(config, "gaze.smooth_min_ratio", 0.5))
        # 구간 분석(COLD, 사후)용 임계값. raw 저장 시점에 박지 않는다.
        # gap_tol_sec 은 pose 샘플 주기(enrich_interval_sec × face_run_every_n)보다 커야 한다.
        # 짧으면 연속 응시가 샘플마다 끊겨 Attention 이 0 으로 주저앉는다. config_validate 가 검사한다.
        self.gaze_gap_tol_sec = float(get(config, "gaze.gap_tol_sec", 1.0))
        self.gaze_grade_glance_sec = float(get(config, "gaze.grade_glance_sec", 0.2))
        self.gaze_grade_view_sec = float(get(config, "gaze.grade_view_sec", 1.0))
        self.gaze_grade_dwell_sec = float(get(config, "gaze.grade_dwell_sec", 2.0))
        self.poses_dir = Path(get(config, "gaze.poses_dir", "data/poses"))

        # realtime (2차): 1초 스냅샷 SQS 전송. enable=false면 로컬 기록만.
        self.rt_enable = bool(get(config, "realtime.enable", False))
        self.rt_device_id = str(env_value("device_id", "DEVICE_ID", default=get(config, "realtime.device_id", "loovi-cam-01")))
        self.rt_sqs_queue_url = str(env_value("SQS_QUEUE_URL", "LOOVI_SQS_QUEUE_URL", default=get(config, "realtime.sqs_queue_url", "")))
        # AWS 리전. 빈 값이면 boto3 기본 해석(AWS_REGION -> AWS_DEFAULT_REGION -> ~/.aws/config).
        self.rt_sqs_region = str(env_value("AWS_REGION", "AWS_DEFAULT_REGION", default=get(config, "realtime.sqs_region", "")))
        # summary(구간별: 카운트 + 성별/연령 + Attention) 전송 주기(기본 5초).
        self.rt_summary_interval = float(get(config, "realtime.summary_interval_sec", 5.0))
        # 응시 "종료" 판정: 마지막으로 보인 지 이 초를 넘으면 끝난 것으로 보고 시청시간 분포에 1회 계상한다.
        # (트래커가 track을 놓치는 시간대와 비슷하게 두는 게 안전. 너무 짧으면 잠깐 가림에 조기 종료.)
        self.rt_exit_grace_sec = float(get(config, "realtime.exit_grace_sec", 2.0))
        self.rt_buffer_max = int(get(config, "realtime.buffer_max", 600))
        self.rt_spill_dir = Path(get(config, "realtime.spill_dir", "data/outbox"))
        self.rt_clock = str(get(config, "realtime.clock", "utc"))
