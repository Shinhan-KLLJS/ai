import json
import time
from datetime import datetime

import cv2

from loovi_vision.pipelines.threaded_camera import ThreadedCamera
from loovi_vision.pipelines.threaded_writer import ThreadedVideoWriter


def make_run_id(settings):
    # JSONL, 영상, session 메타파일을 같은 basename으로 묶기 위한 실행 ID.
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    return f"{stamp}_{settings.output_suffix}"


def output_path(settings, run_id):
    # 1초 단위 집계 row를 append할 JSONL 파일 경로.
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return settings.output_dir / f"{run_id}.jsonl"


def video_path(settings, run_id):
    # 리뷰용 오버레이(인식결과) 영상을 저장할 MP4 파일 경로.
    settings.video_dir.mkdir(parents=True, exist_ok=True)
    return settings.video_dir / f"{run_id}.mp4"


def raw_video_path(settings, run_id):
    # 원본(무보정) 영상을 저장할 MP4 파일 경로. 오버레이와 겹치지 않게 _raw 접미사를 붙인다.
    settings.video_dir.mkdir(parents=True, exist_ok=True)
    return settings.video_dir / f"{run_id}_raw.mp4"


def session_path(settings, run_id):
    # 한 실행의 설정과 산출물 경로를 묶는 manifest 파일 경로.
    settings.session_dir.mkdir(parents=True, exist_ok=True)
    return settings.session_dir / f"{run_id}.json"


def poses_path(settings, run_id):
    # COLD raw: 매 검출 프레임의 head pose 원시 기록(JSONL) 경로.
    settings.poses_dir.mkdir(parents=True, exist_ok=True)
    return settings.poses_dir / f"{run_id}.jsonl"


def path_text(path):
    return str(path).replace("\\", "/") if path else None


def write_session_manifest(path, payload):
    # 실행 시작 시 running 상태로 쓰고, 종료 시 completed 상태로 덮어쓴다.
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def open_capture(settings):
    # 입력 소스를 연다. camera.video_path 가 있으면 사전 촬영 영상 파일(통제된 실측 시연),
    # 없으면 실시간 웹캠. 두 번째 반환값 source_fps 는 영상 파일일 때만(실시간 페이싱용) 채운다.
    if settings.camera_video:
        cap = cv2.VideoCapture(settings.camera_video)
        return cap, float(cap.get(cv2.CAP_PROP_FPS) or settings.video_fps)
    cap = cv2.VideoCapture(settings.camera_id, settings.camera_backend)
    # 픽셀 포맷 강제(예: MJPG). 무압축으로 열리면 1080p가 대역폭 한계로 프레임 공급이 급락한다.
    # FOURCC는 해상도보다 먼저 지정해야 적용된다. 빈 값이면 강제하지 않음(카메라 기본값 사용).
    if len(settings.camera_fourcc) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*settings.camera_fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_h)
    if settings.camera_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, settings.camera_fps)    # 프레임레이트 요청(카메라가 지원하면 협상됨)
    # 내부 버퍼를 최소화해 최신 프레임 지연을 줄인다(백엔드가 무시할 수 있어 스레드 캡처와 병행).
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    _print_camera_mode(settings, cap)                     # 실제 협상된 해상도/FPS/포맷 확인
    if settings.threaded_capture:
        # 웹캠은 캡처를 별도 스레드로 분리해 추론 속도와 화면 표시를 디커플링한다(라이브 버벅임 제거).
        cap = ThreadedCamera(cap)
    return cap, 0.0


def _print_camera_mode(settings, cap):
    # 실제로 협상된 해상도/FPS/픽셀포맷을 찍어 MJPG·30fps 적용 여부를 확인한다(프레임 공급 진단).
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)).strip()
    print(f"  camera[{settings.camera_id}] {w}x{h} @ {fps:.0f}fps [{fourcc}]")


def pace_realtime(frame_id, source_fps, run_started_at):
    # 사전 촬영 영상을 원래 속도로 재생: 처리가 앞서 나간 만큼만 잠깐 대기한다.
    # 이렇게 하면 시청 시간(elapsed 기준)과 화면 재생 속도가 실시간과 일치한다.
    if source_fps <= 0:
        return
    ahead = frame_id / source_fps - (time.time() - run_started_at)
    if ahead > 0:
        time.sleep(ahead)


def open_video_writer(settings, path, frame_shape):
    # OpenCV VideoWriter는 첫 프레임 크기를 기준으로 고정 크기 MP4를 생성한다.
    height, width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, settings.video_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {path}")
    if settings.threaded_writer:
        # 인코딩을 백그라운드 스레드로 분리해 메인 루프(=화면 표시) 속도를 높인다.
        writer = ThreadedVideoWriter(writer)
    return writer


def build_manifest(settings, run_id, out_path, overlay_path, raw_path, started_text):
    # 실행 설정과 산출물 경로를 담는 session manifest의 초기(running) 상태.
    # video_path 는 리뷰가 재생하는 오버레이 영상, raw_video_path 는 원본 영상 경로.
    return {
        "run_id": run_id,
        "status": "running",
        "board_id": settings.board_id,
        "experiment": settings.experiment_name,
        "mode": "person_only",
        "started_at": started_text,
        "ended_at": None,
        "jsonl_path": path_text(out_path),
        "video_path": path_text(overlay_path),
        "raw_video_path": path_text(raw_path),
        "video_kind": "overlay" if overlay_path else ("raw" if raw_path else None),
        "record_video": settings.record_video,
        "record_overlay": settings.record_overlay,
        "record_raw": settings.record_raw,
        "video_fps": settings.video_fps,
        "frame_width": settings.frame_w,
        "frame_height": settings.frame_h,
        "person_model": path_text(settings.person_onnx),
        "person_conf_min": settings.person_conf_min,
        "iou_threshold": settings.iou_threshold,
        "tracker_backend": settings.tracker_backend,
        "track_min_hits": settings.track_min_hits,
        "face_enabled": settings.face_enable,
        "gaze_enabled": settings.gaze_enable,
        "realtime_enabled": settings.rt_enable,
    }
