"""설정 정합성 검증.

서로 다른 섹션의 값이 곱해져서 조용히 깨지는 관계를 실행 전에 잡아낸다.
값을 고쳐주지 않고 경고만 낸다(실행은 계속). Settings 를 import 하지 않는 순수 함수라
파이프라인/스윕/테스트 어디서든 재사용할 수 있다.
"""

# camera.fps<=0 이면 카메라 기본값을 쓰므로 실제 fps 를 알 수 없다. 보수적으로 30fps 로 가정한다.
DEFAULT_CAMERA_FPS = 30.0

# gap_tol 은 pose 샘플 주기보다 최소 이 배수만큼 커야 안전하다.
# 딱 1.0배면 주기가 조금만 흔들려도(프레임 드롭/얼굴 미검출) 세션이 끊긴다.
GAP_TOL_SAFETY = 1.5

# 평활 윈도우 안에 최소 이 개수의 pose 샘플이 들어와야 facing 비율이 의미를 갖는다.
MIN_SMOOTH_SAMPLES = 2


def pose_sample_period(settings):
    """track 하나가 head pose 샘플을 얻는 주기(초)의 하한.

    보강(enrich)은 enrich_interval_sec 와 처리 프레임 간격 중 느린 쪽으로 돌고(DetectionWorker._enrich_due),
    그 안에서 다시 track 당 face_run_every_n 번에 1번만 얼굴/포즈를 본다(FaceEnricher._should_run).
    _should_run 이 보는 frames_seen 은 보강 프레임에서만 증가하므로 두 값이 그대로 곱해진다.
    실제 주기는 스레드 모드의 프레임 드롭 때문에 이 하한보다 커질 수 있다.
    """
    fps = settings.camera_fps if settings.camera_fps > 0 else DEFAULT_CAMERA_FPS
    frame_period = settings.process_every_n / fps
    enrich_period = max(settings.enrich_interval_sec, frame_period)
    return enrich_period * max(1, settings.face_run_every_n)


def _check_gap_tol(settings, period):
    # gap_tol 이 pose 샘플 주기보다 짧으면 연속 응시가 매 샘플마다 끊긴다.
    # 샘플 1개짜리 세션은 길이가 0초라(_append_session: end-start) grade_glance_sec 미만으로
    # 통째로 버려지고, Attention 이 0 으로 주저앉는다.
    if settings.gaze_gap_tol_sec > period:
        return None
    advised = round(period * GAP_TOL_SAFETY, 2)
    return (f"gaze.gap_tol_sec({settings.gaze_gap_tol_sec}s) <= pose 샘플 주기({period:.2f}s). "
            f"연속 응시가 샘플마다 끊겨 Attention 이 과소 집계된다. "
            f"gap_tol_sec 를 {advised}s 이상으로 올리거나 "
            f"runtime.enrich_interval_sec / face.run_every_n_frames 를 낮춰라.")


def _check_smooth_window(settings, period):
    # 평활 윈도우가 pose 샘플 주기에 비해 좁으면 윈도우 안 샘플이 0~1개다.
    # 그러면 facing 비율이 0.0/1.0 으로만 튀어 smooth_min_ratio 가 무의미해진다(HOT 순간값 깜빡임).
    need = period * MIN_SMOOTH_SAMPLES
    if settings.gaze_smooth_window_sec >= need:
        return None
    return (f"gaze.smooth_window_sec({settings.gaze_smooth_window_sec}s) 안에 pose 샘플이 "
            f"{MIN_SMOOTH_SAMPLES}개 미만이다(주기 {period:.2f}s). "
            f"평활이 사실상 꺼진 것과 같아 concurrent_gazers 가 깜빡인다. "
            f"smooth_window_sec 를 {round(need, 2)}s 이상으로 올려라.")


def _check_tracker_retention(settings):
    # BoT-SORT가 lost track을 재활성화해도 adapter가 counted 상태를 먼저 지우면 같은 ID를 다시 센다.
    # 따라서 adapter 보관 프레임은 내부 track_buffer보다 짧으면 안 된다.
    if settings.tracker_backend not in ("botsort", "bytetrack"):
        return None
    if settings.track_max_missing >= settings.track_buffer:
        return None
    return (f"tracker.max_missing({settings.track_max_missing}) < track_buffer({settings.track_buffer}). "
            f"재활성화된 track_id의 counted 상태가 먼저 사라져 중복 집계될 수 있다. "
            f"max_missing을 {settings.track_buffer} 이상으로 올려라.")


# 등록된 검사 목록. 새 불변식은 여기에 함수만 추가하면 된다.
_GAZE_CHECKS = (_check_gap_tol, _check_smooth_window)


def check_settings(settings):
    """경고 문자열 리스트를 반환한다(빈 리스트면 정상). 실행을 막지는 않는다."""
    warnings = [_check_tracker_retention(settings)]
    # face/gaze 가 꺼져 있으면 pose 샘플 자체가 없어 검증할 관계도 없다.
    if settings.gaze_enable and settings.face_enable:
        period = pose_sample_period(settings)
        warnings.extend(check(settings, period) for check in _GAZE_CHECKS)
    return [warning for warning in warnings if warning]


def report_settings(settings, label="", printer=print):
    """검증 경고를 출력하고 리스트로 돌려준다. 진입점(run)에서 한 번 호출한다."""
    warnings = check_settings(settings)
    prefix = f"{label} " if label else ""
    for warning in warnings:
        printer(f"{prefix}WARNING: {warning}")
    return warnings
