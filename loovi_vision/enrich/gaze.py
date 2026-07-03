# 화면 향함(gaze proxy) 판정. 임계값은 전부 config 주입, 하드코딩 금지.


def facing_from_angles(yaw, pitch, yaw_center, pitch_center, yaw_tol, pitch_tol):
    # 단일 center(C안) 기준 허용 폭 안이면 화면을 향한 것으로 본다.
    return abs(yaw - yaw_center) <= yaw_tol and abs(pitch - pitch_center) <= pitch_tol


def is_facing(pose, settings):
    # 현재 임계값 기준 즉석 판정 (비평활, COLD 기록/참고용).
    if pose is None:
        return False
    return facing_from_angles(
        pose["yaw"], pose["pitch"],
        settings.gaze_yaw_center, settings.gaze_pitch_center,
        settings.gaze_yaw_tol, settings.gaze_pitch_tol,
    )


def is_facing_smoothed(state, settings, now_sec):
    # HOT 스냅샷용: 최근 smooth_window_sec 동안 facing 비율이 임계 이상이면 응시.
    # (한두 프레임 놓침에 동시 인원이 출렁이는 것을 방지)
    window = settings.gaze_smooth_window_sec
    samples = [facing for (ts, facing) in state.facing_smooth_state if ts >= now_sec - window]
    if not samples:
        return False
    ratio = sum(1 for facing in samples if facing) / len(samples)
    return ratio >= settings.gaze_smooth_min_ratio
