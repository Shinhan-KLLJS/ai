import math

import cv2


def draw_pose_axes(frame, yaw, pitch, roll, cx, cy, size):
    # 표준 head pose 3축 시각화(Hopenet/6DRepNet draw_axis 방식).
    # 빨강=X(오른쪽), 초록=Y(아래), 파랑=Z(코 방향, 화면 밖). 부호 규약은 캘리브레이션 의존.
    p = math.radians(pitch)
    y = -math.radians(yaw)
    r = math.radians(roll)
    x_axis = (size * (math.cos(y) * math.cos(r)) + cx,
              size * (math.cos(p) * math.sin(r) + math.cos(r) * math.sin(p) * math.sin(y)) + cy)
    y_axis = (size * (-math.cos(y) * math.sin(r)) + cx,
              size * (math.cos(p) * math.cos(r) - math.sin(p) * math.sin(y) * math.sin(r)) + cy)
    z_axis = (size * (math.sin(y)) + cx,
              size * (-math.cos(y) * math.sin(p)) + cy)
    origin = (int(cx), int(cy))
    cv2.line(frame, origin, (int(x_axis[0]), int(x_axis[1])), (0, 0, 255), 2)   # X 빨강
    cv2.line(frame, origin, (int(y_axis[0]), int(y_axis[1])), (0, 255, 0), 2)   # Y 초록
    cv2.line(frame, origin, (int(z_axis[0]), int(z_axis[1])), (255, 0, 0), 3)   # Z 파랑


def person_color(track_id, face_active, face_boxes, attended_ids):
    # face 비활성: 기존과 동일한 초록. 활성: 주목 상태별 색(1차 근사, head pose는 2차).
    if not face_active:
        return (80, 220, 120)
    if track_id in face_boxes:
        return (80, 220, 120)    # 초록 = 지금 얼굴 보임 (이쪽 향함 ≈ 응시 추정)
    if track_id in attended_ids:
        return (40, 200, 230)    # 노랑 = 주목했었음 (지금은 안 보임)
    return (150, 150, 150)       # 회색 = 통행만 (얼굴 한 번도 안 잡힘)


def draw(frame, detections, stats, settings, det_to_track, face_boxes=None, face_labels=None,
         attended_ids=None, gazing_ids=None, poses=None, gaze_secs=None):
    # bbox와 현재 batch 요약 정보를 영상/라이브 창에 그린다.
    # face_boxes/face_labels가 주어지면 얼굴 박스 + 실시간 성별/나이 라벨을 함께 그린다.
    # gazing_ids(평활 응시 중)면 얼굴 박스를 초록으로, poses면 3축, gaze_secs면 누적 응시시간 표시.
    face_active = face_boxes is not None
    face_boxes = face_boxes or {}
    face_labels = face_labels or {}
    attended_ids = attended_ids or set()
    gazing_ids = gazing_ids or set()
    poses = poses or {}
    gaze_secs = gaze_secs or {}
    frame_h = frame.shape[0]
    for idx, det in enumerate(detections):
        x, y, w, h = det["bbox"]
        conf = float(det.get("confidence", 0.0))
        track_id = det_to_track.get(idx)
        color = person_color(track_id, face_active, face_boxes, attended_ids)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"ID:{track_id} {conf:.2f}" if track_id is not None else f"person {conf:.2f}"
        cv2.putText(frame, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        face_box = face_boxes.get(track_id)
        if face_box is not None:
            fx1, fy1, fx2, fy2 = face_box
            # 평활 응시 중이면 초록, 아니면 파랑 얼굴 박스.
            fcolor = (80, 230, 80) if track_id in gazing_ids else (60, 170, 250)
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), fcolor, 2)
            # 성별/나이 + track별 누적 응시 시간을 얼굴 박스 위에 표시.
            parts = []
            if face_labels.get(track_id):
                parts.append(face_labels[track_id])
            if track_id in gaze_secs:
                parts.append(f"{gaze_secs[track_id]:.1f}s")
            if parts:
                cv2.putText(frame, " ".join(parts), (fx1, max(14, fy1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 170, 250), 2)
            pose = poses.get(track_id)
            if pose:
                # 얼굴 중심에서 3축(빨강X/초록Y/파랑Z) + 박스 아래 yaw/pitch/roll 숫자.
                yaw, pitch, roll = pose
                cx, cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
                draw_pose_axes(frame, yaw, pitch, roll, cx, cy, (fx2 - fx1) * 0.6)
                label = f"Y{yaw:+.0f} P{pitch:+.0f} R{roll:+.0f}"
                cv2.putText(frame, label, (fx1, min(frame_h - 4, fy2 + 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, fcolor, 1)

    lines = [
        "Loovi [Person Only]",
        f"Now     : {stats['now_count']:>3}",
        f"Peak    : {stats['peak_persons']:>3}",
        f"ActiveT : {stats['active_tracks']:>3}",
        f"Unique  : {stats['unique_total']:>3}",
    ]
    if "attended" in stats:
        # face.enable=true일 때만 주목 인원/성별 카운트를 HUD에 추가한다.
        lines.append(f"Attended: {stats['attended']:>3}")
        lines.append(f"M / F   : {stats['males']}/{stats['females']}")
    if "gazers" in stats:
        # gaze.enable=true일 때만 평활 응시 인원 + OTS/LTS를 HUD에 추가한다.
        lines.append(f"Gazers  : {stats['gazers']:>3}")
    if "lts" in stats:
        lines.append(f"OTS     : {stats['unique_total']:>3}")  # 통행(노출 기회)
        lines.append(f"LTS     : {stats['lts']:>3}")           # 응시자(누적 1초+)
    lines += [
        f"Tracker : {settings.tracker_backend}",
        f"Frames  : {stats['frame_count']:>3}",
        f"Model   : {settings.person_onnx.stem}",
    ]

    overlay = frame.copy()
    # face 비활성 시에는 기존과 픽셀 동일하게 158 고정 (회귀 방지).
    box_bottom = 30 + 20 * len(lines) if "attended" in stats else 158
    cv2.rectangle(overlay, (8, 8), (370, box_bottom), (10, 10, 22), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (18, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 255), 1)
    return frame
