import json
import time

from loovi_vision.detectors.headpose import DMHeadPose
from loovi_vision.enrich.gaze import is_facing


class GazeEnricher:
    # 얼굴 crop -> head pose -> facing(비평활) 판정 -> COLD raw 기록 + 평활 상태 갱신.
    # FaceEnricher.on_face 로 연결되어 1차 검출 얼굴을 재사용한다.
    def __init__(self, settings, registry, poses_path):
        self.settings = settings
        self.registry = registry          # FaceEnricher와 공유 (gender/age + pose 한 곳에)
        self.pose_model = DMHeadPose(settings)
        self.poses_path = poses_path
        self._fh = poses_path.open("a", encoding="utf-8")
        self.facing_now = {}              # track_id -> 최근 raw facing (overlay용)
        self.last_pose = {}               # track_id -> (yaw, pitch, roll) 최근값 (overlay용)
        self.infer_ms = []               # head pose 추론 시간(ms) 누적 (성능 로그)

    def observe_face(self, track_id, face_crop, face_bbox_frame, person_bbox, frame_id, timestamp_sec):
        # FaceEnricher 가 찾은 얼굴 crop 마다 호출된다(정확도 우선, 샘플링 금지). 이 한 번의 호출 = pose 샘플 1개.
        # 흐름: 얼굴 crop → head pose 추정(yaw/pitch/roll) → facing 판정 → COLD raw 1줄 기록 + 상태(track) 갱신.
        t0 = time.time()
        pose = self.pose_model.estimate(face_crop)          # 머리방향 3축 추정
        self.infer_ms.append((time.time() - t0) * 1000.0)   # 추론 소요(ms) 성능 로그
        if pose is None:
            return  # 추론 실패 = 이 샘플은 pose 없음(건너뜀)
        fx, fy, fw, fh = face_bbox_frame
        face_px = float(max(fw, fh))
        # 얼굴이 너무 작으면(먼 사람) 각도 신뢰도가 낮음 → low_conf 표시(정책에 따라 응시 집계 제외).
        low_conf = face_px < self.settings.gaze_pose_min_face_px
        record = {
            "track_id": int(track_id),
            "frame_idx": int(frame_id),
            "timestamp_sec": round(float(timestamp_sec), 4),
            "yaw": round(pose["yaw"], 3),
            "pitch": round(pose["pitch"], 3),
            "roll": round(pose["roll"], 3),
            "facing": is_facing(pose, self.settings),   # 현재 임계값 기준 (참고용, 비평활)
            "face_bbox": [int(fx), int(fy), int(fw), int(fh)],
            "face_px_size": round(face_px, 2),
            "person_bbox": [int(v) for v in person_bbox],
            "low_conf": low_conf,
        }
        # HOT: track 상태에 이 샘플을 반영(누적 응시시간 facing_sec + 평활 링버퍼 갱신).
        state = self.registry.states.get(track_id)
        if state is not None:
            state.add_pose(record, self.settings.gaze_smooth_window_sec,
                           self.settings.gaze_low_conf_policy, self.settings.gaze_gap_tol_sec)
        self.facing_now[track_id] = record["facing"]                    # overlay 실시간 표시용
        self.last_pose[track_id] = (pose["yaw"], pose["pitch"], pose["roll"])  # overlay 3축 화살표용
        # COLD: raw 를 즉시 파일에 1줄 기록(+flush). 임계값을 바꿔 사후 재분석해도 재수집이 필요 없는 최종 안전망.
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def avg_infer_ms(self):
        return round(sum(self.infer_ms) / len(self.infer_ms), 3) if self.infer_ms else 0.0

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass
