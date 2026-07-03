from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class TrackState:
    # 사람 한 명(track_id)당 통행/주목 기록을 누적하는 기록장.
    track_id: int
    frames_seen: int = 0                       # 보인 프레임 수 (모든 처리 프레임)
    frames_face_visible: int = 0               # 얼굴이 보인 프레임 수
    best_face: Optional[np.ndarray] = None     # 가장 잘 잡힌 face crop (2차 head pose 재사용)
    best_face_bbox: Optional[Tuple] = None     # best_face 내부 얼굴 bbox (x, y, w, h)
    best_face_kps: Optional[np.ndarray] = None # best_face 내부 5점 landmark
    best_face_score: float = 0.0               # best 선정 기준 = conf x 얼굴 면적
    gender: Optional[int] = None               # 1=male, 0=female, None=미상 (누적 다수결)
    age: Optional[int] = None                  # 정수 나이, None=미상 (누적 중앙값)
    gender_votes: List[int] = field(default_factory=lambda: [0, 0])  # [female, male] 누적표
    age_samples: List[int] = field(default_factory=list)             # 나이 샘플 누적
    # gaze(2차): pose 원시 기록(COLD 미러)과 평활용 링버퍼, 요약 카운터.
    pose_timeline: List[dict] = field(default_factory=list)          # 매 검출 프레임 raw (날것)
    facing_smooth_state: List[tuple] = field(default_factory=list)   # [(timestamp_sec, facing)] 최근 window
    frames_facing: int = 0                                            # facing 프레임 수 (참고용)
    facing_sec: float = 0.0                                           # 누적 응시 시간(초) — LTS 기준
    last_pose_ts: Optional[float] = None                             # 직전 pose timestamp (delta 계산용)
    first_seen: Optional[int] = None           # 최초 관측 frame_id
    last_seen: Optional[int] = None            # 최종 관측 frame_id

    @property
    def attended(self) -> bool:
        # 얼굴이 한 번이라도 잡혔으면 "주목"으로 분류한다.
        return self.frames_face_visible > 0

    def add_pose(self, record, smooth_window_sec, low_conf_policy="exclude", max_gap_sec=1.0):
        # 매 pose 샘플(= 머리방향을 한 번 측정한 것)마다 호출된다. 두 가지를 갱신한다:
        #   COLD: 원시 기록(record)을 pose_timeline 에 그대로 쌓는다(사후 분석용, 평활 없음).
        #   HOT:  누적 응시시간(facing_sec)과 평활용 링버퍼를 갱신한다(실시간 판정용).
        self.pose_timeline.append(record)
        facing = bool(record.get("facing"))            # 이 샘플에서 광고 쪽을 향했나
        # 얼굴이 너무 작아 신뢰가 낮은 샘플은(정책이 exclude면) 응시로 치지 않는다.
        if record.get("low_conf") and low_conf_policy == "exclude":
            facing = False
        ts = float(record.get("timestamp_sec", 0.0))   # 이 샘플의 실제 시각(초)
        if facing:
            self.frames_facing += 1
            # ★ 시청시간은 "프레임 수"가 아니라 "샘플 사이의 실제 시간(초)"으로 잰다.
            #    스톱워치처럼 직전 응시 샘플과 이번 샘플의 시간차(delta)를 더한다.
            #    delta 가 max_gap_sec 를 넘으면(한동안 안 봄/검출 끊김) 그 사이는 안 본 것으로 보고 더하지 않는다.
            #    → 프레임레이트가 흔들려도, 추론을 건너뛰어도 실제 시계 기준이라 정확하다.
            if self.last_pose_ts is not None:
                delta = ts - self.last_pose_ts
                if 0.0 < delta <= max_gap_sec:
                    self.facing_sec += delta
        self.last_pose_ts = ts                          # 다음 샘플의 delta 계산 기준
        # 평활 링버퍼: 최근 smooth_window_sec 동안의 (시각, facing) 만 남겨 실시간 응시 판정에 쓴다
        # (한두 샘플 놓침에 실시간 카운트가 출렁이는 것을 막는다).
        self.facing_smooth_state.append((ts, facing))
        cutoff = ts - smooth_window_sec
        while self.facing_smooth_state and self.facing_smooth_state[0][0] < cutoff:
            self.facing_smooth_state.pop(0)

    def add_genderage(self, gender, age):
        # 성별/나이는 프레임마다 조금씩 흔들리므로 단발값을 쓰지 않고 누적해 수렴시킨다.
        #   성별 = 지금까지 표의 다수결(male/female 표를 세어 많은 쪽).
        #   나이 = 지금까지 샘플들의 중앙값(median, 튀는 값에 강함).
        if gender in (0, 1):
            self.gender_votes[gender] += 1     # gender_votes = [female표, male표]
            self.gender = 0 if self.gender_votes[0] > self.gender_votes[1] else 1
        if age is not None:
            self.age_samples.append(int(age))
            ordered = sorted(self.age_samples)
            mid = len(ordered) // 2
            if len(ordered) % 2:               # 홀수 개 → 가운데 값
                self.age = ordered[mid]
            else:                              # 짝수 개 → 가운데 두 값의 평균
                self.age = (ordered[mid - 1] + ordered[mid]) // 2


class TrackStateRegistry:
    # 여러 track의 기록장을 한꺼번에 관리하는 레지스트리.
    def __init__(self):
        self.states = {}

    def observe(self, track_id, frame_id):
        # 모든 사람은 통행으로 무조건 집계한다 (분모): frames_seen 누적.
        state = self.states.get(track_id)
        if state is None:
            state = TrackState(track_id=track_id, first_seen=frame_id)
            self.states[track_id] = state
        state.frames_seen += 1
        state.last_seen = frame_id
        return state

    def record_face(self, track_id, frame_id, person_crop, face):
        # 얼굴이 잡힌 프레임에서 frames_face_visible 누적 및 best_face 갱신 검토.
        # best_face가 갱신되면 True를 반환한다 (호출부에서 genderage 1회 재판정용).
        state = self.states.get(track_id) or self.observe(track_id, frame_id)
        state.frames_face_visible += 1
        score = face["conf"] * face["area"]
        if score > state.best_face_score:
            crop, bbox, kps = extract_face_crop(person_crop, face)
            state.best_face = crop
            state.best_face_bbox = bbox
            state.best_face_kps = kps
            state.best_face_score = score
            return True
        return False

    def all(self):
        return list(self.states.values())


def extract_face_crop(person_crop, face, margin=0.4):
    # person crop에서 margin 포함 face crop과 그 안의 bbox/kps를 잘라낸다.
    # 2차 head pose 입력으로 재사용하려고 약간의 여백을 둔 원본 crop을 보관한다.
    x, y, w, h = face["bbox"]
    mx, my = int(w * margin), int(h * margin)
    height, width = person_crop.shape[:2]
    nx1, ny1 = max(0, x - mx), max(0, y - my)
    nx2, ny2 = min(width, x + w + mx), min(height, y + h + my)
    crop = person_crop[ny1:ny2, nx1:nx2].copy()
    bbox = (x - nx1, y - ny1, w, h)
    kps = None
    if face.get("kps") is not None:
        kps = np.asarray(face["kps"], dtype=np.float32).copy()
        kps[:, 0] -= nx1
        kps[:, 1] -= ny1
    return crop, bbox, kps
