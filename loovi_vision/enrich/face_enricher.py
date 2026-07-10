from loovi_vision.detectors.face import FaceAnalyzer
from loovi_vision.enrich.track_state import TrackStateRegistry
from loovi_vision.enrich.face_quality import quality_weight


class FaceEnricher:
    # 사람 track별 얼굴 검출 -> 기록장 갱신 -> 세션 종료 시 성별/연령 1회 판정.
    def __init__(self, settings):
        self.settings = settings
        self.analyzer = FaceAnalyzer(settings)
        self.registry = TrackStateRegistry()
        self.last_face_boxes = {}  # track_id -> (x1,y1,x2,y2) 프레임 좌표 (overlay용)
        self.face_labels = {}      # track_id -> "M/27" (overlay 실시간 라벨)
        # 후속 처리 hook(2차 gaze 등). None이면 1차와 100% 동일.
        # 시그니처: on_face(track_id, face_crop, face_bbox_frame, person_bbox, frame_id, timestamp_sec)
        self.on_face = None

    def _should_run(self, state):
        # track당 run_every_n_frames 간격으로만 얼굴 검출을 호출한다 (속도).
        n = max(1, self.settings.face_run_every_n)
        return (state.frames_seen - 1) % n == 0

    def _person_crop(self, frame, bbox):
        # 너무 작은 사람 crop은 얼굴 검출을 건너뛴다.
        x, y, w, h = bbox
        if w < self.settings.face_min_crop_size or h < self.settings.face_min_crop_size:
            return None, 0, 0
        height, width = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 <= x1 or y2 <= y1:
            return None, 0, 0
        return frame[y1:y2, x1:x2], x1, y1

    def _face_allowed(self, detections):
        # face.max_per_frame>0 이면 이번 프레임에 얼굴 분석을 돌릴 det 인덱스를 crop 면적 상위 K개로 제한.
        # 사람이 많을 때 얼굴 모델 호출 폭주를 막는다(가까운=큰 사람 우선). 0이면 무제한.
        cap = self.settings.face_max_per_frame
        if cap <= 0 or len(detections) <= cap:
            return None
        area = lambda d: d["bbox"][2] * d["bbox"][3]
        order = sorted(range(len(detections)), key=lambda i: area(detections[i]), reverse=True)
        return set(order[:cap])

    def process(self, frame, detections, det_to_track, frame_id, timestamp_sec=0.0):
        # 한 프레임 처리: (이번 프레임에 보인 track, 얼굴이 보인 track) 집합 반환.
        seen, faced = set(), set()
        allowed = self._face_allowed(detections)   # None이면 전원, 아니면 상위 K det 인덱스
        for det_idx, det in enumerate(detections):
            track_id = det_to_track.get(det_idx)
            if track_id is None:
                continue
            seen.add(track_id)
            state = self.registry.observe(track_id, frame_id, timestamp_sec)  # frames_seen += 1 (항상)
            if not self._should_run(state):
                continue
            if allowed is not None and det_idx not in allowed:
                continue  # 상한 초과: 이 프레임에선 이 사람 얼굴 분석 건너뜀(seen 집계엔 이미 포함)
            crop, off_x, off_y = self._person_crop(frame, det["bbox"])
            if crop is None:
                continue
            faces = self.analyzer.detect(crop)
            if not faces:
                self.last_face_boxes.pop(track_id, None)
                continue
            best = max(faces, key=lambda f: f["conf"] * f["area"])
            # best_face(head pose 재사용 crop)는 더 좋은 얼굴일 때만 보관.
            self.registry.record_face(track_id, frame_id, crop, best)
            faced.add(track_id)
            state = self.registry.states[track_id]
            fx, fy, fw, fh = best["bbox"]
            face_region = crop[fy:fy + fh, fx:fx + fw]   # 품질 계산·pose 재사용 공용 crop
            # genderage는 검출된 얼굴로 매번 판정하되, 품질 가중치를 줘 누적한다(저품질 표 억제).
            gender, age = self.analyzer.analyze(crop, best["bbox"], best["kps"])
            weight = quality_weight(best["conf"], best["area"], face_region,
                                    self.settings.face_quality_sharp_ref)
            state.add_genderage(gender, age, weight)
            self.face_labels[track_id] = format_label(state.gender, state.age)
            self.last_face_boxes[track_id] = (fx + off_x, fy + off_y, fx + fw + off_x, fy + fh + off_y)
            if self.on_face is not None:
                # 1차에서 검출한 얼굴 crop을 2차 head pose에 그대로 재사용.
                face_bbox_frame = (fx + off_x, fy + off_y, fw, fh)
                self.on_face(track_id, face_region, face_bbox_frame, det["bbox"], frame_id, timestamp_sec)
        return seen, faced

    def attended_ids(self):
        # 얼굴이 한 번이라도 잡힌 track 집합 (overlay 박스 색 결정용).
        # (메서드명은 overlay 계층 호환을 위해 유지 — 의미는 face_visible 집합이다.)
        return {s.track_id for s in self.registry.all() if s.face_visible}

    def live_counts(self):
        # overlay HUD용 실시간 누적: 얼굴 인식 인원, 남/녀 카운트.
        face_visible = males = females = 0
        for state in self.registry.all():
            if not state.face_visible:
                continue
            face_visible += 1
            if state.gender == 1:
                males += 1
            elif state.gender == 0:
                females += 1
        return {"attended": face_visible, "males": males, "females": females}

    def finalize(self):
        # 안전망: best_face는 있는데 누적 판정이 한 번도 안 된 track만 채운다.
        # (실시간 경로에서 검출마다 누적되므로 대부분 no-op)
        for state in self.registry.all():
            if state.best_face is not None and state.gender is None:
                gender, age = self.analyzer.analyze(
                    state.best_face, state.best_face_bbox, state.best_face_kps
                )
                state.add_genderage(gender, age, 1.0)   # 단발 폴백 → 가중치 중립(1.0)
        return self.registry


def format_label(gender, age):
    # 얼굴 박스 위에 띄울 짧은 라벨. 미상이면 None.
    if gender is None and age is None:
        return None
    sex = "M" if gender == 1 else "F" if gender == 0 else "?"
    return f"{sex}/{age}" if age is not None else sex
