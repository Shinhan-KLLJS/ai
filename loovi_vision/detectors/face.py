import numpy as np

from loovi_vision.runtime import onnx_providers


class FaceAnalyzer:
    # insightface FaceAnalysis 래퍼: 사람 crop에서 얼굴 검출 + (요청 시) 성별/연령.
    # 검출(detect)과 성별/연령(analyze)을 분리해 genderage는 best_face 1회만 돌린다.
    def __init__(self, settings):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "insightface가 설치되어 있지 않습니다. face.enable=true 를 쓰려면 설치하세요:\n"
                "    pip install -r requirements.txt\n"
                "    pip uninstall -y onnxruntime   # GPU 충돌 방지 (CPU 버전 제거)\n"
                "얼굴 분석 없이 기존 동작만 원하면 "
                "loovi_vision/configs/person_only.yaml 에서 face.enable: false 로 두세요."
            ) from exc

        self.settings = settings
        self.conf_min = settings.face_conf_min
        providers = onnx_providers(settings.enable_cuda)
        # detection + genderage만 로드해 recognition/landmark 로딩 비용을 줄인다.
        self.app = FaceAnalysis(
            name=settings.face_pack,
            allowed_modules=["detection", "genderage"],
            providers=providers,
        )
        ctx_id = 0 if settings.enable_cuda else -1
        self.app.prepare(ctx_id=ctx_id, det_thresh=self.conf_min, det_size=settings.face_det_size)
        self.det_model = self.app.det_model
        self.genderage = self.app.models.get("genderage")
        label = providers[0].replace("ExecutionProvider", "")
        print(f"  OK FaceAnalyzer [{settings.face_pack}] [{label}]")

    def detect(self, crop):
        # 사람 crop에서 얼굴 목록 반환: [{bbox:(x,y,w,h), conf, area, kps}]
        bboxes, kpss = self.det_model.detect(crop, max_num=0, metric="default")
        results = []
        for i in range(len(bboxes)):
            conf = float(bboxes[i, 4])
            if conf < self.conf_min:
                continue
            x1, y1, x2, y2 = bboxes[i, :4]
            w, h = float(x2 - x1), float(y2 - y1)
            if w <= 0 or h <= 0:
                continue
            results.append({
                "bbox": (int(x1), int(y1), int(w), int(h)),
                "conf": conf,
                "area": w * h,
                "kps": kpss[i] if kpss is not None else None,
            })
        return results

    def analyze(self, face_img, face_bbox, face_kps=None):
        # best_face 1장에 대해서만 genderage를 1회 판정한다 (성능). (gender, age) 반환.
        if self.genderage is None or face_img is None or face_bbox is None:
            return None, None
        from insightface.app.common import Face

        x, y, w, h = face_bbox
        face = Face(
            bbox=np.array([x, y, x + w, y + h], dtype=np.float32),
            kps=np.asarray(face_kps, dtype=np.float32) if face_kps is not None else None,
            det_score=1.0,
        )
        # Attribute.get은 face.bbox 중심으로 정렬한 뒤 face.gender(1=male)/face.age를 채운다.
        self.genderage.get(face_img, face)
        gender = int(face.gender) if face.gender is not None else None
        age = int(face.age) if face.age is not None else None
        return gender, age
