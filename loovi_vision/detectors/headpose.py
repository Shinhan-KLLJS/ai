from pathlib import Path

import cv2
import numpy as np

from loovi_vision.runtime import onnx_providers, provider_label


class DMHeadPose:
    # DMHead ONNX 래퍼: 얼굴 crop -> {yaw, pitch, roll} (degree).
    # 공식 demo(PINTO0309/DMHead) 전처리/출력 규약을 그대로 따른다:
    #   입력: BGR->RGB, 224x224, NCHW float32 (정규화는 모델 내부에 fused).
    #   출력: [yaw, roll, pitch] (degree). 부호 규약은 캘리브레이션으로 실측 확인.
    INPUT_SIZE = 224

    def __init__(self, settings):
        import onnxruntime as ort

        self.model_path = Path(settings.headpose_onnx)
        if not self.model_path.exists():
            raise RuntimeError(
                f"head pose 모델이 없습니다: {self.model_path}\n"
                "DMHead ONNX(예: dmhead_Nx3x224x224.onnx)를 PINTO0309/DMHead 릴리스에서 받아\n"
                "models/dmhead.onnx 로 두거나, gaze.enable: false 로 두세요."
            )
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=onnx_providers(settings.enable_cuda),
        )
        self.input_name = self.session.get_inputs()[0].name
        print(f"  OK DMHeadPose [{self.model_path.stem}] [{provider_label(self.session)}]")

    def estimate(self, face_crop):
        # 얼굴 crop 1장 -> {yaw,pitch,roll}. 추론 실패 시 None (해당 프레임 pose 없음).
        if face_crop is None or face_crop.size == 0:
            return None
        try:
            resized = cv2.resize(face_crop, (self.INPUT_SIZE, self.INPUT_SIZE))
            rgb = resized[..., ::-1]                       # BGR -> RGB (공식 규약)
            nchw = np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis], dtype=np.float32)
            out = np.squeeze(self.session.run(None, {self.input_name: nchw})[0])
            yaw, roll, pitch = float(out[0]), float(out[1]), float(out[2])  # 출력 순서 주의
            return {"yaw": yaw, "pitch": pitch, "roll": roll}
        except Exception:
            return None
