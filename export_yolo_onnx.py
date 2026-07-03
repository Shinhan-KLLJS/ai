"""YOLO .pt 모델을 Loovi 런타임이 쓰는 ONNX로 변환하는 빌드용 스크립트.

이 스크립트가 만든 .onnx를 config(models.person_onnx)가 소비한다. 런타임 코드 의존은 없다.

사용 예:
  python export_yolo_onnx.py models/yolo11l.pt
  python export_yolo_onnx.py models/yolov11l-face.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    # CLI 인자: 변환할 .pt 경로와 입력 크기/opset(호환성 조정용).
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="변환할 YOLO .pt 모델 경로")
    parser.add_argument("--imgsz", type=int, default=640, help="export 입력 이미지 크기")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset 버전")
    args = parser.parse_args()

    # 입력 검증: 존재하는 .pt 파일만 허용한다.
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if model_path.suffix.lower() != ".pt":
        raise ValueError("입력 모델은 .pt 파일이어야 합니다")

    # simplify=True로 그래프를 단순화해 onnxruntime 로딩/추론을 가볍게 한다.
    YOLO(str(model_path)).export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
    )

    print(f"변환 완료: {model_path.with_suffix('.onnx')}")


if __name__ == "__main__":
    main()
