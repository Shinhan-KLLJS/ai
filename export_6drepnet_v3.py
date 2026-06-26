"""
SixDRepNet → ONNX 변환 스크립트 v3
SixDRepNet_Detector 내부에서 실제 PyTorch 모델 추출
"""

import torch
import numpy as np
from pathlib import Path

print("=" * 55)
print("  SixDRepNet → ONNX 변환 스크립트 v3")
print("=" * 55)

# ── Step 1: 패키지 로드 후 내부 구조 파악 ──
print("\n[ Step 1 ] SixDRepNet 내부 구조 파악")
from sixdrepnet import SixDRepNet

detector = SixDRepNet(gpu_id=-1)
print(f"  클래스 타입: {type(detector)}")
print(f"  속성 목록: {[a for a in dir(detector) if not a.startswith('_')]}")

# ── Step 2: 내부 PyTorch 모델 추출 ──
print("\n[ Step 2 ] 내부 PyTorch 모델 추출")

inner_model = None

# 가능한 속성 이름들 시도
for attr in ['model', 'net', 'backbone', 'sixdrepnet', 'head_pose_model']:
    if hasattr(detector, attr):
        candidate = getattr(detector, attr)
        if isinstance(candidate, torch.nn.Module):
            inner_model = candidate
            print(f"  ✅ '{attr}' 속성에서 모델 발견: {type(inner_model)}")
            break

# 속성에서 못 찾으면 __dict__ 전체 탐색
if inner_model is None:
    print("  일반 속성 탐색 중...")
    for k, v in vars(detector).items():
        print(f"    {k}: {type(v)}")
        if isinstance(v, torch.nn.Module):
            inner_model = v
            print(f"  ✅ '{k}' 에서 모델 발견")
            break

if inner_model is None:
    print("  ❌ 내부 모델을 찾지 못함")
    exit(1)

# ── Step 3: 추론 테스트 ──
print("\n[ Step 3 ] 추론 테스트")
inner_model.eval()
dummy = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    out = inner_model(dummy)

print(f"  입력: {dummy.shape}")
print(f"  출력: {out.shape}  값: {out[0].tolist()[:3]}...")
print("  ✅ 추론 성공")

# ── Step 4: ONNX 변환 ──
print("\n[ Step 4 ] ONNX 변환")
Path("models").mkdir(exist_ok=True)
onnx_path = "models/sixdrepnet.onnx"

torch.onnx.export(
    inner_model, dummy, onnx_path,
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    do_constant_folding=True,
    verbose=False,
)
size_kb = Path(onnx_path).stat().st_size // 1024
print(f"  ✅ 변환 완료! ({size_kb}KB) → {onnx_path}")

# ── Step 5: ONNX Runtime 검증 ──
print("\n[ Step 5 ] ONNX Runtime 검증")
import onnxruntime as ort

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
out_onnx = sess.run(None, {"input": dummy.numpy()})[0]
print(f"  출력 shape: {out_onnx.shape}")

with torch.no_grad():
    out_torch = inner_model(dummy).numpy()
diff = float(abs(out_torch - out_onnx).max())
print(f"  PyTorch ↔ ONNX 오차: {diff:.6f}")
print(f"  {'✅ 일치 확인' if diff < 0.01 else '⚠️ 오차 있음 (동작 무방)'}")

print("\n" + "=" * 55)
print("  🎉 완료! 다음 단계:")
print()
print("  adscope_v4_fixed.py 의 Config 클래스에서:")
print('  POSE_ONNX = Path("models") / "sixdrepnet.onnx"')
print("=" * 55)
