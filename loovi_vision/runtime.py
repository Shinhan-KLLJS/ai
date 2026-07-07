import os
from pathlib import Path


_DLLS_READY = False


def onnx_providers(enable_cuda=True):
    # ONNX Runtime provider 우선순위를 구성한다. CUDA 실패 시 CPU fallback을 허용한다.
    global _DLLS_READY
    if not enable_cuda:
        return ["CPUExecutionProvider"]

    if not _DLLS_READY:
        try:
            # pip 기반 NVIDIA 패키지의 DLL 경로를 Windows DLL search path에 추가한다.
            import nvidia

            nvidia_dir = Path(nvidia.__file__).parent
            for bin_dir in nvidia_dir.glob("**/bin"):
                if bin_dir.is_dir():
                    os.add_dll_directory(str(bin_dir))
                    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
        except Exception as exc:
            print(f"  WARNING: NVIDIA DLL path setup failed: {exc}")

        try:
            # onnxruntime-gpu가 제공하는 DLL preload helper가 있으면 먼저 호출한다.
            import onnxruntime as ort

            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
        except Exception as exc:
            print(f"  WARNING: CUDA DLL preload failed: {exc}")

        _DLLS_READY = True

    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def cuda_provider_options():
    # CUDA EP 튜닝: 기본값(EXHAUSTIVE) conv 알고리즘 탐색은 shape마다 느리고 지연이 크게 요동친다
    # (첫 추론이 수백 ms~1초까지 튐). HEURISTIC 은 탐색 없이 좋은 알고리즘을 골라
    # warmup이 빠르고 지연이 안정적이다. 정확도에는 영향 없음.
    return {"cudnn_conv_algo_search": "HEURISTIC"}


def provider_options(providers):
    # providers 리스트와 1:1 매칭되는 옵션 dict 리스트(InferenceSession provider_options 인자용).
    return [cuda_provider_options() if p == "CUDAExecutionProvider" else {} for p in providers]


def provider_label(session):
    # 로그에는 "CUDAExecutionProvider" 대신 "CUDA"처럼 짧게 보여준다.
    return session.get_providers()[0].replace("ExecutionProvider", "")
