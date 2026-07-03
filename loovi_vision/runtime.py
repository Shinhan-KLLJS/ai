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


def provider_label(session):
    # 로그에는 "CUDAExecutionProvider" 대신 "CUDA"처럼 짧게 보여준다.
    return session.get_providers()[0].replace("ExecutionProvider", "")
