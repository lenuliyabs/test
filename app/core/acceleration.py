from __future__ import annotations

import platform
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class AccelerationInfo:
    vendor: str
    backend: str
    details: str


def detect_acceleration(preferred_vendor: str = "auto") -> AccelerationInfo:
    pref = preferred_vendor.lower()

    # NVIDIA via torch CUDA
    try:
        import torch

        if (pref in {"auto", "nvidia"}) and torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return AccelerationInfo("NVIDIA", "CUDA", f"GPU: {name}; torch={torch.__version__}")
    except Exception:
        pass

    # AMD/Intel via ONNX Runtime DirectML
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if (pref in {"auto", "amd", "intel"}) and "DmlExecutionProvider" in providers:
            return AccelerationInfo(
                "AMD/Intel",
                "DirectML",
                f"onnxruntime={ort.__version__}; providers={providers}",
            )
    except Exception:
        pass

    # Intel via OpenVINO
    try:
        import openvino as ov

        if pref in {"auto", "intel"}:
            return AccelerationInfo(
                "Intel", "OpenVINO", f"openvino={getattr(ov, '__version__', 'n/a')}"
            )
    except Exception:
        pass

    return AccelerationInfo("CPU", "CPU", f"platform={platform.platform()}")


def benchmark_acceleration(preferred_vendor: str = "auto", iterations: int = 80) -> dict:
    info = detect_acceleration(preferred_vendor)
    x = np.random.rand(512, 512).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(iterations):
        x = np.sqrt(x * 1.0001 + 0.0001)
    dt = time.perf_counter() - t0
    return {
        "vendor": info.vendor,
        "backend": info.backend,
        "details": info.details,
        "time_s": round(dt, 4),
        "iterations": iterations,
    }
