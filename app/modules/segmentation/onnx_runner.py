from __future__ import annotations

import numpy as np

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except Exception:
    ort = None
    ONNX_AVAILABLE = False


def run_onnx_segmentation(image_rgb: np.ndarray, model_path: str) -> tuple[np.ndarray, float]:
    if not ONNX_AVAILABLE:
        raise RuntimeError("onnxruntime не установлен")
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    x = image_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]
    out = sess.run(None, {in_name: x})[0]
    if out.ndim == 4:
        out = out[0, 0]
    conf = float(np.mean(out))
    mask = (out > 0.5).astype(np.uint8)
    return mask, conf
