from __future__ import annotations

import numpy as np

try:
    from cellpose import models

    CELLPPOSE_AVAILABLE = True
except Exception:
    models = None
    CELLPPOSE_AVAILABLE = False


def run_cellpose(
    image_rgb: np.ndarray, model_type: str = "cyto", diameter: float | None = None
) -> np.ndarray:
    if not CELLPPOSE_AVAILABLE:
        raise RuntimeError("Cellpose is not installed")
    model = models.Cellpose(model_type=model_type)
    masks, *_ = model.eval(image_rgb, diameter=diameter, channels=[0, 0])
    return (masks > 0).astype(np.uint8)
