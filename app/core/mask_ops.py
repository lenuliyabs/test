from __future__ import annotations

import cv2
import numpy as np


def threshold_segmentation(
    image_rgb: np.ndarray,
    mode: str = "otsu",
    close_size: int = 3,
    open_size: int = 3,
    manual_threshold: int = 128,
) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    if mode == "manual":
        _, mask = cv2.threshold(gray, manual_threshold, 255, cv2.THRESH_BINARY)
    else:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    close_k = max(1, close_size)
    open_k = max(1, open_size)
    close_kernel = np.ones((close_k, close_k), np.uint8)
    open_kernel = np.ones((open_k, open_k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    return (mask > 0).astype(np.uint8)


def apply_brush(mask: np.ndarray, center: tuple[int, int], radius: int, value: int) -> np.ndarray:
    out = mask.copy()
    cv2.circle(out, center, max(1, radius), int(value), thickness=-1)
    return out


def mask_to_rgba(
    mask: np.ndarray, alpha: float = 0.4, color: tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    active = mask > 0
    overlay[active, :3] = color
    overlay[active, 3] = int(np.clip(alpha, 0.0, 1.0) * 255)
    return overlay
