from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class EnhanceParams:
    brightness: int = 0
    contrast: int = 0
    highlights: int = 0
    shadows: int = 0
    saturation: int = 0
    warmth: int = 0
    sharpness: int = 0
    noise_reduction: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EnhanceParams":
        return cls(**{k: int(v) for k, v in data.items() if k in cls.__annotations__})


def _apply_highlights_shadows(img: np.ndarray, highlights: int, shadows: int) -> np.ndarray:
    f = img.astype(np.float32) / 255.0
    if highlights != 0:
        strength = np.clip(highlights / 100.0, -1.0, 1.0)
        mask = f > 0.5
        if strength > 0:
            f[mask] = f[mask] + (1.0 - f[mask]) * strength
        else:
            f[mask] = f[mask] * (1.0 + strength)
    if shadows != 0:
        strength = np.clip(shadows / 100.0, -1.0, 1.0)
        mask = f <= 0.5
        if strength > 0:
            f[mask] = f[mask] + (1.0 - f[mask]) * strength * 0.8
        else:
            f[mask] = f[mask] * (1.0 + strength)
    return np.clip(f * 255.0, 0, 255).astype(np.uint8)


def apply_enhancements(image_rgb: np.ndarray, params: EnhanceParams) -> np.ndarray:
    img = image_rgb.copy()
    if params.noise_reduction > 0:
        ksize = 3 + 2 * min(3, params.noise_reduction // 30)
        img = cv2.medianBlur(img, ksize)

    alpha = 1.0 + params.contrast / 100.0
    beta = params.brightness * 2.0
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    img = _apply_highlights_shadows(img, params.highlights, params.shadows)

    if params.saturation != 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + params.saturation / 100.0), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if params.warmth != 0:
        shift = np.clip(params.warmth, -100, 100)
        r_shift = max(0, shift) * 1.2
        b_shift = max(0, -shift) * 1.2
        img = img.astype(np.float32)
        img[..., 0] = np.clip(img[..., 0] + r_shift - b_shift * 0.4, 0, 255)
        img[..., 2] = np.clip(img[..., 2] - r_shift * 0.4 + b_shift, 0, 255)
        img = img.astype(np.uint8)

    if params.sharpness > 0:
        sigma = max(0.3, params.sharpness / 80.0)
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
        amount = params.sharpness / 50.0
        img = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)

    return img
