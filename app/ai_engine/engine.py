from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import cv2
import numpy as np

from app.ai_engine.analytics import (
    detect_artifacts,
    fallback_segmentation,
    illumination_correction,
    qc_metrics,
    reinhard_normalize,
)


@dataclass
class PreviewResult:
    image: np.ndarray
    metrics: dict
    suggestions: dict


class AIEngine:
    def __init__(self) -> None:
        self._cache: dict[str, PreviewResult] = {}

    def _hash_key(self, image: np.ndarray, stage: str, params: dict, roi: tuple | None) -> str:
        h = hashlib.sha256()
        h.update(image.tobytes())
        h.update(stage.encode("utf-8"))
        h.update(json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(str(roi).encode("utf-8"))
        return h.hexdigest()

    def analyze_quality(self, image: np.ndarray) -> dict:
        m = qc_metrics(image)
        return {
            "metrics": {k: round(float(v), 4) for k, v in m.items()},
            "suggestions": {
                "normalization": m["tissue_ratio"] > 0.2,
                "illumination_correction": m["vignette"] > 4.0,
                "noise_reduction": int(min(60, max(0, m["noise"] * 3))),
                "sharpness_adjust": (
                    -20 if m["sharpness"] > 400 else (20 if m["sharpness"] < 30 else 0)
                ),
            },
        }

    def suggest_stain_preset(self, image: np.ndarray) -> dict:
        r, g, b = [float(image[..., i].mean()) for i in range(3)]
        if r > g + 12 and r > b + 12:
            return {"preset": "Van Gieson", "confidence": 0.71}
        if b > r + 8:
            return {"preset": "H&E", "confidence": 0.69}
        if r > b and g > b:
            return {"preset": "PAS", "confidence": 0.58}
        return {"preset": "Auto", "confidence": 0.55}

    def preview_stage(
        self, stage: str, image: np.ndarray, roi: tuple[int, int, int, int] | None, params: dict
    ) -> PreviewResult:
        key = self._hash_key(image, stage, params, roi)
        if key in self._cache:
            return self._cache[key]

        if roi is not None:
            x, y, w, h = roi
            roi_img = image[y : y + h, x : x + w]
        else:
            roi_img = image

        if stage == "quality_check":
            q = self.analyze_quality(roi_img)
            res = PreviewResult(roi_img, q["metrics"], q["suggestions"])
        elif stage == "stain_normalization":
            normalized = reinhard_normalize(roi_img)
            res = PreviewResult(
                normalized,
                {"mean": float(normalized.mean()), "std": float(normalized.std())},
                self.suggest_stain_preset(roi_img),
            )
        elif stage == "illumination":
            corrected = illumination_correction(roi_img)
            g = cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY)
            res = PreviewResult(
                corrected, {"illum_gradient": float(g.std())}, {"method": "background subtraction"}
            )
        elif stage == "artifacts":
            m = detect_artifacts(roi_img)
            over = roi_img.copy()
            over[m > 0] = (255, 128, 0)
            res = PreviewResult(
                over, {"artifact_ratio": float(np.mean(m > 0))}, {"mask": "эвристика"}
            )
        elif stage == "segmentation":
            seg = fallback_segmentation(roi_img)
            over = roi_img.copy()
            over[seg.mask > 0] = (over[seg.mask > 0] * 0.6 + np.array([255, 0, 0]) * 0.4).astype(
                np.uint8
            )
            res = PreviewResult(
                over,
                {
                    "coverage": seg.coverage,
                    "small_objects_ratio": seg.small_objects_ratio,
                    "merged_ratio": seg.merged_ratio,
                    "confidence": seg.confidence,
                },
                {"backend": "fallback_cv"},
            )
        else:
            res = PreviewResult(roi_img, {}, {})

        self._cache[key] = res
        return res

    def apply_stage(
        self, stage: str, image: np.ndarray, rois: list[tuple[int, int, int, int]], params: dict
    ) -> PreviewResult:
        if rois:
            merged = image.copy()
            metrics = {"roi_count": len(rois)}
            for roi in rois:
                p = self.preview_stage(stage, image, roi, params)
                x, y, w, h = roi
                merged[y : y + h, x : x + w] = p.image
            return PreviewResult(merged, metrics, {"applied": True})
        return self.preview_stage(stage, image, None, params)
