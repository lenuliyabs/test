from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SegmentationResult:
    mask: np.ndarray
    coverage: float
    small_objects_ratio: float
    merged_ratio: float
    confidence: float


def qc_metrics(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    overexposed = float(np.mean(gray >= 250))
    underexposed = float(np.mean(gray <= 5))
    noise = float(np.mean(np.abs(gray.astype(np.float32) - cv2.GaussianBlur(gray, (5, 5), 0))))
    vignette = illumination_gradient_score(gray)
    tissue_ratio = float(np.mean(gray < 230))
    return {
        "sharpness": sharpness,
        "overexposed": overexposed,
        "underexposed": underexposed,
        "noise": noise,
        "vignette": vignette,
        "tissue_ratio": tissue_ratio,
    }


def focus_heatmap(image: np.ndarray, block: int = 64) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h, block):
        for x in range(0, w, block):
            tile = gray[y : y + block, x : x + block]
            v = float(cv2.Laplacian(tile, cv2.CV_64F).var())
            out[y : y + tile.shape[0], x : x + tile.shape[1]] = v
    if out.max() > out.min():
        out = (out - out.min()) / (out.max() - out.min())
    return out


def illumination_gradient_score(gray: np.ndarray) -> float:
    low = cv2.GaussianBlur(gray, (0, 0), sigmaX=45)
    gy, gx = np.gradient(low.astype(np.float32))
    return float(np.mean(np.sqrt(gx**2 + gy**2)))


def reinhard_normalize(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    means = lab.reshape(-1, 3).mean(axis=0)
    stds = lab.reshape(-1, 3).std(axis=0) + 1e-6
    target_mean = np.array([170.0, 128.0, 128.0], dtype=np.float32)
    target_std = np.array([35.0, 12.0, 12.0], dtype=np.float32)
    norm = (lab - means) * (target_std / stds) + target_mean
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    return cv2.cvtColor(norm, cv2.COLOR_LAB2RGB)


def illumination_correction(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    bg = cv2.GaussianBlur(l_ch, (0, 0), sigmaX=45)
    corrected = cv2.normalize(
        l_ch.astype(np.float32) - bg.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX
    )
    out = cv2.merge([corrected.astype(np.uint8), a_ch, b_ch])
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


def detect_artifacts(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bright = gray > 245
    dark = gray < 8
    sat = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[..., 1] > 200
    mask = (bright | dark | sat).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def select_rois(
    image: np.ndarray,
    artifact_mask: np.ndarray | None,
    tile_size: int = 256,
    n_rois: int = 6,
) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    scored: list[tuple[float, tuple[int, int, int, int]]] = []
    for y in range(0, max(1, h - tile_size + 1), tile_size):
        for x in range(0, max(1, w - tile_size + 1), tile_size):
            tile = gray[y : y + tile_size, x : x + tile_size]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                continue
            edges = cv2.Canny(tile, 40, 120)
            edge_density = float(np.mean(edges > 0))
            hist = cv2.calcHist([tile], [0], None, [32], [0, 256]).flatten() + 1e-6
            p = hist / hist.sum()
            entropy = float(-(p * np.log2(p)).sum())
            tissue = float(np.mean(tile < 230))
            penalty = 0.0
            if artifact_mask is not None:
                a = artifact_mask[y : y + tile_size, x : x + tile_size]
                penalty = float(np.mean(a > 0))
            score = edge_density + 0.08 * entropy + tissue - 1.5 * penalty
            scored.append((score, (x, y, tile_size, tile_size)))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected: list[tuple[int, int, int, int]] = []
    for _, roi in scored:
        if len(selected) >= n_rois:
            break
        x, y, w1, h1 = roi
        overlap = False
        for sx, sy, sw, sh in selected:
            inter_x = max(0, min(x + w1, sx + sw) - max(x, sx))
            inter_y = max(0, min(y + h1, sy + sh) - max(y, sy))
            inter = inter_x * inter_y
            if inter > 0.2 * w1 * h1:
                overlap = True
                break
        if not overlap:
            selected.append(roi)
    return selected


def fallback_segmentation(image: np.ndarray) -> SegmentationResult:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return SegmentationResult(
            mask=(mask > 0).astype(np.uint8),
            coverage=0.0,
            small_objects_ratio=0.0,
            merged_ratio=0.0,
            confidence=0.1,
        )
    areas = stats[1:, cv2.CC_STAT_AREA]
    coverage = float(np.mean(mask > 0))
    small = float(np.mean(areas < 50))
    merged = float(np.mean(areas > np.percentile(areas, 90))) if len(areas) else 0.0
    confidence = float(np.clip(1.0 - small * 0.8 - abs(coverage - 0.35), 0.0, 1.0))
    return SegmentationResult(
        mask=(mask > 0).astype(np.uint8),
        coverage=coverage,
        small_objects_ratio=small,
        merged_ratio=merged,
        confidence=confidence,
    )


def collagen_index_vangieson_proxy(
    image: np.ndarray, tissue_mask: np.ndarray | None = None
) -> float:
    r = image[..., 0].astype(np.float32)
    g = image[..., 1].astype(np.float32)
    b = image[..., 2].astype(np.float32)
    idx = (r - 0.5 * g - 0.2 * b) / 255.0
    if tissue_mask is not None:
        vals = idx[tissue_mask > 0]
    else:
        vals = idx.reshape(-1)
    return float(np.clip(vals.mean(), -1.0, 1.0)) if vals.size else 0.0


def orientation_anisotropy(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 3)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 3)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 3)
    trace = jxx + jyy + 1e-6
    det = jxx * jyy - jxy * jxy
    tmp = np.sqrt(np.maximum(trace * trace / 4 - det, 0))
    l1 = trace / 2 + tmp
    l2 = trace / 2 - tmp
    anis = np.mean((l1 - l2) / (l1 + l2 + 1e-6))
    angles = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    hist, _ = np.histogram(angles, bins=24, range=(-np.pi / 2, np.pi / 2), density=True)
    hist = hist + 1e-9
    entropy = float(-(hist * np.log2(hist)).sum())
    return {"anisotropy": float(np.clip(anis, 0.0, 1.0)), "orientation_entropy": entropy}


def nearest_neighbor_metrics(points: np.ndarray) -> dict:
    if points.shape[0] < 2:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}
    d = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
    d[d == 0] = np.inf
    nn = np.min(d, axis=1)
    return {
        "mean": float(nn.mean()),
        "median": float(np.median(nn)),
        "p90": float(np.percentile(nn, 90)),
    }


def density_heatmap(mask: np.ndarray, block: int = 64) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h, block):
        for x in range(0, w, block):
            tile = mask[y : y + block, x : x + block]
            v = float(np.mean(tile > 0))
            out[y : y + tile.shape[0], x : x + tile.shape[1]] = v
    return out
