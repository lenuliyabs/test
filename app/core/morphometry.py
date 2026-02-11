from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Measurement:
    kind: str
    value_px: float
    value_um: float | None
    details: str = ""


def polyline_length(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    p = np.asarray(points, dtype=np.float64)
    diffs = np.diff(p, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    p = np.asarray(points, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def to_um(value_px: float, um_per_px: float | None) -> float | None:
    if um_per_px is None:
        return None
    return float(value_px * um_per_px)


def _resample_polyline(points: np.ndarray, n: int = 200) -> np.ndarray:
    if len(points) < 2:
        return points
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.insert(np.cumsum(seg), 0, 0.0)
    total = cum[-1]
    if total == 0:
        return np.repeat(points[:1], n, axis=0)
    t = np.linspace(0, total, n)
    out = []
    for ti in t:
        idx = np.searchsorted(cum, ti, side="right") - 1
        idx = np.clip(idx, 0, len(seg) - 1)
        local = (ti - cum[idx]) / max(seg[idx], 1e-9)
        out.append(points[idx] + local * (points[idx + 1] - points[idx]))
    return np.asarray(out)


def thickness_distribution(
    line1: list[tuple[float, float]], line2: list[tuple[float, float]], samples: int = 200
) -> dict[str, float]:
    p1 = _resample_polyline(np.asarray(line1, dtype=np.float64), n=samples)
    p2 = _resample_polyline(np.asarray(line2, dtype=np.float64), n=samples)
    if len(p1) == 0 or len(p2) == 0:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}

    dists = np.sqrt(((p1[:, None, :] - p2[None, :, :]) ** 2).sum(axis=2))
    nn = dists.min(axis=1)
    return {
        "mean": float(np.mean(nn)),
        "median": float(np.median(nn)),
        "min": float(np.min(nn)),
        "max": float(np.max(nn)),
    }
