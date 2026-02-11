from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class Measurement:
    kind: str
    value_px: float
    value_um: float | None
    details: str = ""


@dataclass
class CalibrationProfile:
    name: str
    objective: str
    um_per_px: float
    date: str
    source_image_hash: str
    method: str
    n_repeats: int
    sd: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "objective": self.objective,
            "um_per_px": self.um_per_px,
            "date": self.date,
            "source_image_hash": self.source_image_hash,
            "method": self.method,
            "n_repeats": self.n_repeats,
            "sd": self.sd,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationProfile":
        return cls(
            name=str(data.get("name", "Профиль")),
            objective=str(data.get("objective", "custom")),
            um_per_px=float(data.get("um_per_px", 1.0)),
            date=str(data.get("date", datetime.now().isoformat())),
            source_image_hash=str(data.get("source_image_hash", "")),
            method=str(data.get("method", "manual")),
            n_repeats=int(data.get("n_repeats", 1)),
            sd=float(data.get("sd", 0.0)),
        )


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


def area_to_um2(value_px2: float, um_per_px: float | None) -> float | None:
    if um_per_px is None:
        return None
    return float(value_px2 * (um_per_px**2))


def convert_measurement_value(
    value_px: float, measure_type: str, um_per_px: float | None
) -> tuple[float | None, str]:
    if um_per_px is None:
        return None, "px²" if measure_type == "area" else "px"
    if measure_type == "area":
        return area_to_um2(value_px, um_per_px), "мкм²"
    return to_um(value_px, um_per_px), "мкм"


def calculate_um_per_px_from_divisions(
    px_distance: float, n_divisions: int, um_per_division: float
) -> float:
    real_um = n_divisions * um_per_division
    return float(real_um / max(px_distance, 1e-9))


def calculate_calibration_stats(
    px_distances: list[float], n_divisions: int, um_per_division: float
) -> dict[str, float]:
    values = [
        calculate_um_per_px_from_divisions(d, n_divisions, um_per_division) for d in px_distances
    ]
    arr = np.asarray(values, dtype=np.float64)
    return {
        "um_per_px": float(arr.mean()) if len(arr) else 0.0,
        "sd": float(arr.std(ddof=0)) if len(arr) else 0.0,
        "n_repeats": int(len(arr)),
    }


def recalc_measurements_with_scale(measurements: list[dict], um_per_px: float | None) -> list[dict]:
    out: list[dict] = []
    for m in measurements:
        value_px = float(m.get("value_px", 0.0))
        mtype = str(m.get("type", "line"))
        converted, units = convert_measurement_value(value_px, mtype, um_per_px)
        item = dict(m)
        item["value_um"] = None if converted is None else round(float(converted), 6)
        item["units"] = units
        out.append(item)
    return out


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
