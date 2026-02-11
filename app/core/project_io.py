from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.image_ops import EnhanceParams


@dataclass
class ProjectData:
    image_path: str
    enhance: dict[str, Any] = field(default_factory=dict)
    annotations: list[dict[str, Any]] = field(default_factory=list)
    measurements: list[dict[str, Any]] = field(default_factory=list)
    um_per_px: float | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


def save_project(
    project_file: str,
    image_path: str,
    enhance: EnhanceParams,
    mask: np.ndarray,
    annotations: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    um_per_px: float | None,
) -> None:
    project_path = Path(project_file)
    project_dir = project_path.with_suffix("")
    project_dir.mkdir(parents=True, exist_ok=True)

    mask_path = project_dir / "mask.png"
    cv2.imwrite(str(mask_path), (mask > 0).astype(np.uint8) * 255)

    payload = asdict(
        ProjectData(
            image_path=str(Path(image_path).resolve()),
            enhance=enhance.to_dict(),
            annotations=annotations,
            measurements=measurements,
            um_per_px=um_per_px,
        )
    )
    payload["mask_path"] = str(mask_path.name)
    payload["project_dir"] = str(project_dir)

    with open(project_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_project(project_file: str) -> tuple[dict[str, Any], np.ndarray]:
    project_path = Path(project_file)
    with open(project_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    project_dir = Path(payload.get("project_dir") or project_path.with_suffix(""))
    mask_path = project_dir / payload.get("mask_path", "mask.png")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        mask = np.zeros((1, 1), dtype=np.uint8)
    return payload, (mask > 0).astype(np.uint8)
