from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str
    paths: tuple[str, ...]


def models_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(base) / "HistoAnalyzer" / "models"
    return Path.home() / ".cache" / "HistoAnalyzer" / "models"


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("phikon_v2", "Phikon v2 (zones)", ("phikon_v2/config.json", "phikon_v2/model.safetensors")),
    ModelSpec(
        "cellpose_weights",
        "Cellpose weights",
        (
            "cellpose/nuclei",
            "cellpose/cyto3",
            "cellpose/cpsam",
        ),
    ),
    ModelSpec("stardist_weights", "StarDist weights", ("stardist/model.weights.h5",)),
    ModelSpec("hovernet_onnx", "HoVer-Net ONNX", ("hovernet/hovernet.onnx",)),
    ModelSpec("sam_checkpoint", "SAM checkpoint", ("sam/sam_vit_h_4b8939.pth",)),
)


def model_specs() -> tuple[ModelSpec, ...]:
    return MODEL_SPECS


def get_spec(key: str) -> ModelSpec:
    for spec in MODEL_SPECS:
        if spec.key == key:
            return spec
    raise KeyError(key)


def check_installed(key: str) -> tuple[bool, list[Path]]:
    spec = get_spec(key)
    root = models_dir()
    missing = [root / rel for rel in spec.paths if not (root / rel).exists()]
    return len(missing) == 0, missing


def model_main_path(key: str) -> Path:
    spec = get_spec(key)
    return models_dir() / spec.paths[0]
