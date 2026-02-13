from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.ai.models.downloader import DownloadPackWorker, download_model_pack, remove_model
from app.ai.models.registry import check_installed, model_main_path, models_dir

MODEL_CATALOG = {
    "Phikon": {"key": "phikon_v2", "version": "modelpack"},
    "Phikon-v2": {"key": "phikon_v2", "version": "modelpack"},
    "Cellpose nuclei": {"key": "cellpose_weights", "version": "modelpack"},
    "Cellpose cyto": {"key": "cellpose_weights", "version": "modelpack"},
    "Cellpose cpsam": {"key": "cellpose_weights", "version": "modelpack"},
    "StarDist": {"key": "stardist_weights", "version": "modelpack"},
    "HoVer-Net": {"key": "hovernet_onnx", "version": "modelpack"},
    "SAM": {"key": "sam_checkpoint", "version": "modelpack"},
}


@dataclass
class ModelStatus:
    name: str
    installed: bool
    version: str
    path: str
    size_mb: float
    sha256: str


def model_root() -> Path:
    return models_dir()


def get_model_statuses() -> list[ModelStatus]:
    out: list[ModelStatus] = []
    for name, meta in MODEL_CATALOG.items():
        key = meta["key"]
        ok, _ = check_installed(key)
        p = model_main_path(key)
        size = round(p.stat().st_size / (1024 * 1024), 2) if p.exists() and p.is_file() else 0.0
        out.append(
            ModelStatus(
                name=name,
                installed=ok,
                version=meta["version"],
                path=str(p),
                size_mb=size,
                sha256="",
            )
        )
    return out


def delete_model(name: str) -> tuple[bool, str]:
    meta = MODEL_CATALOG.get(name)
    if not meta:
        return False, f"Неизвестная модель: {name}"
    return remove_model(meta["key"])


def ensure_default_models() -> list[str]:
    ok, msg = download_model_pack()
    return [msg if ok else f"Ошибка: {msg}"]


DownloadWorker = DownloadPackWorker
