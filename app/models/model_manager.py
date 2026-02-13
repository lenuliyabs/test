from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Signal

MODEL_CATALOG = {
    "Phikon": {
        "url": "https://huggingface.co/owkin/phikon/resolve/main/config.json",
        "file": "phikon_config.json",
        "version": "latest",
    },
    "Phikon-v2": {
        "url": "https://huggingface.co/owkin/phikon-v2/resolve/main/config.json",
        "file": "phikon_v2_config.json",
        "version": "latest",
    },
    "Cellpose nuclei": {
        "url": "https://www.cellpose.org/static/models/nuclei",
        "file": "cellpose_nuclei.mdl",
        "version": "latest",
    },
    "Cellpose cyto": {
        "url": "https://www.cellpose.org/static/models/cyto3",
        "file": "cellpose_cyto3.mdl",
        "version": "latest",
    },
    "Cellpose cpsam": {
        "url": "https://www.cellpose.org/static/models/cpsam",
        "file": "cellpose_cpsam.mdl",
        "version": "latest",
    },
}


def model_root() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(base) / "HistoAnalyzer" / "models"
    return Path.home() / ".cache" / "HistoAnalyzer" / "models"


@dataclass
class ModelStatus:
    name: str
    installed: bool
    version: str
    path: str
    size_mb: float
    sha256: str


class DownloadSignals(QObject):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal(bool, str)


class DownloadWorker(QRunnable):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.cancel_requested = False
        self.signals = DownloadSignals()

    def cancel(self) -> None:
        self.cancel_requested = True

    def run(self) -> None:
        try:
            ok, msg = download_model(
                self.name,
                self.signals.progress.emit,
                self.signals.log.emit,
                lambda: self.cancel_requested,
            )
            self.signals.finished.emit(ok, msg)
        except Exception as exc:  # pragma: no cover
            self.signals.finished.emit(False, str(exc))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def get_model_statuses() -> list[ModelStatus]:
    root = model_root()
    root.mkdir(parents=True, exist_ok=True)
    out: list[ModelStatus] = []
    for name, meta in MODEL_CATALOG.items():
        p = root / meta["file"]
        if p.exists():
            out.append(
                ModelStatus(
                    name=name,
                    installed=True,
                    version=meta["version"],
                    path=str(p),
                    size_mb=round(p.stat().st_size / (1024 * 1024), 2),
                    sha256=_sha256(p),
                )
            )
        else:
            out.append(
                ModelStatus(
                    name=name,
                    installed=False,
                    version=meta["version"],
                    path=str(p),
                    size_mb=0.0,
                    sha256="",
                )
            )
    return out


def download_model(name: str, progress_cb=None, log_cb=None, cancel_cb=None) -> tuple[bool, str]:
    if name not in MODEL_CATALOG:
        return False, f"Неизвестная модель: {name}"
    root = model_root()
    root.mkdir(parents=True, exist_ok=True)
    meta = MODEL_CATALOG[name]
    url = meta["url"]
    dst = root / meta["file"]
    tmp = dst.with_suffix(dst.suffix + ".part")

    if log_cb:
        log_cb(f"Скачивание {name}: {url}")

    with urllib.request.urlopen(url, timeout=30) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", "0") or 0)
        read = 0
        while True:
            if cancel_cb and cancel_cb():
                if log_cb:
                    log_cb("Загрузка отменена")
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
                return False, "Отменено"
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)
            read += len(chunk)
            if progress_cb and total > 0:
                progress_cb(int(read * 100 / total))

    tmp.replace(dst)
    if progress_cb:
        progress_cb(100)
    if log_cb:
        log_cb(f"Готово: {dst}")
    return True, f"Модель {name} скачана"


def delete_model(name: str) -> tuple[bool, str]:
    if name not in MODEL_CATALOG:
        return False, f"Неизвестная модель: {name}"
    p = model_root() / MODEL_CATALOG[name]["file"]
    if not p.exists():
        return False, "Файл не найден"
    p.unlink()
    return True, f"Удалена: {name}"


def ensure_default_models() -> list[str]:
    messages = []
    for n in ("Phikon", "Phikon-v2"):
        if not (model_root() / MODEL_CATALOG[n]["file"]).exists():
            try:
                ok, msg = download_model(n)
                messages.append(msg if ok else f"{n}: {msg}")
            except Exception as exc:
                messages.append(f"{n}: нет интернета/доступа ({exc})")
    return messages
