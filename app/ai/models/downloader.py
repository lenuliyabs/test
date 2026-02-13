from __future__ import annotations

import hashlib
import shutil
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Signal

from app.ai.models.registry import check_installed, get_spec, model_specs, models_dir

MODEL_PACK_URL = (
    "https://github.com/histoanalyzer/modelpack/releases/latest/download/ModelPack.zip"
)
MODEL_PACK_SHA256 = ""
LOG_NAME = "modelpack_install.log"


class DownloadSignals(QObject):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal(bool, str)


class DownloadPackWorker(QRunnable):
    def __init__(self, url: str = MODEL_PACK_URL, sha256_hex: str = MODEL_PACK_SHA256) -> None:
        super().__init__()
        self.url = url
        self.sha256_hex = sha256_hex
        self.cancel_requested = False
        self.signals = DownloadSignals()

    def cancel(self) -> None:
        self.cancel_requested = True

    def run(self) -> None:
        try:
            ok, msg = download_model_pack(
                url=self.url,
                sha256_hex=self.sha256_hex,
                progress_cb=self.signals.progress.emit,
                log_cb=self.signals.log.emit,
                cancel_cb=lambda: self.cancel_requested,
            )
            self.signals.finished.emit(ok, msg)
        except Exception as exc:  # pragma: no cover
            self.signals.finished.emit(False, str(exc))


def _write_log(line: str) -> None:
    root = models_dir()
    root.mkdir(parents=True, exist_ok=True)
    with open(root / LOG_NAME, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {line}\n")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model_pack(
    url: str = MODEL_PACK_URL,
    sha256_hex: str = MODEL_PACK_SHA256,
    progress_cb=None,
    log_cb=None,
    cancel_cb=None,
) -> tuple[bool, str]:
    root = models_dir()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "ModelPack.zip.part"

    def log(msg: str) -> None:
        _write_log(msg)
        if log_cb:
            log_cb(msg)

    log(f"Model pack URL: {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(archive, "wb") as out:
            total = int(resp.headers.get("Content-Length", "0") or 0)
            read = 0
            while True:
                if cancel_cb and cancel_cb():
                    archive.unlink(missing_ok=True)
                    log("Скачивание отменено")
                    return False, "Скачивание отменено"
                chunk = resp.read(1024 * 512)
                if not chunk:
                    break
                out.write(chunk)
                read += len(chunk)
                if progress_cb and total > 0:
                    progress_cb(int(read * 100 / total))
    except urllib.error.URLError:
        log("Нет подключения к интернету. Можно скачать позже: ИИ → Модели.")
        return False, "Нет интернета. Скачайте модели позже в ИИ → Модели."

    actual = _sha256(archive)
    if sha256_hex and actual.lower() != sha256_hex.lower():
        archive.unlink(missing_ok=True)
        log(f"SHA256 mismatch: expected={sha256_hex} actual={actual}")
        return False, "Ошибка проверки SHA256 ModelPack.zip"
    log(f"SHA256 ok: {actual}")

    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(root)
    except zipfile.BadZipFile:
        archive.unlink(missing_ok=True)
        log("Архив повреждён")
        return False, "ModelPack.zip повреждён"
    finally:
        archive.unlink(missing_ok=True)

    if progress_cb:
        progress_cb(100)

    missing_any = []
    for spec in model_specs():
        ok, missing = check_installed(spec.key)
        if not ok:
            missing_any.extend(missing)
    if missing_any:
        missing_text = ", ".join(str(p) for p in missing_any[:5])
        log(f"После распаковки не найдены файлы: {missing_text}")
        return False, "ModelPack распакован не полностью"

    log("Модели установлены и готовы к оффлайн-работе")
    return True, "Модели установлены"


def remove_model(key: str) -> tuple[bool, str]:
    spec = get_spec(key)
    root = models_dir()
    removed = 0
    for rel in spec.paths:
        p = root / rel
        if p.is_file():
            p.unlink(missing_ok=True)
            removed += 1
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
    if removed == 0:
        return False, f"{spec.title}: файлы не найдены"
    return True, f"{spec.title}: удалено объектов {removed}"
