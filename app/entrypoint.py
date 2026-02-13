from __future__ import annotations

import sys
from pathlib import Path

from app.ai.models.downloader import download_model_pack

APP_VERSION = "0.1.0"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if "--help" in args or "-h" in args:
        print(
            "HistoAnalyzer\n"
            "Usage:\n"
            "  python -m app [--download-models] [--version] [--selftest] [image_or_project_path]\n"
        )
        return 0

    if "--version" in args:
        print(APP_VERSION)
        return 0

    if "--selftest" in args:
        print("selftest: ok")
        return 0

    if "--download-models" in args:
        def _progress(v: int) -> None:
            print(f"download: {v}%")

        def _log(m: str) -> None:
            print(m)

        ok, msg = download_model_pack(progress_cb=_progress, log_cb=_log)
        print(msg)
        return 0 if ok else 1

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from app.ui.main_window import MainWindow
    from app.ui.style import apply_light_palette

    app = QApplication(sys.argv if argv is None else [sys.argv[0], *args])
    apply_light_palette(app)
    window = MainWindow()
    window.show()

    candidates = [a for a in args if not a.startswith("--")]
    if candidates:
        candidate = Path(candidates[0])
        if candidate.exists() and candidate.is_file():
            QTimer.singleShot(0, lambda p=str(candidate): window.open_path(p))

    return app.exec()
