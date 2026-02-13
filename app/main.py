from __future__ import annotations

import sys
from pathlib import Path

from app.ai.models.downloader import download_model_pack


def main() -> int:
    if "--download-models" in sys.argv:
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

    app = QApplication(sys.argv)
    apply_light_palette(app)
    window = MainWindow()
    window.show()

    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        candidate = Path(sys.argv[1])
        if candidate.exists() and candidate.is_file():
            QTimer.singleShot(0, lambda p=str(candidate): window.open_path(p))

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
