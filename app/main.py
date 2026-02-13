from __future__ import annotations

import sys
from pathlib import Path

from app.models.model_manager import ensure_default_models


def main() -> int:
    if "--download-models" in sys.argv:
        msgs = ensure_default_models()
        for m in msgs:
            print(m)
        return 0

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
