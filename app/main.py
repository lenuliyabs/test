from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app.ui.main_window import MainWindow
from app.ui.style import apply_light_palette


def main() -> int:
    app = QApplication(sys.argv)
    apply_light_palette(app)
    window = MainWindow()
    window.show()

    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1])
        if candidate.exists() and candidate.is_file():
            QTimer.singleShot(0, lambda p=str(candidate): window.open_path(p))

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
