from __future__ import annotations

from PySide6.QtGui import QColor, QPalette

APP_QSS = """
QMainWindow {
    background: #f4f6f8;
}
QToolBar {
    spacing: 6px;
    padding: 6px;
    border: none;
    background: #ffffff;
}
QToolButton {
    padding: 6px 8px;
    border-radius: 6px;
}
QToolButton:checked {
    background: #d9ecff;
    border: 1px solid #84bfff;
}
QTabWidget::pane {
    border: 1px solid #d7dde5;
    border-radius: 8px;
    background: #ffffff;
}
QTabBar::tab {
    padding: 8px 12px;
    margin: 2px;
    border-radius: 6px;
}
QTabBar::tab:selected {
    background: #d9ecff;
}
QGroupBox {
    border: 1px solid #d7dde5;
    border-radius: 8px;
    margin-top: 10px;
    padding: 8px;
    background: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}
QPushButton {
    padding: 8px 10px;
    border-radius: 6px;
    border: 1px solid #c7d2de;
    background: #ffffff;
}
QPushButton[primary="true"] {
    background: #2d7ef7;
    color: white;
    border: 1px solid #2d7ef7;
    font-weight: 600;
}
QStatusBar {
    background: #ffffff;
    border-top: 1px solid #d7dde5;
}
QDockWidget {
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}
"""


def apply_light_palette(app) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(244, 246, 248))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(27, 31, 35))
    pal.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(245, 245, 245))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))
    pal.setColor(QPalette.ColorRole.Text, QColor(33, 37, 41))
    pal.setColor(QPalette.ColorRole.Button, QColor(255, 255, 255))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(33, 37, 41))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(45, 126, 247))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)


def apply_dark_palette(app) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(37, 40, 45))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(226, 230, 235))
    pal.setColor(QPalette.ColorRole.Base, QColor(30, 32, 36))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(44, 47, 53))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(44, 47, 53))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(230, 230, 230))
    pal.setColor(QPalette.ColorRole.Text, QColor(223, 227, 232))
    pal.setColor(QPalette.ColorRole.Button, QColor(50, 54, 60))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(226, 230, 235))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(64, 143, 255))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)
