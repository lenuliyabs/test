from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QComboBox, QDialog, QFormLayout, QLabel, QPushButton, QVBoxLayout

from app.core.acceleration import benchmark_acceleration, detect_acceleration


class SettingsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Настройки")
        self.settings = QSettings("HistoAnalyzer", "HistoAnalyzer")

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.vendor_combo = QComboBox()
        self.vendor_combo.addItems(["Auto", "NVIDIA", "AMD", "Intel", "CPU"])
        saved = self.settings.value("accel_vendor", "Auto", type=str)
        idx = self.vendor_combo.findText(saved)
        if idx >= 0:
            self.vendor_combo.setCurrentIndex(idx)
        form.addRow("Производительность / Вендор", self.vendor_combo)

        self.device_label = QLabel("Устройство: —")
        self.backend_label = QLabel("Бэкенд: —")
        self.details_label = QLabel("Детали: —")
        self.details_label.setWordWrap(True)
        form.addRow(self.device_label)
        form.addRow(self.backend_label)
        form.addRow(self.details_label)

        layout.addLayout(form)

        self.test_btn = QPushButton("Проверить ускорение")
        self.save_btn = QPushButton("Сохранить")
        self.close_btn = QPushButton("Закрыть")
        layout.addWidget(self.test_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.close_btn)

        self.vendor_combo.currentTextChanged.connect(self.refresh_info)
        self.test_btn.clicked.connect(self.run_benchmark)
        self.save_btn.clicked.connect(self.save)
        self.close_btn.clicked.connect(self.accept)

        self.refresh_info(self.vendor_combo.currentText())

    def refresh_info(self, text: str) -> None:
        info = detect_acceleration(text)
        self.device_label.setText(f"Устройство: {info.vendor}")
        self.backend_label.setText(f"Бэкенд: {info.backend}")
        self.details_label.setText(f"Детали: {info.details}")

    def run_benchmark(self) -> None:
        m = benchmark_acceleration(self.vendor_combo.currentText())
        self.details_label.setText(
            f"{m['details']}\nТест: {m['iterations']} итераций за {m['time_s']} сек."
        )

    def save(self) -> None:
        self.settings.setValue("accel_vendor", self.vendor_combo.currentText())
        self.accept()
