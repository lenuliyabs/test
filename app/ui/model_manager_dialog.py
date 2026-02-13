from __future__ import annotations

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)

from app.ai.models.downloader import DownloadPackWorker, remove_model
from app.ai.models.registry import check_installed, model_main_path, model_specs, models_dir


class ModelManagerDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ИИ → Модели")
        self.resize(860, 520)
        self.pool = QThreadPool.globalInstance()
        self.worker: DownloadPackWorker | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Папка моделей: {models_dir()}"))

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Ключ", "Модель", "Статус", "Путь"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        row = QHBoxLayout()
        self.btn_download = QPushButton("Скачать")
        self.btn_verify = QPushButton("Проверить")
        self.btn_delete = QPushButton("Удалить")
        self.btn_cancel = QPushButton("Отмена")
        self.btn_cancel.setEnabled(False)
        row.addWidget(self.btn_download)
        row.addWidget(self.btn_verify)
        row.addWidget(self.btn_delete)
        row.addWidget(self.btn_cancel)
        layout.addLayout(row)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(QLabel("Лог"))
        layout.addWidget(self.log)

        self.btn_verify.clicked.connect(self.refresh)
        self.btn_download.clicked.connect(self.download_pack)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_cancel.clicked.connect(self.cancel_download)

        self.refresh()

    def refresh(self) -> None:
        specs = model_specs()
        self.table.setRowCount(len(specs))
        for i, spec in enumerate(specs):
            ok, missing = check_installed(spec.key)
            status = "установлено" if ok else f"нет ({len(missing)} файлов)"
            self.table.setItem(i, 0, QTableWidgetItem(spec.key))
            self.table.setItem(i, 1, QTableWidgetItem(spec.title))
            self.table.setItem(i, 2, QTableWidgetItem(status))
            self.table.setItem(i, 3, QTableWidgetItem(str(model_main_path(spec.key))))

    def _selected_key(self) -> str | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        return self.table.item(rows[0].row(), 0).text()

    def download_pack(self) -> None:
        if self.worker is not None:
            return
        self.progress.setValue(0)
        self.worker = DownloadPackWorker()
        self.btn_cancel.setEnabled(True)
        self.worker.signals.progress.connect(self.progress.setValue)
        self.worker.signals.log.connect(lambda msg: self.log.append(msg))

        def _finish(ok: bool, msg: str):
            self.log.append(msg)
            if not ok:
                QMessageBox.information(self, "Модели", msg)
            self.worker = None
            self.btn_cancel.setEnabled(False)
            self.refresh()

        self.worker.signals.finished.connect(_finish)
        self.pool.start(self.worker)

    def cancel_download(self) -> None:
        if self.worker is not None:
            self.worker.cancel()

    def delete_selected(self) -> None:
        key = self._selected_key()
        if not key:
            return
        ok, msg = remove_model(key)
        self.log.append(msg)
        if not ok:
            QMessageBox.warning(self, "Удаление", msg)
        self.refresh()
