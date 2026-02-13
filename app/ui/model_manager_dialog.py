from __future__ import annotations

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)

from app.models.model_manager import DownloadWorker, delete_model, get_model_statuses


class ModelManagerDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Модели ИИ")
        self.resize(820, 520)
        self.pool = QThreadPool.globalInstance()
        self.worker: DownloadWorker | None = None

        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Модель", "Статус", "Версия", "Размер (MB)", "Хэш", "Путь"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        row = QHBoxLayout()
        self.btn_download = QPushButton("Скачать/Обновить")
        self.btn_delete = QPushButton("Удалить")
        self.btn_refresh = QPushButton("Обновить список")
        self.btn_cancel = QPushButton("Отмена")
        self.btn_cancel.setEnabled(False)
        row.addWidget(self.btn_download)
        row.addWidget(self.btn_delete)
        row.addWidget(self.btn_refresh)
        row.addWidget(self.btn_cancel)
        layout.addLayout(row)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(QLabel("Лог"))
        layout.addWidget(self.log)

        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_download.clicked.connect(self.download_selected)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_cancel.clicked.connect(self.cancel_download)

        self.refresh()

    def refresh(self) -> None:
        items = get_model_statuses()
        self.table.setRowCount(len(items))
        for i, m in enumerate(items):
            self.table.setItem(i, 0, QTableWidgetItem(m.name))
            self.table.setItem(i, 1, QTableWidgetItem("установлено" if m.installed else "нет"))
            self.table.setItem(i, 2, QTableWidgetItem(m.version))
            self.table.setItem(i, 3, QTableWidgetItem(str(m.size_mb)))
            self.table.setItem(i, 4, QTableWidgetItem(m.sha256))
            self.table.setItem(i, 5, QTableWidgetItem(m.path))

    def _selected_name(self) -> str | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        return self.table.item(rows[0].row(), 0).text()

    def download_selected(self) -> None:
        name = self._selected_name()
        if not name or self.worker is not None:
            return
        self.progress.setValue(0)
        self.worker = DownloadWorker(name)
        self.btn_cancel.setEnabled(True)
        self.worker.signals.progress.connect(self.progress.setValue)
        self.worker.signals.log.connect(lambda msg: self.log.append(msg))

        def _finish(ok: bool, msg: str):
            self.log.append(msg)
            self.worker = None
            self.btn_cancel.setEnabled(False)
            self.refresh()

        self.worker.signals.finished.connect(_finish)
        self.pool.start(self.worker)

    def cancel_download(self) -> None:
        if self.worker is not None:
            self.worker.cancel()

    def delete_selected(self) -> None:
        name = self._selected_name()
        if not name:
            return
        ok, msg = delete_model(name)
        self.log.append(msg)
        if ok:
            self.refresh()
