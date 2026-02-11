from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Signal


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)


class CancellableWorker(QRunnable):
    def __init__(self, fn: Callable[..., Any], *args, **kwargs) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.cancel_requested = False

    def cancel(self) -> None:
        self.cancel_requested = True

    def run(self) -> None:
        try:
            if self.cancel_requested:
                return
            result = self.fn(*self.args, **self.kwargs)
            if not self.cancel_requested:
                self.signals.finished.emit(result)
        except Exception as exc:  # pragma: no cover
            self.signals.error.emit(str(exc))
