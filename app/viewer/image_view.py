from __future__ import annotations


import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt, Signal
from PySide6.QtGui import QImage, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


class ImageView(QGraphicsView):
    measurement_finished = Signal(dict)
    scale_line_finished = Signal(float)
    mask_changed = Signal(np.ndarray)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHints(self.renderHints() | self.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)

        self.base_item = QGraphicsPixmapItem()
        self.overlay_item = QGraphicsPixmapItem()
        self.scene.addItem(self.base_item)
        self.scene.addItem(self.overlay_item)

        self.current_tool = "hand"
        self.brush_size = 10
        self.um_per_px: float | None = None

        self.original: np.ndarray | None = None
        self.enhanced: np.ndarray | None = None
        self.mask: np.ndarray | None = None

        self._is_panning = False
        self._last_mouse_pos = QPoint()
        self._points: list[tuple[float, float]] = []
        self._preview_item: QGraphicsPathItem | None = None

    def set_tool(self, tool: str) -> None:
        self.current_tool = tool
        self._points.clear()

    def set_um_per_px(self, value: float | None) -> None:
        self.um_per_px = value

    def set_images(self, original: np.ndarray, enhanced: np.ndarray, mask: np.ndarray) -> None:
        self.original = original
        self.enhanced = enhanced
        self.mask = mask
        self._refresh_scene()

    def update_enhanced(self, enhanced: np.ndarray) -> None:
        self.enhanced = enhanced
        self._refresh_scene()

    def update_mask(self, mask: np.ndarray) -> None:
        self.mask = mask
        self._refresh_scene()

    def fit_image(self) -> None:
        self.fitInView(self.base_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        pos = self.mapToScene(event.pos())
        x, y = int(pos.x()), int(pos.y())
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and self.current_tool == "hand"
        ):
            self._is_panning = True
            self._last_mouse_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if self.mask is not None and self.current_tool in {"brush", "eraser"}:
            value = 1 if self.current_tool == "brush" else 0
            self._paint_mask(x, y, value)
            return

        if self.current_tool in {
            "line",
            "polyline",
            "area",
            "scale_line",
            "thickness_1",
            "thickness_2",
        }:
            self._points.append((pos.x(), pos.y()))
            self._update_preview()
            if self.current_tool == "line" and len(self._points) == 2:
                self._finish_line_measurement("line")
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = self.mapToScene(event.pos())
        if self._is_panning:
            delta = event.pos() - self._last_mouse_pos
            self._last_mouse_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return

        if self.mask is not None and event.buttons() & Qt.MouseButton.LeftButton:
            if self.current_tool in {"brush", "eraser"}:
                value = 1 if self.current_tool == "brush" else 0
                self._paint_mask(int(pos.x()), int(pos.y()), value)
                return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._is_panning:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.current_tool in {"polyline", "area", "scale_line", "thickness_1", "thickness_2"}:
            if self.current_tool == "polyline":
                self._finish_line_measurement("polyline")
            elif self.current_tool == "area":
                self._finish_area()
            elif self.current_tool == "scale_line":
                self._finish_scale_line()
            else:
                self._finish_polyline_annotation(self.current_tool)
            return
        super().mouseDoubleClickEvent(event)

    def _paint_mask(self, x: int, y: int, value: int) -> None:
        if self.mask is None:
            return
        h, w = self.mask.shape
        if x < 0 or y < 0 or x >= w or y >= h:
            return
        cv2.circle(self.mask, (x, y), self.brush_size, value, thickness=-1)
        self.mask_changed.emit(self.mask.copy())
        self._refresh_scene()

    def _to_qimage_rgb(self, arr: np.ndarray) -> QImage:
        h, w, ch = arr.shape
        return QImage(arr.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

    def _to_qimage_rgba(self, arr: np.ndarray) -> QImage:
        h, w, ch = arr.shape
        return QImage(arr.data, w, h, ch * w, QImage.Format.Format_RGBA8888).copy()

    def _refresh_scene(self) -> None:
        if self.enhanced is None:
            return
        base = QPixmap.fromImage(self._to_qimage_rgb(self.enhanced))
        self.base_item.setPixmap(base)
        self.overlay_item.setPixmap(QPixmap())
        if self.mask is not None:
            overlay = np.zeros((self.mask.shape[0], self.mask.shape[1], 4), dtype=np.uint8)
            overlay[self.mask > 0] = (255, 0, 0, 100)
            self.overlay_item.setPixmap(QPixmap.fromImage(self._to_qimage_rgba(overlay)))

        self.scene.setSceneRect(self.base_item.boundingRect())

    def _update_preview(self) -> None:
        if self._preview_item is not None:
            self.scene.removeItem(self._preview_item)
            self._preview_item = None
        if len(self._points) < 2:
            return
        path = QPainterPath(QPointF(*self._points[0]))
        for p in self._points[1:]:
            path.lineTo(QPointF(*p))
        item = QGraphicsPathItem(path)
        item.setPen(QPen(Qt.GlobalColor.green, 2))
        self.scene.addItem(item)
        self._preview_item = item

    def _finish_line_measurement(self, kind: str) -> None:
        if len(self._points) < 2:
            return
        points = self._points.copy()
        self._points.clear()
        self._update_preview()
        self.measurement_finished.emit({"type": kind, "points": points})

    def _finish_area(self) -> None:
        if len(self._points) < 3:
            return
        points = self._points.copy()
        self._points.clear()
        self._update_preview()
        self.measurement_finished.emit({"type": "area", "points": points})

    def _finish_scale_line(self) -> None:
        if len(self._points) < 2:
            return
        p1, p2 = self._points[0], self._points[-1]
        dist = float(np.linalg.norm(np.array(p1) - np.array(p2)))
        self._points.clear()
        self._update_preview()
        self.scale_line_finished.emit(dist)

    def _finish_polyline_annotation(self, kind: str) -> None:
        if len(self._points) < 2:
            return
        points = self._points.copy()
        self._points.clear()
        self._update_preview()
        self.measurement_finished.emit({"type": kind, "points": points})
