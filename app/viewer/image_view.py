from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


class ImageView(QGraphicsView):
    measurement_finished = Signal(dict)
    scale_line_finished = Signal(float)
    mask_changed = Signal(np.ndarray)
    cursor_moved = Signal(int, int)
    zoom_changed = Signal(float)
    compare_changed = Signal(bool, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)

        self.base_item = QGraphicsPixmapItem()
        self.overlay_item = QGraphicsPixmapItem()
        self.scene.addItem(self.base_item)
        self.scene.addItem(self.overlay_item)

        self.current_tool = "hand"
        self.brush_size = 10
        self.um_per_px: float | None = None
        self.mask_opacity = 0.4

        self.layers: dict[str, np.ndarray | None] = {
            "original": None,
            "normalized": None,
            "illumination": None,
            "enhanced": None,
            "artifacts": None,
            "roi": None,
            "segmentation": None,
        }
        self.layer_visible = {
            "original": False,
            "normalized": False,
            "illumination": False,
            "enhanced": True,
            "artifacts": False,
            "roi": False,
            "segmentation": True,
        }

        self.mask: np.ndarray | None = None

        self.compare_mode = False
        self.compare_left_layer = "original"
        self.compare_right_layer = "enhanced"
        self.split_ratio = 0.5
        self._drag_split = False

        self._is_panning = False
        self._space_pressed = False
        self._last_mouse_pos = QPoint()
        self._points: list[tuple[float, float]] = []
        self._preview_item: QGraphicsPathItem | None = None

    def set_tool(self, tool: str) -> None:
        self.current_tool = tool
        self._points.clear()

    def set_um_per_px(self, value: float | None) -> None:
        self.um_per_px = value

    def set_mask_opacity(self, alpha: float) -> None:
        self.mask_opacity = float(np.clip(alpha, 0.0, 1.0))
        self._refresh_scene()

    def set_layer_image(self, name: str, image: np.ndarray | None) -> None:
        if name in self.layers:
            self.layers[name] = image
            self._refresh_scene()

    def set_layer_visibility(self, name: str, visible: bool) -> None:
        if name in self.layer_visible:
            self.layer_visible[name] = visible
            self._refresh_scene()

    def set_compare_mode(self, enabled: bool) -> None:
        self.compare_mode = enabled
        self.compare_changed.emit(self.compare_mode, self.split_ratio)
        self._refresh_scene()

    def set_compare_layers(self, left: str, right: str) -> None:
        self.compare_left_layer = left
        self.compare_right_layer = right
        self._refresh_scene()

    def reset_split_center(self) -> None:
        self.split_ratio = 0.5
        self.compare_changed.emit(self.compare_mode, self.split_ratio)
        self._refresh_scene()

    def set_images(self, original: np.ndarray, enhanced: np.ndarray, mask: np.ndarray) -> None:
        self.layers["original"] = original
        self.layers["enhanced"] = enhanced
        self.mask = mask
        self._refresh_scene()

    def update_enhanced(self, enhanced: np.ndarray) -> None:
        self.layers["enhanced"] = enhanced
        self._refresh_scene()

    def update_mask(self, mask: np.ndarray) -> None:
        self.mask = mask
        self._refresh_scene()

    def fit_image(self) -> None:
        if self.base_item.pixmap().isNull():
            return
        self.fitInView(self.base_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.zoom_changed.emit(self.current_zoom_percent())

    def current_zoom_percent(self) -> float:
        return self.transform().m11() * 100.0

    def wheelEvent(self, event):
        if self.base_item.pixmap().isNull():
            return
        old_pos = self.mapToScene(event.position().toPoint())
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)
        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
        self.zoom_changed.emit(self.current_zoom_percent())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self._space_pressed = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self._space_pressed = False
            if not self._is_panning:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event):
        pos = self.mapToScene(event.pos())
        x, y = int(pos.x()), int(pos.y())
        if (
            self.compare_mode
            and event.button() == Qt.MouseButton.LeftButton
            and self._base_layer() is not None
        ):
            base = self._base_layer()
            if base is not None:
                divider_x = int(base.shape[1] * self.split_ratio)
                if abs(x - divider_x) <= 10:
                    self._drag_split = True
                    self.setCursor(Qt.CursorShape.SizeHorCursor)
                    return
        pan_mode = self.current_tool == "hand" or self._space_pressed
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and pan_mode
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
        self.cursor_moved.emit(int(pos.x()), int(pos.y()))
        if self._drag_split and self._base_layer() is not None:
            w = self._base_layer().shape[1]
            self.split_ratio = float(np.clip(pos.x() / max(w, 1), 0.0, 1.0))
            self.compare_changed.emit(self.compare_mode, self.split_ratio)
            self._refresh_scene()
            return
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
        if self._drag_split:
            self._drag_split = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        if self._is_panning:
            self._is_panning = False
            self.setCursor(
                Qt.CursorShape.OpenHandCursor if self._space_pressed else Qt.CursorShape.ArrowCursor
            )
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if (
            self.compare_mode
            and event.button() == Qt.MouseButton.LeftButton
            and self._base_layer() is not None
        ):
            pos = self.mapToScene(event.pos())
            divider_x = int(self._base_layer().shape[1] * self.split_ratio)
            if abs(int(pos.x()) - divider_x) <= 14:
                self.reset_split_center()
                return
        if event.button() == Qt.MouseButton.LeftButton and self.current_tool == "hand":
            self.fit_image()
            return
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

    def _layer_by_name(self, name: str) -> np.ndarray | None:
        return self.layers.get(name)

    def _base_layer(self) -> np.ndarray | None:
        for name in ["enhanced", "illumination", "normalized", "original"]:
            if self.layer_visible.get(name) and self.layers.get(name) is not None:
                return self.layers[name]
        return self.layers.get("enhanced") or self.layers.get("original")

    def _refresh_scene(self) -> None:
        img = self._base_layer()
        if img is None:
            return

        if self.compare_mode:
            left = self._layer_by_name(self.compare_left_layer)
            right = self._layer_by_name(self.compare_right_layer)
            if left is None:
                left = img
            if right is None:
                right = img
            if left.shape != right.shape:
                right = cv2.resize(
                    right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR
                )
            split_x = int(np.clip(self.split_ratio, 0.0, 1.0) * left.shape[1])
            composed = right.copy()
            composed[:, :split_x] = left[:, :split_x]
            # divider and handle
            composed[:, max(0, split_x - 1) : min(composed.shape[1], split_x + 1)] = (255, 255, 255)
            composed[
                max(0, composed.shape[0] // 2 - 20) : min(
                    composed.shape[0], composed.shape[0] // 2 + 20
                ),
                max(0, split_x - 3) : min(composed.shape[1], split_x + 3),
            ] = (200, 200, 200)
            img_to_show = composed
        else:
            img_to_show = img

        base = QPixmap.fromImage(self._to_qimage_rgb(img_to_show))
        self.base_item.setPixmap(base)

        overlay = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        if self.mask is not None and self.layer_visible.get("segmentation", True):
            overlay[self.mask > 0] = (255, 0, 0, int(self.mask_opacity * 255))

        art = self.layers.get("artifacts")
        if art is not None and self.layer_visible.get("artifacts", False):
            if art.ndim == 2:
                overlay[art > 0] = (255, 140, 0, 120)

        roi = self.layers.get("roi")
        if roi is not None and self.layer_visible.get("roi", False):
            if roi.ndim == 2:
                overlay[roi > 0] = (0, 255, 0, 80)

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
