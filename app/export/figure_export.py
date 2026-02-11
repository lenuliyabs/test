from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen


@dataclass
class FigureExportSettings:
    fmt: str = "PNG"
    size_mode: str = "screen"
    custom_width: int = 1920
    custom_height: int = 1080
    include_original: bool = True
    include_enhanced: bool = True
    include_segmentation: bool = True
    include_artifacts: bool = False
    include_heatmap: bool = False
    include_roi: bool = True
    include_annotations: bool = True
    overlay_alpha: int = 40
    show_title: bool = True
    title_text: str = "HistoAnalyzer — Figure"
    show_roi_labels: bool = True
    show_legend: bool = True
    show_scale_bar: bool = True
    scale_length: str = "100 мкм"
    scale_pos: str = "левый-низ"
    include_methods: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> "FigureExportSettings":
        data = json.loads(payload)
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


def target_size_from_mode(
    mode: str, screen_size: tuple[int, int], custom: tuple[int, int]
) -> tuple[int, int]:
    if mode == "a4_300":
        return 2480, 3508
    if mode == "a4_600":
        return 4961, 7016
    if mode == "custom":
        return custom
    return screen_size


def parse_scale_length_to_um(text: str) -> float:
    s = text.strip().lower().replace(" ", "")
    if s.endswith("мм"):
        return float(s[:-2].replace(",", ".")) * 1000.0
    if s.endswith("мкм"):
        return float(s[:-3].replace(",", "."))
    return 100.0


def scale_bar_px(um_per_px: float | None, scale_text: str) -> int | None:
    if um_per_px is None or um_per_px <= 0:
        return None
    um = parse_scale_length_to_um(scale_text)
    return max(1, int(round(um / um_per_px)))


def _overlay_mask(
    base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float
) -> np.ndarray:
    out = base.copy().astype(np.float32)
    active = mask > 0
    out[active] = out[active] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_annotations(p: QPainter, annotations: list[dict], scale_x: float, scale_y: float) -> None:
    p.setPen(QPen(QColor(20, 220, 20), 2))
    for a in annotations:
        pts = a.get("points", [])
        if len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            p.drawLine(int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))


def render_figure_image(
    settings: FigureExportSettings,
    original: np.ndarray | None,
    enhanced: np.ndarray | None,
    segmentation: np.ndarray | None,
    artifacts: np.ndarray | None,
    heatmap: np.ndarray | None,
    roi_mask: np.ndarray | None,
    annotations: list[dict],
    um_per_px: float | None,
    methods_text: str,
    screen_size: tuple[int, int] = (1600, 900),
) -> tuple[QImage, bool]:
    w, h = target_size_from_mode(
        settings.size_mode, screen_size, (settings.custom_width, settings.custom_height)
    )
    img = QImage(w, h, QImage.Format.Format_ARGB32)
    img.fill(QColor("white"))
    p = QPainter(img)
    p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

    left = original if settings.include_original and original is not None else enhanced
    right = enhanced if settings.include_enhanced and enhanced is not None else original
    if left is None and right is None:
        left = np.zeros((512, 512, 3), dtype=np.uint8)
        right = left
    if right is None:
        right = left
    if left is None:
        left = right

    margin = 40
    panel_h = h - 2 * margin - 100
    panel_w = (
        (w - 3 * margin) // 2
        if settings.include_original and settings.include_enhanced
        else w - 2 * margin
    )

    def draw_panel(
        panel_img: np.ndarray, x0: int, y0: int, title: str
    ) -> tuple[int, int, float, float]:
        resized = cv2.resize(panel_img, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)
        qimg = QImage(
            resized.data, panel_w, panel_h, panel_w * 3, QImage.Format.Format_RGB888
        ).copy()
        p.drawImage(x0, y0, qimg)
        p.setPen(QPen(QColor(20, 20, 20), 1))
        p.drawRect(QRectF(x0, y0, panel_w, panel_h))
        p.setFont(QFont("Arial", 11))
        p.drawText(x0, y0 - 8, title)
        return panel_w, panel_h, panel_w / panel_img.shape[1], panel_h / panel_img.shape[0]

    y0 = margin + 30
    if settings.include_original and settings.include_enhanced:
        _, _, sx1, sy1 = draw_panel(left, margin, y0, "До")
        _, _, sx2, sy2 = draw_panel(right, margin * 2 + panel_w, y0, "После")
    else:
        _, _, sx1, sy1 = draw_panel(right, margin, y0, "Результат")
        sx2, sy2 = sx1, sy1

    alpha = np.clip(settings.overlay_alpha / 100.0, 0.0, 1.0)
    # overlays only on right/result panel
    base_x = (
        margin * 2 + panel_w if settings.include_original and settings.include_enhanced else margin
    )
    base_img = right.copy()
    if settings.include_segmentation and segmentation is not None:
        base_img = _overlay_mask(base_img, segmentation, (255, 0, 0), alpha)
    if settings.include_artifacts and artifacts is not None:
        base_img = _overlay_mask(base_img, artifacts, (255, 140, 0), alpha)
    if settings.include_heatmap and heatmap is not None:
        hm = cv2.normalize(heatmap.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        hm_c = cv2.applyColorMap(hm, cv2.COLORMAP_TURBO)
        hm_c = cv2.cvtColor(hm_c, cv2.COLOR_BGR2RGB)
        base_img = ((1 - alpha) * base_img + alpha * hm_c).astype(np.uint8)
    if settings.include_roi and roi_mask is not None:
        base_img = _overlay_mask(base_img, roi_mask, (0, 255, 0), alpha * 0.8)

    resized = cv2.resize(base_img, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)
    qimg2 = QImage(resized.data, panel_w, panel_h, panel_w * 3, QImage.Format.Format_RGB888).copy()
    p.drawImage(base_x, y0, qimg2)

    if settings.include_annotations:
        _draw_annotations(p, annotations, sx2, sy2)

    p.setFont(QFont("Arial", 12, QFont.Bold))
    if settings.show_title:
        p.drawText(margin, 28, settings.title_text)

    scale_drawn = False
    if settings.show_scale_bar:
        px = scale_bar_px(um_per_px, settings.scale_length)
        if px is not None:
            p.setPen(QPen(QColor(10, 10, 10), 4))
            bar_px = int(px * sx2)
            yb = y0 + panel_h - 20
            xb = (
                base_x + 20 if settings.scale_pos == "левый-низ" else base_x + panel_w - bar_px - 20
            )
            p.drawLine(xb, yb, xb + bar_px, yb)
            p.setPen(QPen(QColor(20, 20, 20), 1))
            p.setFont(QFont("Arial", 10))
            p.drawText(xb, yb - 8, settings.scale_length)
            scale_drawn = True

    if settings.show_legend:
        p.setFont(QFont("Arial", 9))
        legend = []
        if settings.include_segmentation:
            legend.append("Красный: сегментация")
        if settings.include_artifacts:
            legend.append("Оранжевый: артефакты")
        if settings.include_roi:
            legend.append("Зелёный: ROI")
        for i, txt in enumerate(legend):
            p.drawText(margin, h - 70 + i * 14, txt)

    if settings.include_methods:
        p.setFont(QFont("Arial", 8))
        p.drawText(margin, h - 20, methods_text[:220])

    p.end()
    return img, scale_drawn


def export_figure(
    path: str,
    settings: FigureExportSettings,
    original: np.ndarray | None,
    enhanced: np.ndarray | None,
    segmentation: np.ndarray | None,
    artifacts: np.ndarray | None,
    heatmap: np.ndarray | None,
    roi_mask: np.ndarray | None,
    annotations: list[dict],
    um_per_px: float | None,
    methods_text: str,
    screen_size: tuple[int, int] = (1600, 900),
) -> bool:
    qimg, scale_drawn = render_figure_image(
        settings,
        original,
        enhanced,
        segmentation,
        artifacts,
        heatmap,
        roi_mask,
        annotations,
        um_per_px,
        methods_text,
        screen_size,
    )
    ok = qimg.save(path, settings.fmt.upper())
    if not ok:
        raise IOError(f"Не удалось сохранить фигуру: {path}")
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return scale_drawn
