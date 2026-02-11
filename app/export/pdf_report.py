from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from app.ui.i18n_ru import DISCLAIMER


def _overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    color = np.zeros_like(out)
    color[..., 0] = 255
    alpha = (mask > 0).astype(np.float32) * 0.35
    out = (out * (1 - alpha[..., None]) + color * alpha[..., None]).astype(np.uint8)
    return out


def _to_reader(img_rgb: np.ndarray) -> ImageReader:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("Failed to encode image for PDF")
    return ImageReader(buf.tobytes())


def export_pdf_report(
    path: str,
    original: np.ndarray,
    enhanced: np.ndarray,
    mask: np.ndarray,
    measurements: list[dict],
    calibration_info: dict | None = None,
    guided_ai: dict | None = None,
) -> None:
    c = canvas.Canvas(path, pagesize=A4)
    w, h = A4
    c.setTitle("Отчёт HistoAnalyzer")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, h - 40, "HistoAnalyzer — отчёт анализа")
    c.setFont("Helvetica", 9)
    c.drawString(40, h - 55, DISCLAIMER)

    y = h - 300
    thumb_w, thumb_h = 170, 170
    for i, (title, img) in enumerate(
        [
            ("Оригинал", original),
            ("Улучшенное", enhanced),
            ("Оверлей маски", _overlay(enhanced, mask)),
        ]
    ):
        x = 40 + i * (thumb_w + 20)
        c.setFont("Helvetica", 10)
        c.drawString(x, y + thumb_h + 10, title)
        c.drawImage(_to_reader(img), x, y, width=thumb_w, height=thumb_h, preserveAspectRatio=True)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y - 25, "Морфометрия")

    table_y = y - 45
    c.setStrokeColor(colors.black)
    c.line(40, table_y, w - 40, table_y)
    c.setFont("Helvetica", 9)
    c.drawString(45, table_y - 14, "Тип")
    c.drawString(145, table_y - 14, "Значение (px)")
    c.drawString(245, table_y - 14, "Значение (мкм/мм)")
    c.drawString(345, table_y - 14, "Детали")
    c.line(40, table_y - 18, w - 40, table_y - 18)

    row_y = table_y - 32
    for m in measurements[:18]:
        c.drawString(45, row_y, str(m.get("type", ""))[:16])
        c.drawString(145, row_y, f"{m.get('value_px', '')}"[:14])
        c.drawString(245, row_y, f"{m.get('value_um', '')}"[:18])
        c.drawString(345, row_y, str(m.get("details", ""))[:36])
        row_y -= 14
        if row_y < 90:
            break

    c.showPage()
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, h - 40, "Параметры воспроизводимости")
    c.setFont("Helvetica", 10)

    y2 = h - 70
    cal = calibration_info or {}
    c.drawString(40, y2, "Калибровка:")
    y2 -= 16
    c.drawString(55, y2, f"Метод: {cal.get('method', 'не указан')}")
    y2 -= 14
    c.drawString(55, y2, f"Масштаб (мкм/px): {cal.get('um_per_px', 'не задан')}")
    y2 -= 14
    c.drawString(55, y2, f"Профиль: {cal.get('profile', 'нет')}")
    y2 -= 14
    c.drawString(55, y2, f"Дата: {cal.get('date', '—')} | SD: {cal.get('sd', '—')}")

    y2 -= 24
    c.drawString(40, y2, "Пошаговый ИИ-мастер:")
    y2 -= 16
    steps = (guided_ai or {}).get("steps", {})
    for k, v in list(steps.items())[:12]:
        c.drawString(55, y2, f"Шаг {k}: {'выполнен' if v.get('done') else 'не выполнен'}")
        y2 -= 14
        if y2 < 80:
            break

    c.setFont("Helvetica", 8)
    c.drawString(40, 35, DISCLAIMER)

    c.save()

    if not Path(path).exists():
        raise FileNotFoundError(path)
