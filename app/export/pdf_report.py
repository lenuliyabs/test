from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


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
) -> None:
    c = canvas.Canvas(path, pagesize=A4)
    w, h = A4
    c.setTitle("HistoAnalyzer report")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, h - 40, "HistoAnalyzer Report")

    y = h - 300
    thumb_w, thumb_h = 170, 170
    for i, (title, img) in enumerate(
        [("Original", original), ("Enhanced", enhanced), ("Mask Overlay", _overlay(enhanced, mask))]
    ):
        x = 40 + i * (thumb_w + 20)
        c.setFont("Helvetica", 10)
        c.drawString(x, y + thumb_h + 10, title)
        c.drawImage(_to_reader(img), x, y, width=thumb_w, height=thumb_h, preserveAspectRatio=True)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y - 25, "Measurements")

    table_y = y - 45
    c.setStrokeColor(colors.black)
    c.line(40, table_y, w - 40, table_y)
    c.setFont("Helvetica", 9)
    c.drawString(45, table_y - 14, "Type")
    c.drawString(145, table_y - 14, "Value (px)")
    c.drawString(245, table_y - 14, "Value (um)")
    c.drawString(345, table_y - 14, "Details")
    c.line(40, table_y - 18, w - 40, table_y - 18)

    row_y = table_y - 32
    for m in measurements[:20]:
        c.drawString(45, row_y, str(m.get("type", ""))[:16])
        c.drawString(145, row_y, f"{m.get('value_px', '')}"[:14])
        c.drawString(245, row_y, f"{m.get('value_um', '')}"[:14])
        c.drawString(345, row_y, str(m.get("details", ""))[:38])
        row_y -= 14
        if row_y < 70:
            break

    c.showPage()
    c.save()

    if not Path(path).exists():
        raise FileNotFoundError(path)
