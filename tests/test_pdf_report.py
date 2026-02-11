from pathlib import Path

import numpy as np

from app.export.pdf_report import export_pdf_report


def test_pdf_report_includes_guided_section(tmp_path: Path) -> None:
    p = tmp_path / "r.pdf"
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    export_pdf_report(
        str(p),
        img,
        img,
        mask,
        measurements=[{"type": "line", "value_px": 1}],
        calibration_info={"method": "line", "um_per_px": 0.5, "profile": "10x", "sd": 0.01},
        guided_ai={"steps": {"1": {"done": True}}},
    )
    assert p.exists()
    assert p.stat().st_size > 500
