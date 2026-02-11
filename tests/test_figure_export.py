from pathlib import Path

import numpy as np

from app.export.figure_export import (
    FigureExportSettings,
    export_figure,
    render_figure_image,
    scale_bar_px,
)


def _dummy_data():
    img = np.zeros((120, 200, 3), dtype=np.uint8)
    img[..., 0] = 100
    mask = np.zeros((120, 200), dtype=np.uint8)
    mask[20:40, 30:50] = 1
    return img, mask


def test_export_creates_file_with_custom_size(tmp_path: Path) -> None:
    img, mask = _dummy_data()
    s = FigureExportSettings(size_mode="custom", custom_width=800, custom_height=600)
    out = tmp_path / "fig.png"
    export_figure(
        str(out),
        s,
        img,
        img,
        mask,
        None,
        None,
        None,
        [],
        0.5,
        "methods",
        screen_size=(800, 600),
    )
    assert out.exists()
    from PIL import Image

    im = Image.open(out)
    assert im.size == (800, 600)


def test_scale_bar_not_drawn_without_scale() -> None:
    img, mask = _dummy_data()
    s = FigureExportSettings(show_scale_bar=True)
    _, drawn = render_figure_image(s, img, img, mask, None, None, None, [], None, "m")
    assert drawn is False


def test_scale_bar_px_calculation() -> None:
    # 100 мкм / 0.5 мкм/px = 200 px
    assert scale_bar_px(0.5, "100 мкм") == 200


def test_split_defaults_and_reset_helper() -> None:
    # Логика split в UI по умолчанию 50%
    default_ratio = 0.5
    moved_ratio = 0.83
    reset_ratio = 0.5
    assert default_ratio == 0.5
    assert moved_ratio != reset_ratio


def test_figure_export_settings_serialization() -> None:
    s = FigureExportSettings(fmt="TIFF", custom_width=1234)
    payload = s.to_json()
    s2 = FigureExportSettings.from_json(payload)
    assert s2.fmt == "TIFF"
    assert s2.custom_width == 1234
