from pathlib import Path

import numpy as np

from app.ai_engine.analytics import (
    collagen_index_vangieson_proxy,
    density_heatmap,
    detect_artifacts,
    fallback_segmentation,
    focus_heatmap,
    illumination_correction,
    illumination_gradient_score,
    nearest_neighbor_metrics,
    orientation_anisotropy,
    qc_metrics,
    reinhard_normalize,
    select_rois,
)
from app.ai_engine.engine import AIEngine
from app.export.geojson_export import export_geojson


def _make_sharp_blur_pair(size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[::8, :] = 255
    img[:, ::8] = 255
    blur = np.stack(
        [np.clip((img[..., 0].astype(np.float32) * 0.4), 0, 255).astype(np.uint8)] * 3, axis=2
    )
    return img, blur


def test_qc_sharp_vs_blur() -> None:
    sharp, blur = _make_sharp_blur_pair()
    ms = qc_metrics(sharp)
    mb = qc_metrics(blur)
    assert ms["sharpness"] > mb["sharpness"]


def test_focus_heatmap_shape() -> None:
    img = np.zeros((128, 160, 3), dtype=np.uint8)
    h = focus_heatmap(img, block=32)
    assert h.shape == (128, 160)


def test_normalization_shape_dtype_range() -> None:
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    n = reinhard_normalize(img)
    assert n.shape == img.shape
    assert n.dtype == np.uint8
    assert n.min() >= 0 and n.max() <= 255


def test_illumination_reduces_vignette_on_synthetic() -> None:
    h, w = 256, 256
    y, x = np.indices((h, w))
    r = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)
    v = np.clip(220 - r * 1.1, 0, 255).astype(np.uint8)
    img = np.stack([v, v, v], axis=2)
    before = illumination_gradient_score(v)
    corr = illumination_correction(img)
    after = illumination_gradient_score(corr[..., 0])
    assert after < before


def test_artifact_mask_detects_bright_blob() -> None:
    img = np.zeros((128, 128, 3), dtype=np.uint8) + 120
    img[40:60, 40:60] = 255
    m = detect_artifacts(img)
    assert m[50, 50] == 1


def test_roi_selection_bounds_and_count() -> None:
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    rois = select_rois(img, artifact_mask=None, tile_size=128, n_rois=6)
    assert len(rois) <= 6
    for x, y, w, h in rois:
        assert x >= 0 and y >= 0 and x + w <= 512 and y + h <= 512


def test_segmentation_fallback_works() -> None:
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[20:100, 20:100] = 200
    seg = fallback_segmentation(img)
    assert seg.mask.shape == img.shape[:2]
    assert 0.0 <= seg.confidence <= 1.0


def test_collagen_proxy_stable() -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[..., 0] = 200
    img[..., 1] = 80
    img[..., 2] = 70
    c = collagen_index_vangieson_proxy(img)
    assert c > 0


def test_orientation_anisotropy_stripes() -> None:
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[:, ::4] = 255
    o = orientation_anisotropy(img)
    assert 0 <= o["anisotropy"] <= 1


def test_nearest_neighbor_metrics_stable() -> None:
    pts = np.array([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=np.float32)
    m = nearest_neighbor_metrics(pts)
    assert m["mean"] > 0


def test_geojson_export_valid(tmp_path: Path) -> None:
    path = tmp_path / "a.geojson"
    export_geojson(str(path), [{"type": "line", "points": [(0, 0), (1, 1)]}], [{"type": "line"}])
    assert path.exists()
    txt = path.read_text(encoding="utf-8")
    assert "FeatureCollection" in txt


def test_cache_hit() -> None:
    eng = AIEngine()
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    r1 = eng.preview_stage("quality_check", img, None, {})
    r2 = eng.preview_stage("quality_check", img, None, {})
    assert r1.metrics == r2.metrics


def test_density_heatmap_shape() -> None:
    m = np.zeros((100, 100), dtype=np.uint8)
    m[10:30, 10:30] = 1
    h = density_heatmap(m, block=20)
    assert h.shape == m.shape


def test_engine_apply_stage_returns_image() -> None:
    eng = AIEngine()
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    res = eng.apply_stage("quality_check", img, [], {})
    assert res.image.shape == img.shape


def test_engine_segmentation_preview_has_confidence() -> None:
    eng = AIEngine()
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    res = eng.preview_stage("segmentation", img, None, {})
    assert "confidence" in res.metrics
