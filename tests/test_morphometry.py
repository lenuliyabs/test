import math

from app.core.morphometry import (
    CalibrationProfile,
    area_to_um2,
    calculate_calibration_stats,
    calculate_um_per_px_from_divisions,
    convert_measurement_value,
    polygon_area,
    polyline_length,
    recalc_measurements_with_scale,
    thickness_distribution,
    to_um,
)


def test_polyline_length() -> None:
    pts = [(0, 0), (3, 4), (6, 8)]
    assert polyline_length(pts) == 10.0


def test_polygon_area_square() -> None:
    pts = [(0, 0), (2, 0), (2, 2), (0, 2)]
    assert polygon_area(pts) == 4.0


def test_to_um() -> None:
    assert to_um(10, 0.5) == 5.0
    assert to_um(10, None) is None


def test_area_to_um2() -> None:
    assert area_to_um2(4, 0.5) == 1.0
    assert area_to_um2(10, None) is None


def test_convert_measurement_line() -> None:
    v, u = convert_measurement_value(10, "line", 0.5)
    assert v == 5.0
    assert u == "мкм"


def test_convert_measurement_area_no_scale() -> None:
    v, u = convert_measurement_value(10, "area", None)
    assert v is None
    assert u == "px²"


def test_um_per_px_from_divisions() -> None:
    # 100 делений * 10 мкм = 1000 мкм; 500 px => 2.0 мкм/px
    assert calculate_um_per_px_from_divisions(500.0, 100, 10.0) == 2.0


def test_calibration_stats() -> None:
    stats = calculate_calibration_stats([500.0, 505.0, 495.0], 100, 10.0)
    assert math.isclose(stats["um_per_px"], 2.0, rel_tol=0.03)
    assert stats["n_repeats"] == 3
    assert stats["sd"] > 0


def test_recalc_measurements_preserves_px() -> None:
    data = [{"type": "line", "value_px": 20.0, "details": "x"}]
    out = recalc_measurements_with_scale(data, 0.5)
    assert out[0]["value_px"] == 20.0
    assert out[0]["value_um"] == 10.0


def test_recalc_measurements_changes_um_with_profile_switch() -> None:
    data = [{"type": "line", "value_px": 20.0, "details": "x"}]
    out1 = recalc_measurements_with_scale(data, 0.5)
    out2 = recalc_measurements_with_scale(data, 1.0)
    assert out1[0]["value_px"] == out2[0]["value_px"]
    assert out1[0]["value_um"] != out2[0]["value_um"]


def test_profile_serialization_roundtrip() -> None:
    p = CalibrationProfile(
        name="10x",
        objective="10x",
        um_per_px=0.65,
        date="2026-01-01T00:00:00",
        source_image_hash="abc",
        method="micrometer",
        n_repeats=3,
        sd=0.01,
    )
    p2 = CalibrationProfile.from_dict(p.to_dict())
    assert p2.name == p.name
    assert p2.um_per_px == p.um_per_px


def test_thickness_distribution_parallel_lines() -> None:
    l1 = [(0, 0), (10, 0)]
    l2 = [(0, 5), (10, 5)]
    stats = thickness_distribution(l1, l2, samples=100)
    assert math.isclose(stats["mean"], 5.0, rel_tol=0.05)
