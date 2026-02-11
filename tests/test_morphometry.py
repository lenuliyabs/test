import math

from app.core.morphometry import polygon_area, polyline_length, thickness_distribution, to_um


def test_polyline_length() -> None:
    pts = [(0, 0), (3, 4), (6, 8)]
    assert polyline_length(pts) == 10.0


def test_polygon_area_square() -> None:
    pts = [(0, 0), (2, 0), (2, 2), (0, 2)]
    assert polygon_area(pts) == 4.0


def test_to_um() -> None:
    assert to_um(10, 0.5) == 5.0
    assert to_um(10, None) is None


def test_thickness_distribution_parallel_lines() -> None:
    l1 = [(0, 0), (10, 0)]
    l2 = [(0, 5), (10, 5)]
    stats = thickness_distribution(l1, l2, samples=100)
    assert math.isclose(stats["mean"], 5.0, rel_tol=0.05)
    assert stats["min"] > 4.5
    assert stats["max"] < 5.5
