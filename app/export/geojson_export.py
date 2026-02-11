from __future__ import annotations

import json
from datetime import datetime


def export_geojson(path: str, annotations: list[dict], measurements: list[dict]) -> None:
    features = []
    for ann in annotations:
        points = ann.get("points", [])
        if not points:
            continue
        coords = [[float(x), float(y)] for x, y in points]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"kind": ann.get("type", "annotation")},
            }
        )
    for m in measurements:
        features.append(
            {
                "type": "Feature",
                "geometry": None,
                "properties": {
                    "kind": "measurement",
                    "type": m.get("type"),
                    "value_px": m.get("value_px"),
                    "value_um": m.get("value_um"),
                    "units": m.get("units"),
                    "details": m.get("details"),
                },
            }
        )
    payload = {
        "type": "FeatureCollection",
        "features": features,
        "meta": {"exported_at": datetime.now().isoformat(), "generator": "HistoAnalyzer"},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
