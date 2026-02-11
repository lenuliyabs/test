from __future__ import annotations

import csv


def export_measurements_csv(path: str, measurements: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "value_px", "value_um", "details"])
        writer.writeheader()
        for m in measurements:
            writer.writerow(
                {
                    "type": m.get("type", ""),
                    "value_px": m.get("value_px", ""),
                    "value_um": m.get("value_um", ""),
                    "details": m.get("details", ""),
                }
            )
