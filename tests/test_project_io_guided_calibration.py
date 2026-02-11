from pathlib import Path

import numpy as np

from app.core.image_ops import EnhanceParams
from app.core.project_io import load_project, save_project


def test_project_serializes_guided_ai_and_calibration(tmp_path: Path) -> None:
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")
    project = tmp_path / "sample.histo"
    mask = np.zeros((20, 20), dtype=np.uint8)

    save_project(
        project_file=str(project),
        image_path=str(img),
        enhance=EnhanceParams(),
        mask=mask,
        annotations=[{"type": "line"}],
        measurements=[{"type": "line", "value_px": 10.0}],
        um_per_px=0.5,
        guided_ai={"current_step": 4, "steps": {"4": {"done": True}}},
        calibration_profiles=[{"name": "10x", "um_per_px": 0.5}],
        active_calibration_profile="10x",
    )

    payload, loaded_mask = load_project(str(project))
    assert payload["guided_ai"]["current_step"] == 4
    assert payload["calibration_profiles"][0]["name"] == "10x"
    assert payload["active_calibration_profile"] == "10x"
    assert loaded_mask.shape == mask.shape
