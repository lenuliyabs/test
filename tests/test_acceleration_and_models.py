from app.core.acceleration import benchmark_acceleration, detect_acceleration
from app.models.model_manager import MODEL_CATALOG, model_root


def test_detect_acceleration_returns_backend() -> None:
    info = detect_acceleration("auto")
    assert info.backend in {"CUDA", "DirectML", "OpenVINO", "CPU"}


def test_benchmark_acceleration_has_time() -> None:
    m = benchmark_acceleration("cpu", iterations=5)
    assert m["time_s"] >= 0


def test_model_root_exists() -> None:
    root = model_root()
    root.mkdir(parents=True, exist_ok=True)
    assert root.exists()


def test_model_catalog_contains_required_items() -> None:
    for name in ["Phikon", "Phikon-v2", "Cellpose nuclei", "Cellpose cyto", "Cellpose cpsam"]:
        assert name in MODEL_CATALOG
