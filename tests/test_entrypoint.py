from app.entrypoint import main


def test_entrypoint_selftest_flag() -> None:
    assert main(["--selftest"]) == 0


def test_entrypoint_help_flag() -> None:
    assert main(["--help"]) == 0
