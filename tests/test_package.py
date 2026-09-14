def test_package_exposes_version() -> None:
    import ftrec

    assert ftrec.__version__ == "0.1.0"
