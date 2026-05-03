def test_package_imports():
    import i2sar

    assert i2sar.__version__ == "0.1.0"


def test_console_entrypoint_target_imports():
    from i2sar.cli.main import main

    assert callable(main)
