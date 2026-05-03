from i2sar.cli.main import main


def test_cli_create_project(tmp_path):
    root = tmp_path / "demo"
    code = main(["create-project", str(root), "--name", "demo"])

    assert code == 0
    assert (root / "project.h5").exists()


def test_cli_run_empty_rtc_workflow(tmp_path):
    root = tmp_path / "demo"
    assert main(["create-project", str(root), "--name", "demo"]) == 0
    assert main(["run-empty", str(root / "project.h5"), "--workflow", "rtc", "--owner-id", "scene1"]) == 0


def test_multilook_cli(tmp_path):
    """测试multilook CLI命令"""
    from osgeo import gdal
    import numpy as np

    input_path = tmp_path / "input.tiff"
    data = np.random.rand(100, 100).astype(np.float32) + 1j * np.random.rand(100, 100).astype(np.float32)
    ds = gdal.GetDriverByName('GTiff').Create(str(input_path), 100, 100, 1, gdal.GDT_CFloat32)
    ds.GetRasterBand(1).WriteArray(data)
    ds = None

    output_path = tmp_path / "output.tiff"

    result = main([
        "multilook",
        str(input_path), str(output_path),
        "--nalks", "2", "--nrlks", "2"
    ])

    assert result == 0
    assert output_path.exists()
