import numpy as np
import pytest


def test_multilook_basic():
    """测试基本多视处理"""
    data = np.ones((4, 4), dtype=np.complex64) * (1 + 1j)
    nalks, nrlks = 2, 2

    from i2sar.processing.multilook import multilook
    result = multilook(data, nalks, nrlks)

    assert result.shape == (2, 2)
    assert np.allclose(result, (1 + 1j))


def test_multilook_chunked(tmp_path):
    """测试分块多视处理"""
    from osgeo import gdal

    input_path = tmp_path / "input.tiff"
    data = np.random.rand(1000, 1000).astype(np.float32) + 1j * np.random.rand(1000, 1000).astype(np.float32)
    ds = gdal.GetDriverByName('GTiff').Create(str(input_path), 1000, 1000, 1, gdal.GDT_CFloat32)
    ds.GetRasterBand(1).WriteArray(data)
    ds = None

    output_path = tmp_path / "output.tiff"

    from i2sar.processing.multilook import multilook_chunked
    result = multilook_chunked(str(input_path), str(output_path), nalks=4, nrlks=4, chunk_lines=100)

    assert result == True
    assert output_path.exists()

    out_ds = gdal.Open(str(output_path))
    assert out_ds.RasterXSize == 250
    assert out_ds.RasterYSize == 250


def test_multilook_preserves_phase():
    """测试相位保持多视处理"""
    phase_const = np.pi / 4
    data = np.ones((8, 8), dtype=np.complex64) * np.exp(1j * phase_const)
    nalks, nrlks = 2, 2

    from i2sar.processing.multilook import multilook
    result = multilook(data, nalks, nrlks)

    result_phase = np.angle(result)
    expected_phase = np.pi / 4
    assert np.allclose(result_phase, expected_phase, atol=1e-6)