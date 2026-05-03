import h5py
import numpy as np

from i2sar import Project
from i2sar.core.enums import AcquisitionMode
from i2sar.io.base import ParsedScene, SourceRef
from i2sar.io.scene_writer import write_parsed_scene


def _write_multiband_tiff(filepath, data):
    try:
        from osgeo import gdal
    except ImportError:
        import tifffile

        tifffile.imwrite(filepath, data)
        return

    gdal.UseExceptions()
    driver = gdal.GetDriverByName("GTiff")
    rows, cols, bands = data.shape
    out_ds = driver.Create(str(filepath), cols, rows, bands, gdal.GDT_Int16)
    for idx in range(bands):
        out_ds.GetRasterBand(idx + 1).WriteArray(data[..., idx])
    out_ds = None


def test_scene_writer_imports_two_band_iq_int16_as_single_compound_complex_dataset(tmp_path):
    tiff_path = tmp_path / "iq.tiff"
    data = np.array(
        [
            [[1, -1], [2, -2], [3, -3]],
            [[4, -4], [5, -5], [6, -6]],
        ],
        dtype=np.int16,
    )
    _write_multiband_tiff(tiff_path, data)
    project = Project.create(tmp_path / "project", name="project")
    parsed = ParsedScene(
        scene_id="iq_scene",
        sensor="lutan",
        acquisition_mode=AcquisitionMode.STRIPMAP,
        acquisition_time="2026-04-28T00:00:00Z",
        slc=SourceRef(str(tiff_path)),
        slc_attrs={"sample_format": "iq_int16", "storage_layout": "two_band_iq", "complex_band_count": 1},
    )

    result = write_parsed_scene(project, parsed)

    with h5py.File(result.scene_path, "r") as h5:
        data = h5["slc/data"]
        assert data.shape == (2, 3)
        assert data.dtype.fields["real"][0] == np.dtype("int16")
        assert data.dtype.fields["imag"][0] == np.dtype("int16")
        assert data[0, 1]["real"] == 2
        assert data[0, 1]["imag"] == -2
        assert h5["slc"].attrs["hdf5_complex_layout"] == "compound_real_imag"
        assert h5["slc"].attrs["source_storage_layout"] == "two_band_iq"


def test_scene_writer_imports_single_band_complex64_preserving_float_parts(tmp_path):
    tiff_path = tmp_path / "complex.tiff"
    import tifffile
    tifffile.imwrite(
        tiff_path,
        np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex64),
    )
    project = Project.create(tmp_path / "project", name="project")
    parsed = ParsedScene(
        scene_id="complex_scene",
        sensor="sentinel1",
        acquisition_mode=AcquisitionMode.TOPS,
        acquisition_time="2026-04-28T00:00:00Z",
        slc=SourceRef(str(tiff_path)),
        slc_attrs={"sample_format": "cfloat32", "storage_layout": "single_band_complex", "complex_band_count": 1},
    )

    result = write_parsed_scene(project, parsed)

    with h5py.File(result.scene_path, "r") as h5:
        data = h5["slc/data"]
        assert data.shape == (2, 2)
        assert data.dtype.fields["real"][0] == np.dtype("float32")
        assert data.dtype.fields["imag"][0] == np.dtype("float32")
        assert np.isclose(data[1, 0]["real"], 5.0)
        assert np.isclose(data[1, 0]["imag"], 6.0)
