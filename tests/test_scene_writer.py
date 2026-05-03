import json

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


def _parsed_scene(tmp_path) -> ParsedScene:
    slc_path = tmp_path / "scene.tiff"
    data = np.array([[[1, -1]]], dtype=np.int16)
    _write_multiband_tiff(slc_path, data)
    orbit = {
        "smoothed": True,
        "smoothing": {"algorithm": "isce2-lutan1-orbit-filter", "status": "applied"},
        "stateVectors": [
            {"gpsTime": 1.0, "posX": 1.0, "posY": 2.0, "posZ": 3.0, "velX": 4.0, "velY": 5.0, "velZ": 6.0}
        ],
    }
    raw = {"stateVectors": [{**orbit["stateVectors"][0], "posX": 10.0}]}
    return ParsedScene(
        scene_id="lutan_scene",
        sensor="lutan",
        acquisition_mode=AcquisitionMode.STRIPMAP,
        acquisition_time="2026-04-28T00:00:00Z",
        acquisition={"polarisation": "HH"},
        scene={"sceneCorners": [{"lat": 0.0, "lon": 0.0}]},
        radar_grid={"numberOfRows": 10, "numberOfColumns": 20},
        orbit=orbit,
        orbit_raw=raw,
        doppler={"combinedDoppler": {"coefficients": [0.0]}},
        slc=SourceRef(str(slc_path)),
        slc_attrs={"sample_format": "iq_int16", "storage_layout": "two_band_iq", "complex_band_count": 1},
    )


def test_write_parsed_scene_populates_hdf5_metadata(tmp_path):
    project = Project.create(tmp_path / "project", name="project")
    result = write_parsed_scene(project, _parsed_scene(tmp_path))

    with h5py.File(result.scene_path, "r") as h5:
        assert h5.attrs["sensor"] == "lutan"
        assert h5["metadata/acquisition"].attrs["json"]
        assert h5["radar_grid"].attrs["json"]
        assert h5["orbit"].attrs["smoothed"] == True
        assert json.loads(h5["orbit"].attrs["smoothing_json"])["status"] == "applied"
        assert h5["orbit/time"].shape == (1,)
        assert h5["orbit/position"].shape == (1, 3)
        assert h5["orbit_raw/position"][0, 0] == 10.0
        assert h5["slc"].attrs["sample_format"] == "iq_int16"
        assert h5["slc"].attrs["storage_layout"] == "two_band_iq"
        assert h5["slc/data"].shape == (1, 1)


def test_write_parsed_scene_keeps_empty_orbit_vectors_two_dimensional(tmp_path):
    project = Project.create(tmp_path / "project", name="project")
    slc_path = tmp_path / "scene.tiff"
    import tifffile
    tifffile.imwrite(slc_path, np.array([[1 + 2j]], dtype=np.complex64))
    parsed = ParsedScene(
        scene_id="empty_orbit",
        sensor="sentinel1",
        acquisition_mode=AcquisitionMode.TOPS,
        acquisition_time="2026-04-28T00:00:00Z",
        orbit={"stateVectors": []},
        orbit_raw={"stateVectors": []},
        slc=SourceRef(str(slc_path)),
        slc_attrs={"sample_format": "cfloat32", "storage_layout": "single_band_complex", "complex_band_count": 1},
    )

    result = write_parsed_scene(project, parsed)

    with h5py.File(result.scene_path, "r") as h5:
        assert h5["orbit/position"].shape == (0, 3)
        assert h5["orbit/velocity"].shape == (0, 3)
        assert h5["orbit_raw/position"].shape == (0, 3)
