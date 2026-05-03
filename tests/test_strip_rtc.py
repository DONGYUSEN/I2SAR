from __future__ import annotations

import json
import zipfile
from pathlib import Path

import h5py
import numpy as np

from i2sar import Project
from i2sar.core.enums import AcquisitionMode
from i2sar.io.base import ImportResult, SourceRef
from i2sar.model import SceneInfo


def _write_hgt_zip(path: Path, member: str, values: np.ndarray) -> None:
    raw = np.asarray(values, dtype=">i2")
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, raw.tobytes())


def test_ensure_dem_product_for_scene_uses_cached_srtm_zip(tmp_path):
    from i2sar.rtc.strip_rtc import ensure_dem_product_for_scene

    project = Project.create(tmp_path / "project", name="project")
    scene_path = project.create_scene(
        SceneInfo("strip_scene", "tianyi", AcquisitionMode.STRIPMAP, "2023-11-10T04:39:48")
    )
    with h5py.File(scene_path, "a") as h5:
        h5["derived"].attrs["json"] = json.dumps(
            {
                "sceneCorners": [
                    {"lat": 29.1, "lon": 94.1},
                    {"lat": 29.2, "lon": 94.2},
                ]
            }
        )

    dem_cache = tmp_path / "dem"
    dem_cache.mkdir()
    _write_hgt_zip(
        dem_cache / "N29E094.SRTMGL1.hgt.zip",
        "N29E094.hgt",
        np.array([[1, 2], [3, -32768]], dtype=np.int16),
    )

    dem_path = ensure_dem_product_for_scene(
        project,
        "strip_scene",
        dem_cache=dem_cache,
        margin_deg=0.0,
        download_missing=False,
    )

    with h5py.File(dem_path, "r") as h5:
        assert h5["dem/elevation"].shape == (2, 2)
        assert h5["dem/elevation"][0, 0] == 1.0
        assert np.isnan(h5["dem/elevation"][1, 1])
        assert list(h5["dem"].attrs["geotransform"]) == [94.0, 1.0, 0.0, 30.0, 0.0, -1.0]


def test_run_strip_rtc_uses_import_scene_and_processing_pipeline(monkeypatch, tmp_path):
    from i2sar.rtc import strip_rtc

    calls = {}

    def fake_import_scene(project, source, *, sensor="auto", acquisition_mode="auto"):
        calls["import"] = (project, Path(source), sensor, acquisition_mode)
        scene_path = project.create_scene(
            SceneInfo("strip_scene", "tianyi", AcquisitionMode.STRIPMAP, "2023-11-10T04:39:48")
        )
        with h5py.File(scene_path, "a") as h5:
            h5["derived"].attrs["json"] = json.dumps(
                {"sceneCorners": [{"lat": 29.1, "lon": 94.1}, {"lat": 29.2, "lon": 94.2}]}
            )
            h5["slc"].attrs["path"] = str(tmp_path / "source.zip")
            h5["slc"].attrs["storage"] = "zip"
            h5["slc"].attrs["member"] = "measurement/scene.tiff"
        return ImportResult(
            scene_id="strip_scene",
            scene_path=scene_path,
            sensor="tianyi",
            acquisition_mode=AcquisitionMode.STRIPMAP,
            slc=SourceRef(str(tmp_path / "source.zip"), storage="zip", member="measurement/scene.tiff"),
        )

    def fake_ensure_dem(project, scene_id, **kwargs):
        calls["dem"] = (project, scene_id, kwargs)
        dem_path = project.root / "products" / "dem_strip_scene.h5"
        dem_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(dem_path, "w") as h5:
            h5.require_group("dem").create_dataset("elevation", data=np.zeros((2, 2), dtype=np.float32))
        return dem_path

    def fake_process(scene_h5_path, dem_h5_path, output_dir, **kwargs):
        calls["process"] = (Path(scene_h5_path), Path(dem_h5_path), Path(output_dir), kwargs)
        out = Path(output_dir) / "strip_scene_strip_rtc.tif"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"rtc")
        return strip_rtc.StripRTCResult(
            project_path=Path(scene_h5_path).parents[1] / "project.h5",
            scene_id="strip_scene",
            scene_h5_path=Path(scene_h5_path),
            dem_h5_path=Path(dem_h5_path),
            output_dir=Path(output_dir),
            rtc_png=out,
            kml_file=Path(output_dir) / "strip_scene_strip_rtc.kml",
            metadata_h5=Path(output_dir) / "strip_scene_strip_rtc_metadata.h5",
            epsg=32646,
            output_resolution=3.5,
            full_resolution=True,
        )

    monkeypatch.setattr(strip_rtc, "import_scene", fake_import_scene)
    monkeypatch.setattr(strip_rtc, "ensure_dem_product_for_scene", fake_ensure_dem)
    monkeypatch.setattr(strip_rtc, "process_strip_scene_rtc", fake_process)

    result = strip_rtc.run_strip_rtc(
        tmp_path / "source.zip",
        output_dir=tmp_path / "out",
        sensor="tianyi",
        dem_cache=tmp_path / "dem",
    )

    assert calls["import"][2:] == ("tianyi", "auto")
    assert calls["dem"][1] == "strip_scene"
    assert calls["process"][0] == calls["import"][0].root / "scenes" / "strip_scene.h5"
    assert result.rtc_png.read_bytes() == b"rtc"


def test_run_strip_rtc_reuses_existing_imported_scene(monkeypatch, tmp_path):
    from i2sar.rtc import strip_rtc

    source = tmp_path / "source.zip"
    source.write_bytes(b"zip")
    project = Project.create(tmp_path / "out" / "project", name="strip_rtc")
    scene_path = project.create_scene(
        SceneInfo("strip_scene", "tianyi", AcquisitionMode.STRIPMAP, "2023-11-10T04:39:48")
    )
    with h5py.File(scene_path, "a") as h5:
        h5["derived"].attrs["json"] = json.dumps({"sceneCorners": [{"lat": 29.1, "lon": 94.1}]})
        h5["slc"].attrs["path"] = str(source.resolve())
        h5["slc"].attrs["storage"] = "zip"
        h5["slc"].attrs["member"] = "measurement/scene.tiff"

    def fail_import(*args, **kwargs):
        raise AssertionError("run_strip_rtc should reuse the existing imported scene")

    def fake_ensure_dem(project, scene_id, **kwargs):
        dem_path = project.root / "products" / "dem_strip_scene.h5"
        dem_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(dem_path, "w") as h5:
            h5.require_group("dem").create_dataset("elevation", data=np.zeros((2, 2), dtype=np.float32))
        return dem_path

    def fake_process(scene_h5_path, dem_h5_path, output_dir, **kwargs):
        out = Path(output_dir) / "strip_scene_strip_rtc.tif"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"rtc")
        return strip_rtc.StripRTCResult(
            project_path=project.path,
            scene_id="strip_scene",
            scene_h5_path=Path(scene_h5_path),
            dem_h5_path=Path(dem_h5_path),
            output_dir=Path(output_dir),
            rtc_png=out,
            kml_file=Path(output_dir) / "strip_scene_strip_rtc.kml",
            metadata_h5=Path(output_dir) / "strip_scene_strip_rtc_metadata.h5",
            epsg=32646,
            output_resolution=3.5,
            full_resolution=True,
        )

    monkeypatch.setattr(strip_rtc, "import_scene", fail_import)
    monkeypatch.setattr(strip_rtc, "ensure_dem_product_for_scene", fake_ensure_dem)
    monkeypatch.setattr(strip_rtc, "process_strip_scene_rtc", fake_process)

    result = strip_rtc.run_strip_rtc(
        source,
        output_dir=tmp_path / "out",
        sensor="tianyi",
    )

    assert result.scene_h5_path == scene_path


def test_slc_block_to_power_uses_complex_magnitude_squared():
    from i2sar.rtc.strip_rtc import _slc_block_to_power

    block = np.array([[1.0 + 2.0j, 3.0 - 4.0j]], dtype=np.complex64)

    power = _slc_block_to_power(block, scale=2.0)

    np.testing.assert_allclose(power, np.array([[10.0, 50.0]], dtype=np.float32))
    assert power.dtype == np.float32


def test_stretch_valid_data_to_uint8_masks_invalid_values():
    from i2sar.rtc.strip_rtc import _stretch_valid_to_uint8

    data = np.arange(101, dtype=np.float32).reshape(1, 101)
    valid_mask = np.isfinite(data)

    stretched = _stretch_valid_to_uint8(data, valid_mask)

    assert stretched.dtype == np.uint8
    assert stretched[0, 1] == 0
    assert stretched[0, 5] == 0
    assert stretched[0, 95] == 255
    assert stretched[0, 99] == 255


def test_stretch_valid_data_to_uint8_masks_nan_and_inf():
    from i2sar.rtc.strip_rtc import _stretch_valid_to_uint8

    data = np.array([[np.nan, 0.0, 50.0], [100.0, np.inf, 75.0]], dtype=np.float32)
    stretched = _stretch_valid_to_uint8(data, np.isfinite(data))

    assert stretched[0, 0] == 0
    assert stretched[1, 1] == 0


def test_log10_power_to_uint8_masks_invalid_and_non_positive_values():
    from i2sar.rtc.strip_rtc import _log10_power_to_uint8

    data = np.array([[0.0, 1.0, 10.0], [100.0, -5.0, np.inf]], dtype=np.float32)

    stretched = _log10_power_to_uint8(data, lower_percent=0.0, upper_percent=100.0)

    assert stretched.dtype == np.uint8
    assert stretched[0, 0] == 0
    assert stretched[1, 1] == 0
    assert stretched[1, 2] == 0
    assert stretched[0, 1] == 0
    assert stretched[0, 2] in (127, 128)
    assert stretched[1, 0] == 255


def test_rtc_metadata_filename_uses_satellite_and_acquisition_date(tmp_path):
    from i2sar.rtc.strip_rtc import _rtc_metadata_filename

    project = Project.create(tmp_path / "project", name="project")
    scene_path = project.create_scene(
        SceneInfo("strip_scene", "tianyi", AcquisitionMode.STRIPMAP, "2023-11-10T04:39:48")
    )

    assert _rtc_metadata_filename(scene_path, "strip_scene", {}) == "Tianyi_20231110_RTC.h5"


def test_process_strip_scene_rtc_writes_png_not_final_tif(tmp_path):
    from osgeo import gdal
    from i2sar.rtc.strip_rtc import process_strip_scene_rtc

    source_tif = tmp_path / "source.tif"
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(source_tif), 4, 4, 1, gdal.GDT_CFloat32)
    ds.GetRasterBand(1).WriteArray(np.full((4, 4), 1.0 + 2.0j, dtype=np.complex64))
    ds = None

    project = Project.create(tmp_path / "project", name="project")
    scene_path = project.create_scene(
        SceneInfo("strip_scene", "tianyi", AcquisitionMode.STRIPMAP, "2023-11-10T04:39:48")
    )
    with h5py.File(scene_path, "a") as h5:
        h5["derived"].attrs["json"] = json.dumps(
            {
                "sceneCorners": [
                    {"line": 0, "pixel": 0, "lat": 0.0, "lon": 0.0},
                    {"line": 0, "pixel": 3, "lat": 0.0, "lon": 0.001},
                    {"line": 3, "pixel": 0, "lat": -0.001, "lon": 0.0},
                    {"line": 3, "pixel": 3, "lat": -0.001, "lon": 0.001},
                ]
            }
        )
        h5["slc"].attrs["path"] = str(source_tif)
        h5["slc"].attrs["storage"] = "file"
        h5["radar_grid"].attrs["json"] = json.dumps(
            {
                "numberOfRows": 4,
                "numberOfColumns": 4,
                "rowSpacing": 2.0,
                "columnSpacing": 2.0,
                "prf": 1000.0,
                "rangeTimeFirstPixel": 0.004,
                "sensing_start_s": 0.0,
            }
        )
        h5["metadata/acquisition"].attrs["json"] = json.dumps({"centerFrequency": 5.4e9})
        
        orbit_grp = h5["orbit"]
        orbit_grp.create_dataset("time", data=np.array([0.0, 1.0], dtype=np.float64))
        orbit_grp.create_dataset(
            "position",
            data=np.array([[7_000_000.0, 0.0, 0.0], [7_000_000.0, 0.0, 0.0]], dtype=np.float64),
        )
        orbit_grp.create_dataset(
            "velocity",
            data=np.array([[0.0, 7000.0, 0.0], [0.0, 7000.0, 0.0]], dtype=np.float64),
        )

    dem_path = tmp_path / "dem.h5"
    with h5py.File(dem_path, "w") as h5:
        dem = h5.require_group("dem")
        dem.create_dataset("elevation", data=np.zeros((4, 4), dtype=np.float32))
        dem.create_dataset("mask", data=np.ones((4, 4), dtype=bool))
        dem.attrs["geotransform"] = np.array([-0.01, 0.01, 0.0, 0.01, 0.0, -0.01], dtype=np.float64)

    result = process_strip_scene_rtc(
        scene_path,
        dem_path,
        tmp_path / "out",
        output_resolution=100.0,
        scene_id="strip_scene",
    )

    assert result.rtc_png.exists()
    assert result.rtc_png.suffix == ".png"
    assert result.rtc_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert result.metadata_h5 == tmp_path / "out" / "Tianyi_20231110_RTC.h5"
    assert result.metadata_h5.exists()
    assert result.kml_file.exists()
    kml_text = result.kml_file.read_text(encoding="utf-8")
    assert "<GroundOverlay>" in kml_text
    assert "<href>strip_scene_strip_rtc.png</href>" in kml_text
    assert "<gx:LatLonQuad>" in kml_text
    assert "<coordinates>" in kml_text
    assert "<LatLonBox>" in kml_text
    assert "<north>0." in kml_text
    assert "<east>0." in kml_text
    assert not (tmp_path / "out" / "strip_scene_strip_rtc.tif").exists()
