import h5py
import json
import numpy as np

from i2sar import Project
from i2sar.core.enums import AcquisitionMode
from i2sar.dem import (
    create_dem_product_from_scene_corners,
    scene_bbox_from_corners,
    srtm_tiles_for_bbox,
    write_dem_hdf,
    write_dem_product_from_hgt,
)
from i2sar.model import SceneInfo


def test_scene_bbox_from_corners_expands_by_default_point_two_degrees():
    corners = [
        {"lat": 30.0, "lon": 100.0},
        {"lat": 30.5, "lon": 101.0},
        {"lat": 31.0, "lon": 100.5},
    ]

    assert scene_bbox_from_corners(corners) == [99.8, 101.2, 29.8, 31.2]


def test_srtm_tiles_for_bbox_uses_expanded_bbox_extent():
    tiles = srtm_tiles_for_bbox([99.8, 101.2, 29.8, 31.2])

    assert tiles == [
        "N29E099",
        "N29E100",
        "N29E101",
        "N30E099",
        "N30E100",
        "N30E101",
        "N31E099",
        "N31E100",
        "N31E101",
    ]


def test_write_dem_hdf_stores_elevation_grid_and_metadata(tmp_path):
    dem_path = tmp_path / "dem.h5"
    elevation = np.array([[100.0, 101.0], [102.0, np.nan]], dtype=np.float32)
    geotransform = [99.8, 0.1, 0.0, 31.2, 0.0, -0.1]

    write_dem_hdf(
        dem_path,
        dem_id="dem_scene",
        elevation=elevation,
        geotransform=geotransform,
        bbox=[99.8, 100.0, 31.0, 31.2],
        source={"type": "array"},
        margin_deg=0.2,
    )

    with h5py.File(dem_path, "r") as h5:
        assert h5.attrs["entity_type"] == "product"
        assert h5.attrs["product_type"] == "dem"
        assert h5["dem/elevation"].shape == (2, 2)
        assert h5["dem/elevation"].dtype == np.dtype("float32")
        assert h5["dem/mask"][1, 1] == False
        assert h5["dem"].attrs["crs"] == "EPSG:4326"
        assert list(h5["dem"].attrs["bbox"]) == [99.8, 100.0, 31.0, 31.2]
        assert h5["provenance"].attrs["margin_deg"] == 0.2


def test_write_dem_product_from_hgt_registers_project_product(tmp_path):
    hgt_path = tmp_path / "N30E100.hgt"
    np.array([[1, 2], [3, -32768]], dtype=">i2").tofile(hgt_path)
    project = Project.create(tmp_path / "project", name="project")

    dem_path = write_dem_product_from_hgt(project, hgt_path, product_id="dem_n30e100")

    with h5py.File(dem_path, "r") as dem_h5:
        assert dem_h5["dem/elevation"][0, 0] == 1.0
        assert np.isnan(dem_h5["dem/elevation"][1, 1])
        assert list(dem_h5["dem"].attrs["geotransform"]) == [100.0, 1.0, 0.0, 31.0, 0.0, -1.0]

    with h5py.File(project.path, "r") as project_h5:
        assert project_h5["products/dem_n30e100"].attrs["product_type"] == "dem"


def test_create_dem_product_from_scene_corners_records_default_margin(tmp_path):
    project = Project.create(tmp_path / "project", name="project")
    scene_path = project.create_scene(SceneInfo("scene_a", "lutan", AcquisitionMode.STRIPMAP, "2026-01-01T00:00:00Z"))
    with h5py.File(scene_path, "a") as h5:
        h5["derived"].attrs["json"] = json.dumps(
            {
                "sceneCorners": [
                    {"lat": 30.0, "lon": 100.0},
                    {"lat": 30.5, "lon": 101.0},
                ]
            }
        )
    elevation = np.ones((2, 2), dtype=np.float32)

    dem_path = create_dem_product_from_scene_corners(
        project,
        "scene_a",
        elevation=elevation,
        geotransform=[99.8, 0.7, 0.0, 30.7, 0.0, -0.45],
    )

    with h5py.File(dem_path, "r") as h5:
        assert list(h5["dem"].attrs["bbox"]) == [99.8, 101.2, 29.8, 30.7]
        assert h5["provenance"].attrs["margin_deg"] == 0.2
        source = json.loads(h5["provenance"].attrs["source_json"])
        assert source["scene_id"] == "scene_a"
        assert source["tiles"] == ["N29E099", "N29E100", "N29E101", "N30E099", "N30E100", "N30E101"]
