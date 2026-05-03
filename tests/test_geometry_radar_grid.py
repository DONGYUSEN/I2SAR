import json

import h5py
import numpy as np

from i2sar.geometry import SPEED_OF_LIGHT, RadarGrid, read_radar_grid


def test_radar_grid_converts_line_pixel_to_time_and_slant_range():
    grid = RadarGrid(
        length=100,
        width=200,
        sensing_start_s=1000.0,
        prf_hz=2000.0,
        starting_range_m=800000.0,
        range_pixel_spacing_m=5.0,
    )

    assert grid.line_to_azimuth_time(10.0) == 1000.005
    assert np.isclose(grid.azimuth_time_to_line(1000.005), 10.0)
    assert grid.pixel_to_slant_range(3.0) == 800015.0
    assert grid.slant_range_to_pixel(800015.0) == 3.0


def test_radar_grid_vectorizes_line_pixel_conversions():
    grid = RadarGrid(
        length=100,
        width=200,
        sensing_start_s=0.0,
        prf_hz=10.0,
        starting_range_m=100.0,
        range_pixel_spacing_m=2.0,
    )

    assert np.allclose(grid.line_to_azimuth_time(np.array([0.0, 5.0])), [0.0, 0.5])
    assert np.allclose(grid.pixel_to_slant_range(np.array([0.0, 5.0])), [100.0, 110.0])


def test_radar_grid_from_mapping_supports_slant_range_time_first_pixel():
    grid = RadarGrid.from_mapping(
        {
            "numberOfRows": 10,
            "numberOfColumns": 20,
            "prf": 1000.0,
            "sensingStartSeconds": 5.0,
            "rangeTimeFirstPixel": 0.001,
            "rangePixelSpacing": 3.0,
        }
    )

    assert grid.length == 10
    assert grid.width == 20
    assert np.isclose(grid.starting_range_m, SPEED_OF_LIGHT * 0.001 / 2.0)
    assert grid.range_pixel_spacing_m == 3.0


def test_read_radar_grid_from_scene_hdf5_json_attrs(tmp_path):
    scene_path = tmp_path / "scene.h5"
    with h5py.File(scene_path, "w") as h5:
        group = h5.require_group("radar_grid")
        group.attrs["json"] = json.dumps(
            {
                "numberOfRows": 10,
                "numberOfColumns": 20,
                "prf": 1000.0,
                "sensingStartSeconds": 7.0,
                "startingRange": 800000.0,
                "rangePixelSpacing": 4.0,
            }
        )

    grid = read_radar_grid(scene_path)

    assert grid.line_to_azimuth_time(1.0) == 7.001
    assert grid.pixel_to_slant_range(2.0) == 800008.0
