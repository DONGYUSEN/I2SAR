import numpy as np

from i2sar.core.enums import LookSide
from i2sar.geometry import RadarGrid, WGS84
from i2sar.geometry.accelerated_geometry import geo2rdr_fast, rdr2geo_fast
from i2sar.geometry.geo2rdr import compute_geo2rdr_mapping, geo2rdr
from i2sar.geometry.rdr2geo import compute_rdr2geo_mapping, rdr2geo


def _static_grid():
    return RadarGrid(
        length=10,
        width=100,
        sensing_start_s=0.0,
        prf_hz=1000.0,
        starting_range_m=400_000.0,
        range_pixel_spacing_m=10.0,
    )


def test_numba_rdr2geo_matches_numpy_for_static_orbit_vectors():
    sat_position = np.array([WGS84.a + 500_000.0, 0.0, 0.0])
    velocity = np.array([0.0, 3000.0, 0.0])
    grid = _static_grid()
    lines = np.array([0.0, 5.0, 9.0])
    pixels = np.array([1.0, 2.0, 3.0])

    expected = rdr2geo(
        lines,
        pixels,
        grid,
        sat_position,
        velocity,
        0.0,
        dem_elevation=np.zeros_like(lines),
    )
    actual = rdr2geo_fast(
        lines,
        pixels,
        grid,
        sat_position,
        velocity,
        0.0,
        dem=0.0,
        method="numba",
    )

    np.testing.assert_allclose(actual[:2], expected[:2], atol=1e-6)
    np.testing.assert_allclose(actual[2], expected[2], atol=1e-6)


def test_numba_geo2rdr_matches_numpy_for_static_orbit_vectors():
    sat_position = np.array([WGS84.a + 500_000.0, 0.0, 0.0])
    velocity = np.array([0.0, 3000.0, 0.0])
    grid = _static_grid()
    lats = np.array([0.0, 0.0, 0.0])
    lons = np.array([1.0, 1.1, 1.2])
    heights = np.zeros_like(lats)

    expected = geo2rdr(
        lats,
        lons,
        heights,
        grid,
        sat_position,
        velocity,
        0.0,
        look_side=LookSide.LEFT,
    )
    actual = geo2rdr_fast(
        lats,
        lons,
        heights,
        grid,
        sat_position,
        velocity,
        0.0,
        method="numba",
        look_side=LookSide.LEFT,
    )

    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_compute_rdr2geo_mapping_uses_vectorized_rdr2geo(monkeypatch):
    import importlib

    module = importlib.import_module("i2sar.geometry.rdr2geo")

    calls = {"count": 0}
    original = module.rdr2geo

    def counting_rdr2geo(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "rdr2geo", counting_rdr2geo)

    grid = RadarGrid(
        length=4,
        width=3,
        sensing_start_s=0.0,
        prf_hz=1000.0,
        starting_range_m=400_000.0,
        range_pixel_spacing_m=10.0,
    )
    sat_position = np.array([WGS84.a + 500_000.0, 0.0, 0.0])
    velocity = np.array([0.0, 3000.0, 0.0])

    lats, lons, heights = compute_rdr2geo_mapping(
        radar_grid=grid,
        satellite_position=sat_position,
        velocity=velocity,
        doppler=0.0,
        dem_elevation=np.zeros((4, 3)),
        method="original",
    )

    assert calls["count"] == 1
    assert lats.shape == (4, 3)
    assert np.all(np.isfinite(lons))
    assert np.all(np.isfinite(heights))


def test_compute_geo2rdr_mapping_uses_vectorized_geo2rdr(monkeypatch):
    import importlib

    module = importlib.import_module("i2sar.geometry.geo2rdr")

    calls = {"count": 0}
    original = module.geo2rdr

    def counting_geo2rdr(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "geo2rdr", counting_geo2rdr)

    grid = _static_grid()
    sat_position = np.array([WGS84.a + 500_000.0, 0.0, 0.0])
    velocity = np.array([0.0, 3000.0, 0.0])

    aztimes, ranges = compute_geo2rdr_mapping(
        lats=np.array([0.0, 0.1, 0.2]),
        lons=np.array([1.0, 1.1]),
        radar_grid=grid,
        satellite_position=sat_position,
        velocity=velocity,
        doppler=0.0,
    )

    assert calls["count"] == 1
    assert aztimes.shape == (3, 2)
    assert np.all(np.isfinite(ranges))


def test_compute_rdr2geo_mapping_supports_explicit_numba_method():
    grid = RadarGrid(
        length=3,
        width=4,
        sensing_start_s=0.0,
        prf_hz=1000.0,
        starting_range_m=400_000.0,
        range_pixel_spacing_m=10.0,
    )
    sat_position = np.array([0.0, 0.0, WGS84.a + 500_000.0])
    velocity = np.array([0.0, 3000.0, 0.0])
    dem = np.zeros((grid.length, grid.width))

    expected = compute_rdr2geo_mapping(
        radar_grid=grid,
        satellite_position=sat_position,
        velocity=velocity,
        doppler=0.0,
        dem_elevation=dem,
        method="original",
    )
    actual = compute_rdr2geo_mapping(
        radar_grid=grid,
        satellite_position=sat_position,
        velocity=velocity,
        doppler=0.0,
        dem_elevation=dem,
        method="numba",
    )

    for actual_arr, expected_arr in zip(actual, expected):
        np.testing.assert_allclose(actual_arr, expected_arr, atol=1e-6)


def test_compute_geo2rdr_mapping_supports_explicit_numba_method():
    grid = _static_grid()
    sat_position = np.array([WGS84.a + 500_000.0, 0.0, 0.0])
    velocity = np.array([0.0, 3000.0, 0.0])
    lats = np.array([0.0, 0.1, 0.2])
    lons = np.array([1.0, 1.1])

    expected = compute_geo2rdr_mapping(
        lats=lats,
        lons=lons,
        radar_grid=grid,
        satellite_position=sat_position,
        velocity=velocity,
        doppler=0.0,
        method="original",
        look_side=LookSide.LEFT,
    )
    actual = compute_geo2rdr_mapping(
        lats=lats,
        lons=lons,
        radar_grid=grid,
        satellite_position=sat_position,
        velocity=velocity,
        doppler=0.0,
        method="numba",
        look_side=LookSide.LEFT,
    )

    for actual_arr, expected_arr in zip(actual, expected):
        np.testing.assert_allclose(actual_arr, expected_arr, atol=1e-6)
