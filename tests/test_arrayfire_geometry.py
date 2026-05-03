import numpy as np

from i2sar.core.enums import LookSide
from i2sar.accel.arrayfire_backend import arrayfire_available
from i2sar.geometry import RadarGrid, WGS84, llh_to_ecef
from i2sar.geometry.accelerated_geometry import geo2rdr_fast, rdr2geo_fast
from i2sar.geometry.geo2rdr import geo2rdr, geo2rdr_arrayfire
from i2sar.geometry.rdr2geo import rdr2geo, rdr2geo_arrayfire


def test_rdr2geo_arrayfire_matches_numpy_for_vectorized_static_orbit():
    if not arrayfire_available():
        return

    sat_position = np.array([WGS84.a + 500_000.0, 0.0, 0.0])
    velocity = np.array([0.0, 3000.0, 0.0])
    radar_grid = RadarGrid(
        length=10,
        width=5,
        sensing_start_s=0.0,
        prf_hz=1000.0,
        starting_range_m=400_000.0,
        range_pixel_spacing_m=10.0,
    )
    lines = np.array([0.0, 5.0, 9.0])
    pixels = np.array([1.0, 2.0, 3.0])
    dem = np.zeros_like(lines)

    expected = rdr2geo(
        line=lines,
        pixel=pixels,
        radar_grid=radar_grid,
        satellite_position=sat_position,
        velocity=velocity,
        doppler=0.0,
        dem_elevation=dem,
    )
    actual = rdr2geo_arrayfire(
        line=lines,
        pixel=pixels,
        radar_grid=radar_grid,
        satellite_position=sat_position,
        velocity=velocity,
        doppler=0.0,
        dem=dem,
    )

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual[:2], expected[:2], atol=1e-6)
    assert np.all(np.isfinite(actual[2]))


def test_geo2rdr_arrayfire_matches_numpy_for_vectorized_static_orbit():
    if not arrayfire_available():
        return

    sat_position = np.array([WGS84.a + 500_000.0, 0.0, 0.0])
    velocity = np.array([0.0, 3000.0, 0.0])
    radar_grid = RadarGrid(
        length=10,
        width=5000,
        sensing_start_s=0.0,
        prf_hz=1000.0,
        starting_range_m=3_000_000.0,
        range_pixel_spacing_m=200.0,
    )
    lats = np.array([0.0, 0.0, 0.0])
    lons = np.array([1.0, 1.1, 1.2])
    heights = np.zeros_like(lats)

    expected = geo2rdr(
        lat=lats,
        lon=lons,
        height=heights,
        radar_grid=radar_grid,
        satellite_position=sat_position,
        velocity=velocity,
        doppler=0.0,
        look_side=LookSide.LEFT,
    )
    actual = geo2rdr_arrayfire(
        lat=lats,
        lon=lons,
        height=heights,
        radar_grid=radar_grid,
        satellite_position=sat_position,
        velocity=velocity,
        doppler=0.0,
        look_side=LookSide.LEFT,
    )

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_accelerated_geometry_accepts_explicit_arrayfire_method():
    if not arrayfire_available():
        return

    sat_position = np.array([WGS84.a + 500_000.0, 0.0, 0.0])
    velocity = np.array([0.0, 3000.0, 0.0])
    radar_grid = RadarGrid(
        length=10,
        width=5,
        sensing_start_s=0.0,
        prf_hz=1000.0,
        starting_range_m=400_000.0,
        range_pixel_spacing_m=10.0,
    )

    geo = rdr2geo_fast(
        line=np.array([0.0, 1.0]),
        pixel=np.array([1.0, 2.0]),
        radar_grid=radar_grid,
        satellite_position=sat_position,
        velocity=velocity,
        doppler=0.0,
        dem=0.0,
        method="arrayfire",
    )
    rdr = geo2rdr_fast(
        lat=np.array([60.0, 60.1]),
        lon=np.array([0.0, 0.1]),
        height=np.array([0.0, 0.0]),
        radar_grid=radar_grid,
        satellite_position=sat_position,
        velocity=velocity,
        doppler=0.0,
        method="arrayfire",
    )

    assert geo.shape == (3, 2)
    assert rdr.shape == (2, 2)
    assert np.all(np.isfinite(geo))
    assert np.all(np.isfinite(rdr))
