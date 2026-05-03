from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from i2sar.core.enums import LookSide
from i2sar.geometry import RadarGrid, llh_to_ecef
from i2sar.geometry.accelerated_geometry import geo2rdr_fast, rdr2geo_fast
from i2sar.geometry.geo2rdr import geo2rdr
from i2sar.geometry.look_side import validate_look_side
from i2sar.geometry.rdr2geo import rdr2geo
from i2sar.orbit.interpolate import OrbitInterpolator


ISCE3_REE_RSLC = Path("/home/ysdong/Software/isce/isce3/tests/data/REE_RSLC_out17.h5")
SPEED_OF_LIGHT = 299_792_458.0


def _read_scalar(dataset):
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode()
    if hasattr(value, "shape") and value.shape == ():
        value = value.item()
        if isinstance(value, bytes):
            return value.decode()
    return value


def _real_ree_rslc_geometry():
    with h5py.File(ISCE3_REE_RSLC, "r") as h5:
        base = "science/LSAR/SLC"
        swath = f"{base}/swaths"
        freq = f"{swath}/frequencyA"
        orbit_group = h5[f"{base}/metadata/orbit"]

        orbit = OrbitInterpolator(
            orbit_group["time"][...],
            orbit_group["position"][...],
            orbit_group["velocity"][...],
        )
        slant_range = h5[f"{freq}/slantRange"][...]
        zero_doppler_time = h5[f"{swath}/zeroDopplerTime"][...]
        image_shape = h5[f"{freq}/HH"].shape
        zero_doppler_time_spacing = float(h5[f"{swath}/zeroDopplerTimeSpacing"][()])
        center_frequency = float(h5[f"{freq}/processedCenterFrequency"][()])
        look_side = LookSide(
            _read_scalar(h5["science/LSAR/identification/lookDirection"]).strip().lower()
        )

    radar_grid = RadarGrid(
        length=image_shape[0],
        width=image_shape[1],
        sensing_start_s=float(zero_doppler_time[0]),
        prf_hz=1.0 / zero_doppler_time_spacing,
        starting_range_m=float(slant_range[0]),
        range_pixel_spacing_m=float(slant_range[1] - slant_range[0]),
    )
    return orbit, radar_grid, image_shape, look_side, SPEED_OF_LIGHT / center_frequency


def _ecef_distance_m(actual_llh: np.ndarray, expected_llh: np.ndarray) -> np.ndarray:
    actual_xyz = np.vstack(llh_to_ecef(actual_llh[0], actual_llh[1], actual_llh[2]))
    expected_xyz = np.vstack(llh_to_ecef(expected_llh[0], expected_llh[1], expected_llh[2]))
    return np.linalg.norm(actual_xyz - expected_xyz, axis=0)


@pytest.mark.skipif(not ISCE3_REE_RSLC.exists(), reason="ISCE3 real test data not available")
def test_rdr2geo_look_side_with_isce3_ree_rslc_real_data():
    orbit, radar_grid, image_shape, look_side, wavelength = _real_ree_rslc_geometry()
    opposite_side = LookSide.LEFT if look_side is LookSide.RIGHT else LookSide.RIGHT

    for line, pixel in [
        (0, 0),
        (image_shape[0] // 2, image_shape[1] // 2),
        (image_shape[0] - 1, image_shape[1] - 1),
    ]:
        llh = rdr2geo(
            line=line,
            pixel=pixel,
            radar_grid=radar_grid,
            satellite_position=orbit,
            velocity=None,
            doppler=0.0,
            dem=0.0,
            wavelength_m=wavelength,
            look_side=look_side,
        )
        target_ecef = llh_to_ecef(*llh)
        orbit_state = orbit.state_at(radar_grid.line_to_azimuth_time(line))

        assert validate_look_side(target_ecef, orbit_state.position, orbit_state.velocity, look_side)
        assert not validate_look_side(
            target_ecef, orbit_state.position, orbit_state.velocity, opposite_side
        )

        azimuth_time, returned_range = geo2rdr(
            lat=llh[0],
            lon=llh[1],
            height=llh[2],
            radar_grid=radar_grid,
            satellite_position=orbit,
            velocity=None,
            doppler=0.0,
            wavelength_m=wavelength,
            look_side=look_side,
        )

        assert abs(float(radar_grid.azimuth_time_to_line(azimuth_time)) - line) < 1e-3
        assert abs(float(returned_range - radar_grid.pixel_to_slant_range(pixel))) < 1e-3


@pytest.mark.skipif(not ISCE3_REE_RSLC.exists(), reason="ISCE3 real test data not available")
def test_numba_rdr2geo_matches_cpu_with_real_isce3_orbit_interpolator():
    orbit, radar_grid, image_shape, look_side, wavelength = _real_ree_rslc_geometry()
    lines = np.array([0.0, image_shape[0] / 2.0, image_shape[0] - 1.0])
    pixels = np.array([0.0, image_shape[1] / 2.0, image_shape[1] - 1.0])

    expected = rdr2geo(
        line=lines,
        pixel=pixels,
        radar_grid=radar_grid,
        satellite_position=orbit,
        velocity=None,
        doppler=0.0,
        dem=0.0,
        wavelength_m=wavelength,
        look_side=look_side,
    )
    actual = rdr2geo_fast(
        line=lines,
        pixel=pixels,
        radar_grid=radar_grid,
        satellite_position=orbit,
        velocity=None,
        doppler=0.0,
        dem=0.0,
        wavelength_m=wavelength,
        look_side=look_side,
        method="numba",
    )

    assert np.max(_ecef_distance_m(actual, expected)) < 1e-2


@pytest.mark.skipif(not ISCE3_REE_RSLC.exists(), reason="ISCE3 real test data not available")
def test_arrayfire_rdr2geo_matches_cpu_for_real_isce3_static_state():
    from i2sar.accel.arrayfire_backend import arrayfire_available

    if not arrayfire_available():
        pytest.skip("ArrayFire is not available")

    orbit, radar_grid, image_shape, look_side, wavelength = _real_ree_rslc_geometry()
    line = image_shape[0] // 2
    orbit_state = orbit.state_at(radar_grid.line_to_azimuth_time(line))
    static_grid = RadarGrid(
        length=2,
        width=radar_grid.width,
        sensing_start_s=float(radar_grid.line_to_azimuth_time(line) - 1.0 / radar_grid.prf_hz),
        prf_hz=radar_grid.prf_hz,
        starting_range_m=radar_grid.starting_range_m,
        range_pixel_spacing_m=radar_grid.range_pixel_spacing_m,
    )
    lines = np.ones(3, dtype=np.float64)
    pixels = np.array([0.0, image_shape[1] / 2.0, image_shape[1] - 1.0])

    expected = rdr2geo(
        line=lines,
        pixel=pixels,
        radar_grid=static_grid,
        satellite_position=orbit_state.position,
        velocity=orbit_state.velocity,
        doppler=0.0,
        dem=0.0,
        wavelength_m=wavelength,
        look_side=look_side,
    )
    actual = rdr2geo_fast(
        line=lines,
        pixel=pixels,
        radar_grid=static_grid,
        satellite_position=orbit_state.position,
        velocity=orbit_state.velocity,
        doppler=0.0,
        dem=0.0,
        wavelength_m=wavelength,
        look_side=look_side,
        method="arrayfire",
    )

    assert np.max(_ecef_distance_m(actual, expected)) < 1e-2


@pytest.mark.skipif(not ISCE3_REE_RSLC.exists(), reason="ISCE3 real test data not available")
def test_arrayfire_rdr2geo_matches_cpu_with_real_isce3_orbit_interpolator():
    from i2sar.accel.arrayfire_backend import arrayfire_available

    if not arrayfire_available():
        pytest.skip("ArrayFire is not available")

    orbit, radar_grid, image_shape, look_side, wavelength = _real_ree_rslc_geometry()
    lines = np.array([0.0, image_shape[0] / 2.0, image_shape[0] - 1.0])
    pixels = np.array([0.0, image_shape[1] / 2.0, image_shape[1] - 1.0])

    expected = rdr2geo(
        line=lines,
        pixel=pixels,
        radar_grid=radar_grid,
        satellite_position=orbit,
        velocity=None,
        doppler=0.0,
        dem=0.0,
        wavelength_m=wavelength,
        look_side=look_side,
    )
    actual = rdr2geo_fast(
        line=lines,
        pixel=pixels,
        radar_grid=radar_grid,
        satellite_position=orbit,
        velocity=None,
        doppler=0.0,
        dem=0.0,
        wavelength_m=wavelength,
        look_side=look_side,
        method="arrayfire",
    )

    assert np.max(_ecef_distance_m(actual, expected)) < 1e-2


@pytest.mark.skipif(not ISCE3_REE_RSLC.exists(), reason="ISCE3 real test data not available")
def test_numba_geo2rdr_matches_cpu_with_real_isce3_orbit_interpolator():
    orbit, radar_grid, image_shape, look_side, wavelength = _real_ree_rslc_geometry()
    lines = np.array([0.0, image_shape[0] / 2.0, image_shape[0] - 1.0], dtype=np.float64)
    pixels = np.array([0.0, image_shape[1] / 2.0, image_shape[1] - 1.0], dtype=np.float64)
    llh = rdr2geo(
        line=lines,
        pixel=pixels,
        radar_grid=radar_grid,
        satellite_position=orbit,
        velocity=None,
        doppler=0.0,
        dem=0.0,
        wavelength_m=wavelength,
        look_side=look_side,
    )

    cpu = geo2rdr(
        lat=llh[0],
        lon=llh[1],
        height=llh[2],
        radar_grid=radar_grid,
        satellite_position=orbit,
        velocity=None,
        doppler=0.0,
        wavelength_m=wavelength,
        look_side=look_side,
    )
    actual = geo2rdr_fast(
        lat=llh[0],
        lon=llh[1],
        height=llh[2],
        radar_grid=radar_grid,
        satellite_position=orbit,
        velocity=None,
        doppler=0.0,
        wavelength_m=wavelength,
        look_side=look_side,
        method="numba",
    )

    np.testing.assert_allclose(actual, cpu, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not ISCE3_REE_RSLC.exists(), reason="ISCE3 real test data not available")
def test_arrayfire_geo2rdr_matches_cpu_with_real_isce3_orbit_interpolator():
    from i2sar.accel.arrayfire_backend import arrayfire_available

    if not arrayfire_available():
        pytest.skip("ArrayFire is not available")

    orbit, radar_grid, image_shape, look_side, wavelength = _real_ree_rslc_geometry()
    lines = np.array([0.0, image_shape[0] / 2.0, image_shape[0] - 1.0], dtype=np.float64)
    pixels = np.array([0.0, image_shape[1] / 2.0, image_shape[1] - 1.0], dtype=np.float64)
    llh = rdr2geo(
        line=lines,
        pixel=pixels,
        radar_grid=radar_grid,
        satellite_position=orbit,
        velocity=None,
        doppler=0.0,
        dem=0.0,
        wavelength_m=wavelength,
        look_side=look_side,
    )

    cpu = geo2rdr(
        lat=llh[0],
        lon=llh[1],
        height=llh[2],
        radar_grid=radar_grid,
        satellite_position=orbit,
        velocity=None,
        doppler=0.0,
        wavelength_m=wavelength,
        look_side=look_side,
    )
    actual = geo2rdr_fast(
        lat=llh[0],
        lon=llh[1],
        height=llh[2],
        radar_grid=radar_grid,
        satellite_position=orbit,
        velocity=None,
        doppler=0.0,
        wavelength_m=wavelength,
        look_side=look_side,
        method="arrayfire",
    )

    np.testing.assert_allclose(actual, cpu, atol=0.0, rtol=0.0)
