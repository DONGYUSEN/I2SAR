from __future__ import annotations

import numpy as np

from i2sar.dem import DEMInterpolator
from i2sar.rtc.rtc import (
    _build_utm_geocode_plan,
    _geocode_to_utm,
    _geocode_with_plan,
    _radiometric_calibration,
)


def test_geocode_plan_matches_legacy_geocode_to_utm():
    lat = np.array([[0.0, 0.0], [0.001, 0.001]], dtype=np.float64)
    lon = np.array([[3.0, 3.001], [3.0, 3.001]], dtype=np.float64)
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dem = DEMInterpolator(np.zeros((2, 2), dtype=np.float32), [3.0, 0.001, 0.0, 0.001, 0.0, -0.001])

    legacy, legacy_mask, legacy_gt, legacy_min_x, legacy_max_y = _geocode_to_utm(
        data=data,
        lat=lat,
        lon=lon,
        dem_interpolator=dem,
        output_resolution=50.0,
        epsg=32631,
    )

    plan = _build_utm_geocode_plan(
        lat=lat,
        lon=lon,
        output_resolution=50.0,
        epsg=32631,
    )
    planned, planned_mask = _geocode_with_plan(data, plan)

    np.testing.assert_array_equal(np.isnan(planned), np.isnan(legacy))
    np.testing.assert_allclose(np.nan_to_num(planned), np.nan_to_num(legacy), atol=0.0)
    np.testing.assert_array_equal(planned_mask, legacy_mask)
    assert plan.geotransform == legacy_gt
    assert plan.min_x == legacy_min_x
    assert plan.max_y == legacy_max_y


def test_geocode_plan_reuses_same_mapping_for_multiple_bands():
    lat = np.array([[0.0, 0.0], [0.001, 0.001]], dtype=np.float64)
    lon = np.array([[3.0, 3.001], [3.0, 3.001]], dtype=np.float64)
    band1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    band2 = band1 + 10.0

    plan = _build_utm_geocode_plan(lat, lon, output_resolution=50.0, epsg=32631)
    geocoded1, mask1 = _geocode_with_plan(band1, plan)
    geocoded2, mask2 = _geocode_with_plan(band2, plan)

    np.testing.assert_array_equal(mask1, mask2)
    np.testing.assert_array_equal(np.isfinite(geocoded1), np.isfinite(geocoded2))
    np.testing.assert_allclose(geocoded2[np.isfinite(geocoded2)], geocoded1[np.isfinite(geocoded1)] + 10.0)


def test_geocode_plan_keeps_last_source_when_pixels_collide():
    lat = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    lon = np.array([[3.0, 3.0], [3.0, 3.0]], dtype=np.float64)
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    plan = _build_utm_geocode_plan(lat, lon, output_resolution=1000.0, epsg=32631)
    geocoded, valid_mask = _geocode_with_plan(data, plan)

    assert np.count_nonzero(valid_mask) == 1
    assert geocoded[valid_mask][0] == 4.0


def test_radiometric_calibration_complex_and_structured_inputs_match():
    complex_slc = np.array([[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]], dtype=np.complex64)
    structured_slc = np.zeros(complex_slc.shape, dtype=[("real", np.float32), ("imag", np.float32)])
    structured_slc["real"] = complex_slc.real
    structured_slc["imag"] = complex_slc.imag
    incidence = np.full(complex_slc.shape, 0.5, dtype=np.float64)

    complex_result = _radiometric_calibration(complex_slc, incidence, 10.0, 20.0, 1000.0, 0.03)
    structured_result = _radiometric_calibration(structured_slc, incidence, 10.0, 20.0, 1000.0, 0.03)

    for key in ("sigma0", "gamma0", "beta0", "intensity"):
        np.testing.assert_allclose(structured_result[key], complex_result[key], rtol=1e-6)
        assert structured_result[key].dtype == np.float32
