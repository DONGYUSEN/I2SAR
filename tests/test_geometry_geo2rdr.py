"""Tests for geo2rdr (geo to radar) transformation."""

from __future__ import annotations

import numpy as np

from i2sar.geometry import RadarGrid, WGS84, llh_to_ecef
from i2sar.geometry.geo2rdr import geo2rdr, compute_geo2rdr_mapping, geo2rdr_to_pixel


class TestGeo2RdrBasic:
    def test_geo2rdr_returns_aztime_and_slant_range(self):
        ellipsoid = WGS84

        sat_height_m = 500_000.0
        sat_position = np.array([0.0, 0.0, sat_height_m + ellipsoid.a])
        velocity = np.array([0.0, 3000.0, 0.0])

        radar_grid = RadarGrid(
            length=10,
            width=5000,
            sensing_start_s=0.0,
            prf_hz=1000.0,
            starting_range_m=3_000_000.0,
            range_pixel_spacing_m=200.0,
        )

        target_lat = 60.0
        target_lon = 0.0
        target_h = 0.0

        result = geo2rdr(
            lat=target_lat,
            lon=target_lon,
            height=target_h,
            radar_grid=radar_grid,
            satellite_position=sat_position,
            velocity=velocity,
            doppler=0.0,
            ellipsoid=ellipsoid,
        )

        aztime, slant_range = result
        assert np.isfinite(aztime)
        assert np.isfinite(slant_range)
        assert radar_grid.start_time <= aztime <= radar_grid.end_time
        assert slant_range > 0

    def test_geo2rdr_vectorized_inputs(self):
        ellipsoid = WGS84

        sat_position = np.array([0.0, 0.0, WGS84.a + 500_000.0])
        velocity = np.array([0.0, 3000.0, 0.0])

        radar_grid = RadarGrid(
            length=10,
            width=5,
            sensing_start_s=0.0,
            prf_hz=1000.0,
            starting_range_m=400_000.0,
            range_pixel_spacing_m=20.0,
        )

        lats = np.array([45.0, 45.0, 45.0])
        lons = np.array([0.0, 0.1, 0.2])
        heights = np.array([0.0, 0.0, 0.0])

        result = geo2rdr(
            lat=lats,
            lon=lons,
            height=heights,
            radar_grid=radar_grid,
            satellite_position=sat_position,
            velocity=velocity,
            doppler=0.0,
            ellipsoid=ellipsoid,
        )

        assert result.shape == (2, 3)
        assert np.all(np.isfinite(result))

    def test_compute_geo2rdr_mapping_returns_grid(self):
        ellipsoid = WGS84

        sat_position = np.array([0.0, 0.0, WGS84.a + 500_000.0])
        velocity = np.array([0.0, 3000.0, 0.0])

        radar_grid = RadarGrid(
            length=5,
            width=4,
            sensing_start_s=0.0,
            prf_hz=1000.0,
            starting_range_m=400_000.0,
            range_pixel_spacing_m=20.0,
        )

        lats_input = np.array([45.0, 45.1, 45.2, 45.3, 45.4])
        lons_input = np.array([0.0, 0.05, 0.1, 0.15])

        aztimes, ranges = compute_geo2rdr_mapping(
            lats=lats_input,
            lons=lons_input,
            radar_grid=radar_grid,
            satellite_position=sat_position,
            velocity=velocity,
            doppler=0.0,
            ellipsoid=ellipsoid,
        )

        assert aztimes.shape == (5, 4)
        assert ranges.shape == (5, 4)
        assert np.all(np.isfinite(aztimes))
        assert np.all(np.isfinite(ranges))

    def test_geo2rdr_to_pixel_conversion(self):
        radar_grid = RadarGrid(
            length=10,
            width=100,
            sensing_start_s=0.0,
            prf_hz=1000.0,
            starting_range_m=800_000.0,
            range_pixel_spacing_m=100.0,
        )

        aztime = 0.005
        slant_range = 800_500.0

        line, pixel = geo2rdr_to_pixel(aztime, slant_range, radar_grid)

        assert np.isclose(line, 5.0)
        assert np.isclose(pixel, 5.0)


class TestGeo2RdrRoundTrip:
    def test_geo2rdr_then_rdr2geo_roundtrip(self):
        from i2sar.geometry.rdr2geo import rdr2geo

        ellipsoid = WGS84

        sat_height_m = 500_000.0
        sat_position = np.array([0.0, 0.0, sat_height_m + ellipsoid.a])
        velocity = np.array([3000.0, 0.0, 0.0])

        radar_grid = RadarGrid(
            length=10,
            width=100,
            sensing_start_s=0.0,
            prf_hz=1000.0,
            starting_range_m=800_000.0,
            range_pixel_spacing_m=100.0,
        )

        dem_elevation = np.zeros((10, 100))

        original_line = 5
        original_pixel = 50

        geo_result = rdr2geo(
            line=original_line,
            pixel=original_pixel,
            radar_grid=radar_grid,
            satellite_position=sat_position,
            velocity=velocity,
            doppler=0.0,
            dem_elevation=dem_elevation,
            ellipsoid=ellipsoid,
        )

        aztime, slant_range = geo2rdr(
            lat=geo_result[0],
            lon=geo_result[1],
            height=geo_result[2],
            radar_grid=radar_grid,
            satellite_position=sat_position,
            velocity=velocity,
            doppler=0.0,
            ellipsoid=ellipsoid,
        )

        recovered_line, recovered_pixel = geo2rdr_to_pixel(aztime, slant_range, radar_grid)

        np.testing.assert_allclose(recovered_line, original_line, atol=0.5)
        np.testing.assert_allclose(recovered_pixel, original_pixel, atol=2.0)


class TestGeo2RdrWithOrbitInterpolator:
    def test_geo2rdr_uses_orbit_interpolator(self):
        from i2sar.orbit import OrbitInterpolator, OrbitState

        ellipsoid = WGS84

        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([
            [0.0, 0.0, WGS84.a + 500_000.0],
            [0.0, 3000.0, WGS84.a + 500_000.0],
            [0.0, 6000.0, WGS84.a + 500_000.0],
        ])
        velocities = np.array([
            [0.0, 3000.0, 0.0],
            [0.0, 3000.0, 0.0],
            [0.0, 3000.0, 0.0],
        ])

        orbit_interpolator = OrbitInterpolator(
            time=times,
            position=positions,
            velocity=velocities,
        )

        radar_grid = RadarGrid(
            length=5,
            width=3,
            sensing_start_s=0.5,
            prf_hz=1.0,
            starting_range_m=400_000.0,
            range_pixel_spacing_m=20.0,
        )

        result = geo2rdr(
            lat=45.0,
            lon=0.0,
            height=0.0,
            radar_grid=radar_grid,
            satellite_position=orbit_interpolator,
            velocity=None,
            doppler=0.0,
            ellipsoid=ellipsoid,
        )

        aztime, slant_range = result
        assert np.isfinite(aztime)
        assert np.isfinite(slant_range)
