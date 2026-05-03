"""Tests for rdr2geo (radar to geo) transformation."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from i2sar.geometry import RadarGrid, WGS84, ecef_to_llh, llh_to_ecef
from i2sar.geometry.rdr2geo import rdr2geo, compute_rdr2geo_mapping


class TestRdr2GeoBasic:
    def test_rdr2geo_returns_same_location_for_zero_doppler_at_nadir(self):
        ellipsoid = WGS84

        target_llh = (45.0, 0.0, 0.0)
        target_ecef = llh_to_ecef(*target_llh, ellipsoid=ellipsoid)

        sat_height_m = 500_000.0
        prime_vertical = ellipsoid.a / np.sqrt(1.0 - ellipsoid.e2 * np.sin(np.deg2rad(45))**2)
        sat_x = (prime_vertical + sat_height_m) * np.cos(np.deg2rad(45)) * np.cos(np.deg2rad(0))
        sat_y = (prime_vertical + sat_height_m) * np.cos(np.deg2rad(45)) * np.sin(np.deg2rad(0))
        sat_z = (prime_vertical * (1.0 - ellipsoid.e2) + sat_height_m) * np.sin(np.deg2rad(45))
        satellite_position = np.array([sat_x, sat_y, sat_z])

        velocity = np.array([0.0, 3000.0, 0.0])

        length, width = 1, 1
        sensing_start_s = 0.0
        prf_hz = 2000.0
        starting_range_m = sat_height_m - 1000.0
        range_spacing_m = 10.0

        radar_grid = RadarGrid(
            length=length,
            width=width,
            sensing_start_s=sensing_start_s,
            prf_hz=prf_hz,
            starting_range_m=starting_range_m,
            range_pixel_spacing_m=range_spacing_m,
        )

        doppler = 0.0

        dem_elevation = 0.0

        result_llh = rdr2geo(
            line=0,
            pixel=0,
            radar_grid=radar_grid,
            satellite_position=satellite_position,
            velocity=velocity,
            doppler=doppler,
            dem_elevation=dem_elevation,
            ellipsoid=ellipsoid,
        )

        recovered_lat, recovered_lon, recovered_h = result_llh
        np.testing.assert_allclose(recovered_lat, target_llh[0], atol=0.1)
        np.testing.assert_allclose(recovered_lon, target_llh[1], atol=0.1)
        assert np.abs(recovered_h) < 2000.0

    def test_rdr2geo_vectorized_lines(self):
        ellipsoid = WGS84

        sat_height_m = 500_000.0
        sat_position = np.array([0.0, 0.0, sat_height_m + ellipsoid.a])
        velocity = np.array([0.0, 3000.0, 0.0])
        doppler = 0.0

        radar_grid = RadarGrid(
            length=10,
            width=5,
            sensing_start_s=0.0,
            prf_hz=1000.0,
            starting_range_m=600_000.0,
            range_pixel_spacing_m=10.0,
        )

        dem_elevation = np.zeros((10, 5))

        lines = np.array([0, 5, 9])
        pixels = np.array([2, 2, 2])

        result = rdr2geo(
            line=lines,
            pixel=pixels,
            radar_grid=radar_grid,
            satellite_position=sat_position,
            velocity=velocity,
            doppler=doppler,
            dem_elevation=dem_elevation,
            ellipsoid=ellipsoid,
        )

        assert result.shape == (3, 3)
        assert np.all(np.isfinite(result))

    def test_compute_rdr2geo_mapping_returns_grid(self):
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

        dem_elevation = np.zeros((5, 4))

        lats, lons, heights = compute_rdr2geo_mapping(
            radar_grid=radar_grid,
            satellite_position=sat_position,
            velocity=velocity,
            doppler=0.0,
            dem_elevation=dem_elevation,
            ellipsoid=ellipsoid,
        )

        assert lats.shape == (5, 4)
        assert lons.shape == (5, 4)
        assert heights.shape == (5, 4)
        assert np.all(np.isfinite(lats))
        assert np.all(np.isfinite(lons))
        assert np.all(np.isfinite(heights))


class TestRdr2GeoWithOrbitInterpolator:
    def test_rdr2geo_uses_orbit_interpolator_for_satellite_position(self):
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

        orbit_state = OrbitState(position=positions, velocity=velocities)
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

        dem_elevation = np.zeros((5, 3))

        lats, lons, heights = compute_rdr2geo_mapping(
            radar_grid=radar_grid,
            satellite_position=orbit_interpolator,
            velocity=None,
            doppler=0.0,
            dem_elevation=dem_elevation,
            ellipsoid=ellipsoid,
        )

        assert lats.shape == (5, 3)
        assert np.all(np.isfinite(lats))


class TestRdr2GeoWithDoppler:
    def test_rdr2geo_accepts_nonzero_doppler(self):
        ellipsoid = WGS84

        sat_height_m = 500_000.0
        sat_position = np.array([0.0, 0.0, sat_height_m + ellipsoid.a])
        velocity = np.array([0.0, 3000.0, 0.0])

        radar_grid = RadarGrid(
            length=10,
            width=5,
            sensing_start_s=0.0,
            prf_hz=1000.0,
            starting_range_m=400_000.0,
            range_pixel_spacing_m=20.0,
        )

        dem_elevation = np.zeros((10, 5))

        result_doppler_0 = compute_rdr2geo_mapping(
            radar_grid=radar_grid,
            satellite_position=sat_position,
            velocity=velocity,
            doppler=0.0,
            dem_elevation=dem_elevation,
            ellipsoid=ellipsoid,
        )

        result_doppler_nonzero = compute_rdr2geo_mapping(
            radar_grid=radar_grid,
            satellite_position=sat_position,
            velocity=velocity,
            doppler=500.0,
            dem_elevation=dem_elevation,
            ellipsoid=ellipsoid,
        )

        assert np.all(np.isfinite(result_doppler_0[0]))
        assert np.all(np.isfinite(result_doppler_nonzero[0]))


class TestRdr2GeoWithDem:
    def test_rdr2geo_with_varying_elevation(self):
        ellipsoid = WGS84

        sat_position = np.array([0.0, 0.0, WGS84.a + 500_000.0])
        velocity = np.array([0.0, 3000.0, 0.0])

        radar_grid = RadarGrid(
            length=5,
            width=5,
            sensing_start_s=0.0,
            prf_hz=1000.0,
            starting_range_m=600_000.0,
            range_pixel_spacing_m=20.0,
        )

        dem_zero = np.zeros((5, 5))
        dem_mountain = np.full((5, 5), 1000.0)

        lats_0, lons_0, heights_0 = compute_rdr2geo_mapping(
            radar_grid=radar_grid,
            satellite_position=sat_position,
            velocity=velocity,
            doppler=0.0,
            dem_elevation=dem_zero,
            ellipsoid=ellipsoid,
        )

        lats_m, lons_m, heights_m = compute_rdr2geo_mapping(
            radar_grid=radar_grid,
            satellite_position=sat_position,
            velocity=velocity,
            doppler=0.0,
            dem_elevation=dem_mountain,
            ellipsoid=ellipsoid,
        )

        assert not np.allclose(heights_0, heights_m, atol=0.1)
