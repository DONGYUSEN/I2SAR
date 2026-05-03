from __future__ import annotations

import numpy as np
import pytest

from i2sar.core.enums import LookSide
from i2sar.geometry import RadarGrid, WGS84, llh_to_ecef
from i2sar.geometry.look_side import check_look_side, validate_look_side
from i2sar.geometry.rdr2geo import rdr2geo


class TestCheckLookSide:
    def test_right_look_with_positive_cross(self):
        rng_vec = np.array([1.0, 0.0, 0.0])
        velocity = np.array([0.0, 1.0, 0.0])
        platform_position = np.array([0.0, 0.0, 1.0])

        result = check_look_side(rng_vec, velocity, platform_position, LookSide.RIGHT)
        assert result == True

    def test_right_look_with_negative_cross(self):
        rng_vec = np.array([1.0, 0.0, 0.0])
        velocity = np.array([0.0, -1.0, 0.0])
        platform_position = np.array([0.0, 0.0, 1.0])

        result = check_look_side(rng_vec, velocity, platform_position, LookSide.RIGHT)
        assert result == False

    def test_left_look_with_negative_cross(self):
        rng_vec = np.array([1.0, 0.0, 0.0])
        velocity = np.array([0.0, -1.0, 0.0])
        platform_position = np.array([0.0, 0.0, 1.0])

        result = check_look_side(rng_vec, velocity, platform_position, LookSide.LEFT)
        assert result is True

    def test_left_look_with_positive_cross(self):
        rng_vec = np.array([1.0, 0.0, 0.0])
        velocity = np.array([0.0, 1.0, 0.0])
        platform_position = np.array([0.0, 0.0, 1.0])

        result = check_look_side(rng_vec, velocity, platform_position, LookSide.LEFT)
        assert result == False


class TestValidateLookSide:
    def test_valid_right_look(self):
        target_ecef = np.array([1.0, 0.0, 0.0])
        satellite_position = np.array([0.0, 0.0, 1.0])
        velocity = np.array([0.0, 1.0, 0.0])

        result = validate_look_side(
            target_ecef, satellite_position, velocity, LookSide.RIGHT
        )
        assert result is True

    def test_invalid_right_look(self):
        target_ecef = np.array([1.0, 0.0, 0.0])
        satellite_position = np.array([0.0, 0.0, 1.0])
        velocity = np.array([0.0, -1.0, 0.0])

        result = validate_look_side(
            target_ecef, satellite_position, velocity, LookSide.RIGHT
        )
        assert result == False


class TestLookSideSign:
    def test_left_sign(self):
        assert LookSide.LEFT.sign == -1

    def test_right_sign(self):
        assert LookSide.RIGHT.sign == 1


class TestRdr2GeoLookSideConsistency:
    def test_rdr2geo_output_matches_requested_look_side(self):
        satellite_position = np.array([WGS84.a + 500_000.0, 0.0, 0.0])
        velocity = np.array([0.0, 7_500.0, 0.0])
        radar_grid = RadarGrid(
            length=1,
            width=1,
            sensing_start_s=0.0,
            prf_hz=1000.0,
            starting_range_m=850_000.0,
            range_pixel_spacing_m=20.0,
        )

        for look_side in (LookSide.LEFT, LookSide.RIGHT):
            llh = rdr2geo(
                line=0,
                pixel=0,
                radar_grid=radar_grid,
                satellite_position=satellite_position,
                velocity=velocity,
                doppler=0.0,
                dem=0.0,
                look_side=look_side,
            )
            target_ecef = llh_to_ecef(*llh)

            assert validate_look_side(
                target_ecef,
                satellite_position,
                velocity,
                look_side,
            )

    def test_numba_rdr2geo_respects_requested_look_side(self):
        from i2sar.geometry.accelerated_geometry import rdr2geo_fast

        satellite_position = np.array([WGS84.a + 500_000.0, 0.0, 0.0])
        velocity = np.array([0.0, 7_500.0, 0.0])
        radar_grid = RadarGrid(
            length=1,
            width=1,
            sensing_start_s=0.0,
            prf_hz=1000.0,
            starting_range_m=850_000.0,
            range_pixel_spacing_m=20.0,
        )

        left = rdr2geo_fast(
            line=np.array([0.0]),
            pixel=np.array([0.0]),
            radar_grid=radar_grid,
            satellite_position=satellite_position,
            velocity=velocity,
            doppler=0.0,
            dem=0.0,
            look_side=LookSide.LEFT,
            method="numba",
        )
        right = rdr2geo_fast(
            line=np.array([0.0]),
            pixel=np.array([0.0]),
            radar_grid=radar_grid,
            satellite_position=satellite_position,
            velocity=velocity,
            doppler=0.0,
            dem=0.0,
            look_side=LookSide.RIGHT,
            method="numba",
        )

        assert validate_look_side(llh_to_ecef(*left), satellite_position, velocity, LookSide.LEFT)
        assert validate_look_side(llh_to_ecef(*right), satellite_position, velocity, LookSide.RIGHT)
        assert not np.allclose(left[:2], right[:2])
