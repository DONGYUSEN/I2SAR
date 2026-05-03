from __future__ import annotations

import numpy as np
import pytest

from i2sar.core.enums import LookSide
from i2sar.geometry.look_side import check_look_side, validate_look_side


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
        assert LookSide.LEFT.sign == 1

    def test_right_sign(self):
        assert LookSide.RIGHT.sign == -1
