from __future__ import annotations

import numpy as np

from i2sar.core.enums import LookSide


def check_look_side(
    rng_vec: np.ndarray,
    velocity: np.ndarray,
    platform_position: np.ndarray,
    look_side: LookSide,
) -> bool:
    cross_product = np.cross(rng_vec, velocity)
    dot_product = np.dot(cross_product, platform_position)
    is_right = look_side == LookSide.RIGHT
    cross_positive = bool(dot_product > 0)
    return is_right == cross_positive


def validate_look_side(
    target_ecef: np.ndarray,
    satellite_position: np.ndarray,
    velocity: np.ndarray,
    look_side: LookSide,
) -> bool:
    rng_vec = target_ecef - satellite_position
    return check_look_side(rng_vec, velocity, satellite_position, look_side)
