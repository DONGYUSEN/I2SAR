from __future__ import annotations

import numpy as np
import pytest

from i2sar.geometry.pixel import Pixel


class TestPixel:
    def test_create_pixel(self):
        pixel = Pixel(range=1000.0, dopfact=50.0, bin=0)
        assert pixel.range == 1000.0
        assert pixel.dopfact == 50.0
        assert pixel.bin == 0

    def test_from_range_doppler_wavelength(self):
        slant_range = 1000.0
        doppler = 100.0
        wavelength = 0.0565642
        velocity_magnitude = 3000.0

        pixel = Pixel.from_range_doppler_wavelength(
            slant_range, doppler, wavelength, velocity_magnitude
        )

        expected_dopfact = 0.5 * wavelength * doppler * slant_range / velocity_magnitude
        assert np.isclose(pixel.range, slant_range)
        assert np.isclose(pixel.dopfact, expected_dopfact)
        assert pixel.bin == 0
