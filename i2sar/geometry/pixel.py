from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Pixel:
    range: float
    dopfact: float
    bin: int = 0

    @staticmethod
    def from_range_doppler_wavelength(
        slant_range: float,
        doppler: float,
        wavelength: float,
        velocity_magnitude: float,
        bin: int = 0,
    ) -> "Pixel":
        dopfact = 0.5 * wavelength * doppler * slant_range / velocity_magnitude
        return Pixel(range=slant_range, dopfact=dopfact, bin=bin)
