from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass(frozen=True)
class Doppler:
    centroid_hz: float
    toa_poly: np.ndarray | None = None

    def __post_init__(self):
        if self.toa_poly is not None:
            object.__setattr__(self, "toa_poly", np.asarray(self.toa_poly, dtype=np.float64))

    @staticmethod
    def constant(centroid_hz: float) -> "Doppler":
        return Doppler(centroid_hz=centroid_hz, toa_poly=None)

    @staticmethod
    def polynomial(centroid_hz: float, toa_poly: np.ndarray) -> "Doppler":
        return Doppler(centroid_hz=centroid_hz, toa_poly=toa_poly)

    def evaluate(self, slant_range_m: float, wavelength_m: float = 0.0565642) -> float:
        if self.toa_poly is None:
            return self.centroid_hz
        norm_range = slant_range_m / 1e6
        poly_val = float(np.polyval(self.toa_poly, norm_range))
        return self.centroid_hz + poly_val


def read_doppler_from_hdf5(h5, group: str = "doppler") -> Doppler:
    import json

    doppler_group = h5[group]
    json_str = doppler_group.attrs.get("json", "{}")
    payload = json.loads(json_str)

    centroid = float(payload.get("centroid_hz", 0.0))
    toa_poly = payload.get("toa_poly")
    if toa_poly is not None:
        toa_poly = np.array(toa_poly, dtype=np.float64)

    return Doppler(centroid_hz=centroid, toa_poly=toa_poly)


def write_doppler_to_hdf5(h5, doppler: Doppler, group: str = "doppler") -> None:
    import json

    payload = {
        "centroid_hz": float(doppler.centroid_hz),
    }
    if doppler.toa_poly is not None:
        payload["toa_poly"] = doppler.toa_poly.tolist()

    h5[group].attrs["json"] = json.dumps(payload)
