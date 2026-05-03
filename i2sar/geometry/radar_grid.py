from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SPEED_OF_LIGHT = 299792458.0


@dataclass(frozen=True)
class RadarGrid:
    length: int
    width: int
    sensing_start_s: float
    prf_hz: float
    starting_range_m: float
    range_pixel_spacing_m: float

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "RadarGrid":
        length = int(_first(payload, "length", "numberOfRows"))
        width = int(_first(payload, "width", "numberOfColumns"))
        prf_hz = float(_first(payload, "prf", "azimuthFrequency"))
        sensing_start_s = float(_first(payload, "sensingStartSeconds", "sensing_start_s", default=0.0))
        starting_range_m = _first(payload, "startingRange", "starting_range_m", default=None)
        if starting_range_m is None:
            range_time = float(_first(payload, "rangeTimeFirstPixel", "slantRangeTime", "t0"))
            starting_range_m = SPEED_OF_LIGHT * range_time / 2.0
        range_pixel_spacing_m = float(_first(payload, "rangePixelSpacing", "columnSpacing", "range_pixel_spacing_m"))
        return cls(
            length=length,
            width=width,
            sensing_start_s=sensing_start_s,
            prf_hz=prf_hz,
            starting_range_m=float(starting_range_m),
            range_pixel_spacing_m=range_pixel_spacing_m,
        )

    @property
    def start_time(self) -> float:
        return self.sensing_start_s

    @property
    def end_time(self) -> float:
        return self.sensing_start_s + self.length / self.prf_hz

    @property
    def start_range(self) -> float:
        return self.starting_range_m

    @property
    def end_range(self) -> float:
        return self.starting_range_m + self.width * self.range_pixel_spacing_m

    @property
    def number_of_seconds(self) -> float:
        return self.length / self.prf_hz

    def line_to_azimuth_time(self, line):
        return self.sensing_start_s + np.asarray(line, dtype=np.float64) / self.prf_hz

    def azimuth_time_to_line(self, azimuth_time_s):
        return (np.asarray(azimuth_time_s, dtype=np.float64) - self.sensing_start_s) * self.prf_hz

    def pixel_to_slant_range(self, pixel):
        return self.starting_range_m + np.asarray(pixel, dtype=np.float64) * self.range_pixel_spacing_m

    def slant_range_to_pixel(self, slant_range_m):
        return (np.asarray(slant_range_m, dtype=np.float64) - self.starting_range_m) / self.range_pixel_spacing_m


def _first(payload: dict[str, Any], *names: str, default: Any = ...):
    for name in names:
        if name in payload:
            return payload[name]
    if default is not ...:
        return default
    raise KeyError(f"missing radar grid field; expected one of {', '.join(names)}")


def read_radar_grid(scene_h5_path: str | Path) -> RadarGrid:
    with h5py.File(scene_h5_path, "r") as h5:
        raw = h5["radar_grid"].attrs["json"]
        payload = json.loads(raw)
    return RadarGrid.from_mapping(payload)
