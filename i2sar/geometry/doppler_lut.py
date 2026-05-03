from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DopplerLUT2d:
    range_axis: np.ndarray
    azimuth_axis: np.ndarray
    data: np.ndarray

    def __post_init__(self):
        self.range_axis = np.asarray(self.range_axis, dtype=np.float64)
        self.azimuth_axis = np.asarray(self.azimuth_axis, dtype=np.float64)
        self.data = np.asarray(self.data, dtype=np.float64)

        if self.range_axis.ndim != 1:
            raise ValueError("range_axis must be 1D")
        if self.azimuth_axis.ndim != 1:
            raise ValueError("azimuth_axis must be 1D")
        if self.data.shape != (self.azimuth_axis.size, self.range_axis.size):
            raise ValueError(
                f"data shape {self.data.shape} must match (azimuth_size={self.azimuth_axis.size}, range_size={self.range_axis.size})"
            )

    @staticmethod
    def from_centroid_and_slopes(
        centroid_hz: float,
        range_slope: float,
        azimuth_slope: float,
        range_axis: np.ndarray,
        azimuth_axis: np.ndarray,
    ) -> "DopplerLUT2d":
        rg, az = np.meshgrid(range_axis, azimuth_axis, indexing="xy")
        data = centroid_hz + range_slope * (rg - rg[0, 0]) + azimuth_slope * (az - az[0, 0])
        return DopplerLUT2d(
            range_axis=range_axis,
            azimuth_axis=azimuth_axis,
            data=data,
        )

    def eval(self, azimuth_time_s: float, range_m: float) -> float:
        azimuth_time_s = np.asarray(azimuth_time_s)
        range_m = np.asarray(range_m)
        
        if azimuth_time_s.ndim == 0 and range_m.ndim == 0:
            return float(self._eval_single(float(azimuth_time_s), float(range_m)))
        
        return self._eval_batch(azimuth_time_s, range_m)
    
    def _eval_single(self, azimuth_time_s: float, range_m: float) -> float:
        if (
            azimuth_time_s < self.azimuth_axis[0]
            or azimuth_time_s > self.azimuth_axis[-1]
            or range_m < self.range_axis[0]
            or range_m > self.range_axis[-1]
        ):
            return self._extrapolate(azimuth_time_s, range_m)

        i_a = np.searchsorted(self.azimuth_axis, azimuth_time_s) - 1
        i_r = np.searchsorted(self.range_axis, range_m) - 1

        i_r = np.clip(i_r, 0, len(self.range_axis) - 2)
        i_a = np.clip(i_a, 0, len(self.azimuth_axis) - 2)

        a0, a1 = self.azimuth_axis[i_a], self.azimuth_axis[i_a + 1]
        r0, r1 = self.range_axis[i_r], self.range_axis[i_r + 1]

        d00 = self.data[i_a, i_r]
        d01 = self.data[i_a, i_r + 1]
        d10 = self.data[i_a + 1, i_r]
        d11 = self.data[i_a + 1, i_r + 1]

        wa = (azimuth_time_s - a0) / (a1 - a0) if a1 != a0 else 0.0
        wr = (range_m - r0) / (r1 - r0) if r1 != r0 else 0.0

        result = (1 - wr) * (1 - wa) * d00 + wr * (1 - wa) * d01 + (1 - wr) * wa * d10 + wr * wa * d11
        return result
    
    def _eval_batch(self, azimuth_time_s: np.ndarray, range_m: np.ndarray) -> np.ndarray:
        n_points = len(azimuth_time_s)
        result = np.zeros(n_points, dtype=np.float64)
        
        inside_mask = (
            (azimuth_time_s >= self.azimuth_axis[0])
            & (azimuth_time_s <= self.azimuth_axis[-1])
            & (range_m >= self.range_axis[0])
            & (range_m <= self.range_axis[-1])
        )
        
        outside_mask = ~inside_mask
        
        if np.any(inside_mask):
            i_a = np.searchsorted(self.azimuth_axis, azimuth_time_s[inside_mask]) - 1
            i_r = np.searchsorted(self.range_axis, range_m[inside_mask]) - 1

            i_r = np.clip(i_r, 0, len(self.range_axis) - 2)
            i_a = np.clip(i_a, 0, len(self.azimuth_axis) - 2)

            a0 = self.azimuth_axis[i_a]
            a1 = self.azimuth_axis[i_a + 1]
            r0 = self.range_axis[i_r]
            r1 = self.range_axis[i_r + 1]

            d00 = self.data[i_a, i_r]
            d01 = self.data[i_a, i_r + 1]
            d10 = self.data[i_a + 1, i_r]
            d11 = self.data[i_a + 1, i_r + 1]

            wa = (azimuth_time_s[inside_mask] - a0) / np.where(a1 != a0, a1 - a0, 1.0)
            wr = (range_m[inside_mask] - r0) / np.where(r1 != r0, r1 - r0, 1.0)

            result[inside_mask] = (
                (1 - wr) * (1 - wa) * d00 
                + wr * (1 - wa) * d01 
                + (1 - wr) * wa * d10 
                + wr * wa * d11
            )
        
        if np.any(outside_mask):
            for i in np.where(outside_mask)[0]:
                result[i] = self._extrapolate(azimuth_time_s[i], range_m[i])
        
        return result

    def _extrapolate(self, azimuth_time_s: float, range_m: float) -> float:
        r0 = self.range_axis[0]
        r1 = self.range_axis[-1]
        a0 = self.azimuth_axis[0]
        a1 = self.azimuth_axis[-1]

        if range_m < r0:
            if azimuth_time_s < a0:
                return self.data[0, 0]
            elif azimuth_time_s > a1:
                return self.data[-1, 0]
            else:
                i_a = np.searchsorted(self.azimuth_axis, azimuth_time_s) - 1
                i_a = np.clip(i_a, 0, len(self.azimuth_axis) - 2)
                a_lo, a_hi = self.azimuth_axis[i_a], self.azimuth_axis[i_a + 1]
                w = (azimuth_time_s - a_lo) / (a_hi - a_lo) if a_hi != a_lo else 0.0
                return float((1 - w) * self.data[i_a, 0] + w * self.data[i_a + 1, 0])
        elif range_m > r1:
            if azimuth_time_s < a0:
                return self.data[0, -1]
            elif azimuth_time_s > a1:
                return self.data[-1, -1]
            else:
                i_a = np.searchsorted(self.azimuth_axis, azimuth_time_s) - 1
                i_a = np.clip(i_a, 0, len(self.azimuth_axis) - 2)
                a_lo, a_hi = self.azimuth_axis[i_a], self.azimuth_axis[i_a + 1]
                w = (azimuth_time_s - a_lo) / (a_hi - a_lo) if a_hi != a_lo else 0.0
                return float((1 - w) * self.data[i_a, -1] + w * self.data[i_a + 1, -1])
        else:
            i_r = np.searchsorted(self.range_axis, range_m) - 1
            i_r = np.clip(i_r, 0, len(self.range_axis) - 2)
            if azimuth_time_s < a0:
                r_lo, r_hi = self.range_axis[i_r], self.range_axis[i_r + 1]
                w = (range_m - r_lo) / (r_hi - r_lo) if r_hi != r_lo else 0.0
                return float((1 - w) * self.data[0, i_r] + w * self.data[0, i_r + 1])
            else:
                r_lo, r_hi = self.range_axis[i_r], self.range_axis[i_r + 1]
                w = (range_m - r_lo) / (r_hi - r_lo) if r_hi != r_lo else 0.0
                return float((1 - w) * self.data[-1, i_r] + w * self.data[-1, i_r + 1])

    def validate(self) -> bool:
        if self.range_axis.size < 2 or self.azimuth_axis.size < 2:
            return False
        if not np.all(np.diff(self.range_axis) > 0):
            return False
        if not np.all(np.diff(self.azimuth_axis) > 0):
            return False
        return True
