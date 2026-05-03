from __future__ import annotations

from dataclasses import dataclass
from i2sar.core.enums import StrEnum

import numpy as np


class InterpMethod(StrEnum):
    SINC = "sinc"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    NEAREST = "nearest"
    BIQUINTIC = "biquintic"


@dataclass
class DEMInterpolator:
    elevation: np.ndarray
    geotransform: list[float]
    epsg: int = 4326
    ref_height: float = 0.0
    interp_method: InterpMethod = InterpMethod.BILINEAR

    def __post_init__(self):
        self.elevation = np.asarray(self.elevation, dtype=np.float32)
        self.geotransform = list(self.geotransform)

    @staticmethod
    def from_hdf5(h5_or_path, group: str = "dem") -> "DEMInterpolator":
        import h5py

        if hasattr(h5_or_path, "require_group"):
            dem_group = h5_or_path[group]
            elevation = dem_group["elevation"][...]
            geotransform = list(dem_group.attrs.get("geotransform", []))
            epsg = int(dem_group.attrs.get("epsg", 4326))
            ref_height = float(dem_group.attrs.get("ref_height", 0.0))
        else:
            with h5py.File(h5_or_path, "r") as h5:
                dem_group = h5[group]
                elevation = dem_group["elevation"][...]
                geotransform = list(dem_group.attrs.get("geotransform", []))
                epsg = int(dem_group.attrs.get("epsg", 4326))
                ref_height = float(dem_group.attrs.get("ref_height", 0.0))

        return DEMInterpolator(
            elevation=elevation,
            geotransform=geotransform,
            epsg=epsg,
            ref_height=ref_height,
        )

    def x_start(self) -> float:
        return self.geotransform[0]

    def y_start(self) -> float:
        return self.geotransform[3]

    def delta_x(self) -> float:
        return self.geotransform[1]

    def delta_y(self) -> float:
        return self.geotransform[5]

    def width(self) -> int:
        return self.elevation.shape[1]

    def length(self) -> int:
        return self.elevation.shape[0]

    def min_height(self) -> float:
        valid = self.elevation[np.isfinite(self.elevation)]
        if valid.size == 0:
            return self.ref_height
        return float(valid.min())

    def max_height(self) -> float:
        valid = self.elevation[np.isfinite(self.elevation)]
        if valid.size == 0:
            return self.ref_height
        return float(valid.max())

    def mean_height(self) -> float:
        valid = self.elevation[np.isfinite(self.elevation)]
        if valid.size == 0:
            return self.ref_height
        return float(valid.mean())

    def _bound_check(self, row: float, col: float) -> bool:
        return 0 <= row < self.length() and 0 <= col < self.width()

    def interpolate_bilinear(self, row: float, col: float) -> float:
        irow = int(np.floor(row))
        icol = int(np.floor(col))

        if not self._bound_check(row, col):
            return self.ref_height

        r1 = self.elevation[irow, icol]
        r2 = self.elevation[irow, icol + 1]
        r3 = self.elevation[irow + 1, icol]
        r4 = self.elevation[irow + 1, icol + 1]

        if np.isnan(r1) or np.isnan(r2) or np.isnan(r3) or np.isnan(r4):
            return self.ref_height

        dr = row - irow
        dc = col - icol

        return float((1 - dr) * (1 - dc) * r1 + (1 - dr) * dc * r2 + dr * (1 - dc) * r3 + dr * dc * r4)

    def interpolate_nearest(self, row: float, col: float) -> float:
        irow = int(np.round(row))
        icol = int(np.round(col))

        if not (0 <= irow < self.length() and 0 <= icol < self.width()):
            return self.ref_height

        val = self.elevation[irow, icol]
        return float(val) if np.isfinite(val) else self.ref_height

    def _cubic_interpolate(self, p0: float, p1: float, p2: float, p3: float, tfrac: float) -> float:
        tconj = 1.0 - tfrac
        term1 = tfrac * (p2 - p0 * tconj * tconj + (p2 * (tconj * 3.0 + 1.0) - p3 * tconj) * tfrac)
        term2 = p1 * (tfrac * tfrac * (tfrac * 3.0 - 5.0) + 2.0)
        return (term1 + term2) / 2.0

    def interpolate_bicubic(self, row: float, col: float) -> float:
        irow = int(np.floor(row))
        icol = int(np.floor(col))

        if irow < 1 or irow >= self.length() - 2 or icol < 1 or icol >= self.width() - 2:
            return self.interpolate_bilinear(row, col)

        intp = [0.0, 0.0, 0.0, 0.0]
        for i in range(-1, 3):
            vals = [
                self.elevation[irow + i, icol - 1],
                self.elevation[irow + i, icol],
                self.elevation[irow + i, icol + 1],
                self.elevation[irow + i, icol + 2],
            ]
            if any(np.isnan(v) for v in vals):
                return self.ref_height
            intp[i + 1] = self._cubic_interpolate(vals[0], vals[1], vals[2], vals[3], col - icol)

        tfrac_y = row - irow
        return self._cubic_interpolate(intp[0], intp[1], intp[2], intp[3], tfrac_y)

    def _quintic_kernel(self, t: float) -> float:
        if t < 0:
            t = -t
        if t >= 3.0:
            return 0.0
        elif t >= 2.0:
            return ((-t + 3.0)**5 - 6 * (-t + 2.0)**5 + 15 * (-t + 1.0)**5) / 120.0
        elif t >= 1.0:
            return ((3.0 - t)**5 - 6 * (2.0 - t)**5) / 120.0
        else:
            return (312.0 - 150.0 * t**2 + 30.0 * t**4 - 5.0 * t**5) / 120.0

    def interpolate_biquintic(self, row: float, col: float) -> float:
        irow = int(np.floor(row))
        icol = int(np.floor(col))

        margin = 3
        if irow < margin - 1 or irow >= self.length() - margin or icol < margin - 1 or icol >= self.width() - margin:
            return self.interpolate_bicubic(row, col)

        result = 0.0
        weight_sum = 0.0

        for di in range(-margin + 1, margin):
            for dj in range(-margin + 1, margin):
                ni = irow + di
                nj = icol + dj

                if ni < 0 or ni >= self.length() or nj < 0 or nj >= self.width():
                    continue

                val = self.elevation[ni, nj]
                if not np.isfinite(val):
                    continue

                weight_i = self._quintic_kernel(row - ni)
                weight_j = self._quintic_kernel(col - nj)
                weight = weight_i * weight_j

                if weight > 0:
                    result += val * weight
                    weight_sum += weight

        if weight_sum > 0:
            return float(result / weight_sum)
        else:
            return self.ref_height

    def interpolate_sinc(self, row: float, col: float) -> float:
        irow = int(np.floor(row))
        icol = int(np.floor(col))

        half_width = 4
        if irow < half_width or irow >= self.length() - half_width or icol < half_width or icol >= self.width() - half_width:
            return self.interpolate_biquintic(row, col)

        result = 0.0
        weight_sum = 0.0

        for di in range(-half_width, half_width):
            for dj in range(-half_width, half_width):
                ni = irow + di
                nj = icol + dj

                if ni < 0 or ni >= self.length() or nj < 0 or nj >= self.width():
                    continue

                val = self.elevation[ni, nj]
                if not np.isfinite(val):
                    continue

                t_i = row - ni
                t_j = col - nj

                if t_i == 0:
                    weight_i = 1.0
                else:
                    weight_i = np.sin(np.pi * t_i) / (np.pi * t_i)

                if t_j == 0:
                    weight_j = 1.0
                else:
                    weight_j = np.sin(np.pi * t_j) / (np.pi * t_j)

                weight = weight_i * weight_j
                result += val * weight
                weight_sum += weight

        if weight_sum > 0:
            return float(result / weight_sum)
        else:
            return self.ref_height

    def interpolate_at_lonlat(self, lon: float, lat: float) -> float:
        if self.epsg != 4326:
            return self.interpolate_at_xy(lon, lat)

        x = lon
        y = lat
        return self.interpolate_at_xy(x, y)

    def interpolate_at_xy(self, x: float, y: float) -> float:
        if self.epsg == 4326:
            if x > 360 or x < -360:
                x = x % 360
            if x < -180:
                x += 360
            if x - 360 >= self.x_start():
                x -= 360
            elif x < self.x_start() and x + 360 >= self.x_start():
                x += 360
            elif x < self.x_start():
                return self.ref_height

        row = (y - self.y_start()) / self.delta_y()
        col = (x - self.x_start()) / self.delta_x()

        if not self._bound_check(row, col):
            return self.ref_height

        if self.interp_method == InterpMethod.NEAREST:
            return self.interpolate_nearest(row, col)
        elif self.interp_method == InterpMethod.BICUBIC:
            return self.interpolate_bicubic(row, col)
        elif self.interp_method == InterpMethod.BIQUINTIC:
            return self.interpolate_biquintic(row, col)
        elif self.interp_method == InterpMethod.SINC:
            return self.interpolate_sinc(row, col)
        else:
            return self.interpolate_bilinear(row, col)

    def interpolate(self, x: float, y: float) -> float:
        return self.interpolate_at_xy(x, y)
