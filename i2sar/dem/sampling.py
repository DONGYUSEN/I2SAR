from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def read_dem_from_hdf5(h5_or_path, group: str = "dem") -> tuple[np.ndarray, np.ndarray, list[float]]:
    if isinstance(h5_or_path, (str, Path)):
        with h5py.File(h5_or_path, "r") as h5:
            return _read_dem_data(h5, group)
    return _read_dem_data(h5_or_path, group)


def _read_dem_data(h5, group: str) -> tuple[np.ndarray, np.ndarray, list[float]]:
    dem_group = h5[group]
    elevation = dem_group["elevation"][...]
    mask = dem_group["mask"][...]
    geotransform = dem_group.attrs.get("geotransform", [])
    if hasattr(geotransform, "__iter__") and not isinstance(geotransform, str):
        geotransform = list(geotransform)
    return elevation, mask, geotransform


def sample_dem_at_latlons(
    lats: np.ndarray,
    lons: np.ndarray,
    elevation: np.ndarray,
    geotransform: list[float],
    use_numba: bool = True,
) -> np.ndarray:
    if geotransform is None or len(geotransform) < 6:
        raise ValueError("geotransform must have at least 6 elements")

    if use_numba and len(lats) > 100:
        try:
            from .sampling_numba import sample_dem_at_latlons_numba
            return sample_dem_at_latlons_numba(lats, lons, elevation, np.array(geotransform))
        except ImportError:
            pass

    pixel_size = geotransform[1]
    origin_lon = geotransform[0]
    origin_lat = geotransform[3]

    col = (lons - origin_lon) / pixel_size
    row = (origin_lat - lats) / abs(pixel_size)

    rows_f = np.asarray(row, dtype=np.float64)
    cols_f = np.asarray(col, dtype=np.float64)

    elevation_out = np.full_like(lats, np.nan, dtype=np.float64)

    valid = (
        np.isfinite(rows_f)
        & np.isfinite(cols_f)
        & np.isfinite(lats)
        & np.isfinite(lons)
        & (rows_f >= 0)
        & (rows_f < elevation.shape[0] - 1)
        & (cols_f >= 0)
        & (cols_f < elevation.shape[1] - 1)
    )

    if not np.any(valid):
        return elevation_out

    r = rows_f[valid]
    c = cols_f[valid]

    r0 = np.floor(r).astype(int)
    c0 = np.floor(c).astype(int)
    dr = r - r0
    dc = c - c0

    elev00 = elevation[r0, c0]
    elev01 = elevation[r0, c0 + 1]
    elev10 = elevation[r0 + 1, c0]
    elev11 = elevation[r0 + 1, c0 + 1]

    elev_sample = (
        elev00 * (1 - dr) * (1 - dc)
        + elev01 * (1 - dr) * dc
        + elev10 * dr * (1 - dc)
        + elev11 * dr * dc
    )

    elevation_out[valid] = elev_sample

    return elevation_out


def sample_dem_product(
    dem_h5_path: str | Path,
    lats: np.ndarray,
    lons: np.ndarray,
    group: str = "dem",
) -> np.ndarray:
    elevation, mask, geotransform = read_dem_from_hdf5(dem_h5_path, group)
    return sample_dem_at_latlons(lats, lons, elevation, geotransform)
