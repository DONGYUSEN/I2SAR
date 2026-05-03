from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def sample_dem_at_latlons_numba(
    lats: np.ndarray,
    lons: np.ndarray,
    elevation: np.ndarray,
    geotransform: np.ndarray,
) -> np.ndarray:
    """Numba 加速的 DEM 采样函数"""
    n_points = len(lats)
    elevation_out = np.full(n_points, np.nan, dtype=np.float64)
    
    pixel_size = geotransform[1]
    origin_lon = geotransform[0]
    origin_lat = geotransform[3]
    
    rows_f = (origin_lat - lats) / abs(pixel_size)
    cols_f = (lons - origin_lon) / pixel_size
    
    elev_height, elev_width = elevation.shape
    
    for i in prange(n_points):
        r = rows_f[i]
        c = cols_f[i]
        
        if not np.isfinite(r) or not np.isfinite(c):
            continue
        if r < 0 or r >= elev_height - 1 or c < 0 or c >= elev_width - 1:
            continue
        
        r0 = int(np.floor(r))
        c0 = int(np.floor(c))
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
        
        elevation_out[i] = elev_sample
    
    return elevation_out


@njit(parallel=True, fastmath=True)
def sample_dem_bilinear_numba(
    rows: np.ndarray,
    cols: np.ndarray,
    elevation: np.ndarray,
) -> np.ndarray:
    """Numba 加速的双线性插值采样"""
    n_points = len(rows)
    elevation_out = np.full(n_points, np.nan, dtype=np.float64)
    
    elev_height, elev_width = elevation.shape
    
    for i in prange(n_points):
        r = rows[i]
        c = cols[i]
        
        if not np.isfinite(r) or not np.isfinite(c):
            continue
        if r < 0 or r >= elev_height - 1 or c < 0 or c >= elev_width - 1:
            continue
        
        r0 = int(np.floor(r))
        c0 = int(np.floor(c))
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
        
        elevation_out[i] = elev_sample
    
    return elevation_out
