from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(fastmath=True)
def _norm(v):
    return np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


@njit(fastmath=True)
def _dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


@njit(fastmath=True)
def _llh_to_ecef_numba(lat, lon, height, a, b):
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    
    e_sq = 1 - (b/a)**2
    N = a / np.sqrt(1 - e_sq * sin_lat**2)
    
    x = (N + height) * cos_lat * cos_lon
    y = (N + height) * cos_lat * sin_lon
    z = (N * (1 - e_sq) + height) * sin_lat
    
    return x, y, z


@njit(parallel=True, fastmath=True)
def compute_incidence_angle_numba(
    lat: np.ndarray,
    lon: np.ndarray,
    height: np.ndarray,
    az_times: np.ndarray,
    orbit_times: np.ndarray,
    orbit_positions: np.ndarray,
) -> np.ndarray:
    """Numba 加速的入射角计算"""
    n_points = len(lat)
    inc_angle = np.zeros(n_points, dtype=np.float64)
    
    a = 6378137.0
    b = 6356752.314245
    
    n_orbit = len(orbit_times)
    
    for i in prange(n_points):
        lat_i = np.deg2rad(lat[i])
        lon_i = np.deg2rad(lon[i])
        height_i = height[i]
        
        x, y, z = _llh_to_ecef_numba(lat_i, lon_i, height_i, a, b)
        target_ecef = np.array([x, y, z])
        
        time = az_times[i]
        
        idx = np.searchsorted(orbit_times, time, side='right') - 1
        idx = max(0, min(idx, n_orbit - 1))
        
        sat_pos = orbit_positions[idx]
        
        look_vec = target_ecef - sat_pos
        look_vec_norm = _norm(look_vec)
        look_dir = look_vec / look_vec_norm
        
        nadir_vec = -sat_pos / _norm(sat_pos)
        
        cos_inc = _dot(look_dir, nadir_vec)
        cos_inc = max(-1.0, min(1.0, cos_inc))
        inc_angle[i] = np.arccos(cos_inc)
    
    return inc_angle
