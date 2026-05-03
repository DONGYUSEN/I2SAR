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
def _llh_to_ecef(lat, lon, h, a, b):
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    
    e_sq = 1 - (b/a)**2
    N = a / np.sqrt(1 - e_sq * sin_lat**2)
    
    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1 - e_sq) + h) * sin_lat
    
    return np.array([x, y, z])


@njit(fastmath=True)
def _check_look_side(rvec, vel, pos):
    cross_prod = np.array([
        rvec[1]*vel[2] - rvec[2]*vel[1],
        rvec[2]*vel[0] - rvec[0]*vel[2],
        rvec[0]*vel[1] - rvec[1]*vel[0]
    ])
    dot_prod = cross_prod[0]*pos[0] + cross_prod[1]*pos[1] + cross_prod[2]*pos[2]
    return dot_prod > 0


@njit(fastmath=True)
def _find_closest_aztime(target_ecef, sat_positions, sat_velocities, sat_times, look_side_right):
    n_positions = len(sat_positions)
    r_closest = 1e16
    idx_closest = 0
    valid_found = False
    
    for k in range(n_positions):
        pos = sat_positions[k]
        vel = sat_velocities[k]
        rvec = target_ecef - pos
        r = _norm(rvec)
        
        is_valid = _check_look_side(rvec, vel, pos)
        if look_side_right != is_valid:
            continue
        
        valid_found = True
        if r < r_closest:
            r_closest = r
            idx_closest = k
    
    if not valid_found:
        return (sat_times[0] + sat_times[-1]) / 2.0
    
    return sat_times[idx_closest]


@njit(fastmath=True)
def _linear_interpolate(t, t0, t1, v0, v1):
    """线性插值"""
    if t1 == t0:
        return v0
    alpha = (t - t0) / (t1 - t0)
    return v0 + alpha * (v1 - v0)


@njit(fastmath=True)
def _orbit_interpolate(t, sat_times, sat_positions, sat_velocities):
    """对轨道进行线性插值"""
    n = len(sat_times)
    
    if n == 1:
        return sat_positions[0], sat_velocities[0]
    
    idx = 0
    for i in range(n - 1):
        if sat_times[i] <= t <= sat_times[i + 1]:
            idx = i
            break
    
    if idx == n - 1:
        idx = n - 2
    
    t0 = sat_times[idx]
    t1 = sat_times[idx + 1]
    
    pos = _linear_interpolate(t, t0, t1, sat_positions[idx], sat_positions[idx + 1])
    vel = _linear_interpolate(t, t0, t1, sat_velocities[idx], sat_velocities[idx + 1])
    
    return pos, vel


@njit(fastmath=True)
def _geo2rdr_single_numba(
    lat, lon, height,
    sat_positions, sat_velocities, sat_times,
    doppler, wavelength,
    a, b,
    max_iterations, threshold,
    sensing_start_s, prf_hz, length,
    look_side_right
):
    target_ecef = _llh_to_ecef(lat, lon, height, a, b)
    
    n_positions = len(sat_positions)
    
    t_az = _find_closest_aztime(target_ecef, sat_positions, sat_velocities, sat_times, look_side_right)
    
    dt = 0.0
    
    for iteration in range(max_iterations):
        t_az = t_az - dt
        
        t_az = max(sat_times[0], min(t_az, sat_times[-1]))
        
        sat_pos_i, vel_i = _orbit_interpolate(t_az, sat_times, sat_positions, sat_velocities)
        
        rvec = target_ecef - sat_pos_i
        slant_range = _norm(rvec)
        
        dopfact = _dot(rvec, vel_i)
        fdop = 0.5 * wavelength * doppler
        
        c1 = -_dot(vel_i, vel_i)
        c2 = fdop / slant_range
        fnprime = c1 + c2 * dopfact
        
        fn = dopfact - fdop * slant_range
        dt = fn / fnprime if fnprime != 0 else 0.0
        
        if abs(dt) < threshold:
            break
    
    sat_pos_i, vel_i = _orbit_interpolate(t_az, sat_times, sat_positions, sat_velocities)
    rvec = target_ecef - sat_pos_i
    final_range = _norm(rvec)
    
    return t_az, final_range


@njit(parallel=True, fastmath=True)
def geo2rdr_numba_parallel(
    lat_arr, lon_arr, h_arr,
    sat_positions, sat_velocities, sat_times,
    doppler, wavelength,
    a, b,
    max_iterations, threshold,
    sensing_start_s, prf_hz, length,
    look_side_right
):
    n_points = len(lat_arr)
    results_aztime = np.full(n_points, np.nan)
    results_range = np.full(n_points, np.nan)
    
    for i in prange(n_points):
        lat_i = lat_arr[i]
        lon_i = lon_arr[i]
        h_i = h_arr[i]
        
        if not np.isfinite(lat_i) or not np.isfinite(lon_i) or not np.isfinite(h_i):
            continue
        
        if lat_i < -90.0 or lat_i > 90.0 or lon_i < -180.0 or lon_i > 180.0:
            continue
        if h_i < -1000.0 or h_i > 10000.0:
            continue
        
        lat_rad = lat_i * np.pi / 180.0
        lon_rad = lon_i * np.pi / 180.0
        
        aztime, slant_range = _geo2rdr_single_numba(
            lat_rad, lon_rad, h_i,
            sat_positions, sat_velocities, sat_times,
            doppler, wavelength,
            a, b,
            max_iterations, threshold,
            sensing_start_s, prf_hz, length,
            look_side_right
        )
        
        results_aztime[i] = aztime
        results_range[i] = slant_range
    
    return results_aztime, results_range


@njit(parallel=True, fastmath=True)
def static_geo2rdr_numba(
    lat_arr,
    lon_arr,
    h_arr,
    sat_pos,
    vel,
    sensing_start_s: float,
    prf_hz: float,
    length: int,
    a: float,
    b: float,
    doppler: float = 0.0,
    wavelength: float = 0.0565642,
    max_iterations: int = 50,
    threshold: float = 1e-8,
    look_side_right: bool = True,
):
    n_points = len(lat_arr)
    results_aztime = np.full(n_points, np.nan)
    results_range = np.full(n_points, np.nan)
    
    e_sq = 1 - (b / a) ** 2
    delta_range = 10.0
    
    for i in prange(n_points):
        lat_i = lat_arr[i]
        lon_i = lon_arr[i]
        h_i = h_arr[i]
        
        if not np.isfinite(lat_i) or not np.isfinite(lon_i) or not np.isfinite(h_i):
            continue
        if lat_i < -90.0 or lat_i > 90.0 or lon_i < -180.0 or lon_i > 180.0:
            continue
        
        lat_rad = lat_i * np.pi / 180.0
        lon_rad = lon_i * np.pi / 180.0
        
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        
        N = a / np.sqrt(1 - e_sq * sin_lat**2)
        x = (N + h_i) * cos_lat * cos_lon
        y = (N + h_i) * cos_lat * sin_lon
        z = (N * (1 - e_sq) + h_i) * sin_lat
        
        target_ecef = np.array([x, y, z])
        
        aztime = sensing_start_s + (length / prf_hz) / 2.0
        
        rvec = target_ecef - sat_pos
        
        is_valid = _check_look_side(rvec, vel, sat_pos)
        if look_side_right != is_valid:
            results_aztime[i] = aztime
            results_range[i] = _norm(rvec)
            continue
        
        dt = 0.0
        for iteration in range(max_iterations):
            aztime = aztime - dt
            
            rvec = target_ecef - sat_pos
            slant_range = _norm(rvec)
            
            dopfact = _dot(rvec, vel)
            fdop = 0.5 * wavelength * doppler
            
            fdop_der = 0.0
            
            fn = dopfact - fdop * slant_range
            c1 = -_dot(vel, vel)
            c2 = (fdop / slant_range) + fdop_der
            fnprime = c1 + c2 * dopfact
            
            dt = fn / fnprime if fnprime != 0 else 0.0
            
            if abs(dt) < threshold:
                break
        
        results_aztime[i] = aztime
        results_range[i] = _norm(rvec)

    return results_aztime, results_range
