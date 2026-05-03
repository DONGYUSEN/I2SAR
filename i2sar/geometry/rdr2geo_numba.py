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
def _cross(v1, v2):
    return np.array([
        v1[1]*v2[2] - v1[2]*v2[1],
        v1[2]*v2[0] - v1[0]*v2[2],
        v1[0]*v2[1] - v1[1]*v2[0]
    ])


@njit(fastmath=True)
def _xyz_to_lon_lat(xyz, a, b):
    x, y, z = xyz[0], xyz[1], xyz[2]

    e_sq = 1.0 - (b / a) ** 2
    ep_sq = e_sq / (1.0 - e_sq)

    p = np.sqrt(x * x + y * y)
    lon = np.arctan2(y, x)
    theta = np.arctan2(z * a, p * b)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    lat = np.arctan2(
        z + ep_sq * b * sin_theta ** 3,
        p - e_sq * a * cos_theta ** 3,
    )

    sin_lat = np.sin(lat)
    N = a / np.sqrt(1.0 - e_sq * sin_lat * sin_lat)
    h = p / np.cos(lat) - N

    return np.array([lat * 180.0 / np.pi, lon * 180.0 / np.pi, h], dtype=np.float64)


@njit(fastmath=True)
def _llh_to_ecef(lat, lon, h, a, b):
    lat = lat * np.pi / 180.0
    lon = lon * np.pi / 180.0

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    e_sq = 1.0 - (b/a)**2
    N = a / np.sqrt(1.0 - e_sq * sin_lat**2)

    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1.0 - e_sq) + h) * sin_lat

    return np.array([x, y, z], dtype=np.float64)


@njit(fastmath=True)
def _compute_tcn(sat_pos, vel):
    pos_norm = _norm(sat_pos)
    n = -sat_pos / pos_norm

    c = _cross(n, vel)
    c_norm = _norm(c)
    
    # 处理特殊情况，和原始代码保持一致
    if c_norm < 1e-10:
        t = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        c = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return t, c, n
    
    c = c / c_norm
    
    t = _cross(c, n)
    t = t / _norm(t)

    return t, c, n


@njit(fastmath=True)
def _rdr2geo_single_numba(
    sat_pos, vel, slant_range, doppler, wavelength, look_side_sign,
    dem_height, a, b, max_iterations, extra_iterations, threshold
):
    vmag = _norm(vel)
    vhat = vel / vmag

    t, c, n = _compute_tcn(sat_pos, vel)

    n_dot_v = _dot(n, vhat)
    v_dot_t = _dot(vhat, t)

    dopfact = 0.5 * wavelength * doppler * slant_range / vmag

    major = a
    minor = b
    sat_dist = _norm(sat_pos)

    eta = 1.0 / np.sqrt(
        (sat_pos[0] / major)**2 +
        (sat_pos[1] / major)**2 +
        (sat_pos[2] / minor)**2
    )
    radius = eta * sat_dist
    height = (1.0 - eta) * sat_dist

    llh_new = _xyz_to_lon_lat(sat_pos, a, b)
    h = height

    llh_old = np.zeros(3, dtype=np.float64)

    for iteration in range(max_iterations + extra_iterations):
        if height - h >= slant_range:
            break

        a_val = _norm(sat_pos)
        b_val = radius + h

        cos_theta = 0.5 * (a_val / slant_range + slant_range / a_val - (b_val / a_val) * (b_val / slant_range))
        cos_theta = max(-1.0, min(1.0, cos_theta))
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

        gamma = slant_range * cos_theta
        alpha = (dopfact - gamma * n_dot_v) / v_dot_t

        x = slant_range * sin_theta
        beta_sq = x * x - alpha * alpha
        beta = np.sqrt(max(0.0, beta_sq)) * look_side_sign

        delta = alpha * t + beta * c + gamma * n
        xyz = sat_pos + delta

        llh_new = _xyz_to_lon_lat(xyz, a, b)
        h_new = dem_height

        llh_new = np.array([llh_new[0], llh_new[1], h_new], dtype=np.float64)

        xyz_new = _llh_to_ecef(llh_new[0], llh_new[1], llh_new[2], a, b)
        h = _norm(xyz_new) - radius

        rng = _norm(sat_pos - xyz_new)
        dr = abs(slant_range - rng)

        if dr < threshold:
            break

        if iteration > max_iterations:
            xyz_old = _llh_to_ecef(llh_old[0], llh_old[1], llh_old[2], a, b)
            xyz_avg = 0.5 * (xyz_old + xyz_new)
            llh_new = _xyz_to_lon_lat(xyz_avg, a, b)
            h = _norm(xyz_avg) - radius

        llh_old = llh_new

    a_val = _norm(sat_pos)
    b_val = radius + h

    cos_theta = 0.5 * (a_val / slant_range + slant_range / a_val - (b_val / a_val) * (b_val / slant_range))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

    gamma = slant_range * cos_theta
    alpha = (dopfact - gamma * n_dot_v) / v_dot_t

    x = slant_range * sin_theta
    beta_sq = x * x - alpha * alpha
    beta = np.sqrt(max(0.0, beta_sq)) * look_side_sign

    delta = alpha * t + beta * c + gamma * n
    xyz = sat_pos + delta

    final_llh = _xyz_to_lon_lat(xyz, a, b)

    return np.array([final_llh[0], final_llh[1], final_llh[2]], dtype=np.float64)


@njit(parallel=True, fastmath=True)
def rdr2geo_numba_parallel(
    line_arr: np.ndarray,
    pixel_arr: np.ndarray,
    sensing_start_s: float,
    prf_hz: float,
    starting_range_m: float,
    range_pixel_spacing_m: float,
    sat_positions: np.ndarray,
    sat_velocities: np.ndarray,
    doppler: float,
    wavelength: float,
    look_side_sign: float,
    dem_heights: np.ndarray,
    a: float,
    b: float,
    max_iterations: int,
    extra_iterations: int,
    threshold: float,
    orbit_start_time: float = 0.0,
    orbit_duration: float = 100.0,
    length: int = 10000,
    width: int = 10000
):
    n_points = len(line_arr)
    results_lat = np.full(n_points, np.nan)
    results_lon = np.full(n_points, np.nan)
    results_h = np.full(n_points, np.nan)

    n_orbit_points = len(sat_positions)

    for i in prange(n_points):
        l_i = float(line_arr[i])
        p_i = float(pixel_arr[i])
        
        if not np.isfinite(l_i) or not np.isfinite(p_i):
            continue
        if l_i < 0.0 or l_i >= length or p_i < 0.0 or p_i >= width:
            continue

        t_az = sensing_start_s + l_i / prf_hz

        if n_orbit_points == 1:
            idx = 0
        else:
            t_rel = t_az - orbit_start_time
            ratio = max(0.0, min(1.0, t_rel / orbit_duration))
            idx = min(int(ratio * (n_orbit_points - 1)), n_orbit_points - 1)
        
        sat_pos_i = sat_positions[idx]
        vel_i = sat_velocities[idx]

        slant_range = starting_range_m + p_i * range_pixel_spacing_m

        dem_height = float(dem_heights[i]) if dem_heights.ndim > 0 else 0.0

        result = _rdr2geo_single_numba(
            sat_pos_i, vel_i, slant_range, doppler, wavelength, look_side_sign,
            dem_height, a, b, max_iterations, extra_iterations, threshold
        )

        results_lat[i] = result[0]
        results_lon[i] = result[1]
        results_h[i] = result[2]

    return results_lat, results_lon, results_h
