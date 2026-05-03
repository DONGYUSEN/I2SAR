from __future__ import annotations

from typing import Union

import numpy as np

from i2sar.accel.arrayfire_backend import ArrayFireBackend, asarray, to_numpy
from i2sar.core.enums import LookSide
from i2sar.geometry.ellipsoid import Ellipsoid, WGS84
from i2sar.geometry.radar_grid import RadarGrid
from i2sar.orbit.interpolate import OrbitInterpolator


class Rdr2GeoGpuContext:
    """GPU 上下文管理器，负责内存管理和资源复用"""
    
    def __init__(self, af):
        self._af = af
        self._constant_cache = {}
    
    def get_constant(self, name: str, value: float, n_points: int):
        """获取或创建常量数组，复用已分配的内存"""
        key = (name, value, n_points)
        if key not in self._constant_cache:
            self._constant_cache[key] = self._af.constant(value, n_points, dtype=self._af.Dtype.f64)
        return self._constant_cache[key]
    
    def clear_cache(self):
        """清除缓存，释放内存"""
        self._constant_cache.clear()


def rdr2geo_arrayfire_optimized(
    line: Union[int, float, np.ndarray],
    pixel: Union[int, float, np.ndarray],
    radar_grid: RadarGrid,
    satellite_position: np.ndarray,
    velocity: np.ndarray,
    doppler: float,
    dem: Union[float, np.ndarray] = 0.0,
    ellipsoid: Ellipsoid = WGS84,
    wavelength_m: float = 0.0565642,
    look_side: LookSide = LookSide.RIGHT,
    max_iterations: int = 25,
    extra_iterations: int = 15,
    threshold: float = 1e-8,
) -> np.ndarray:
    backend = ArrayFireBackend()
    if not backend.available:
        raise RuntimeError(backend.reason)
    af = backend.module
    
    line_arr = np.atleast_1d(np.asarray(line, dtype=np.float64))
    pixel_arr = np.atleast_1d(np.asarray(pixel, dtype=np.float64))
    n_points = len(line_arr)
    
    if isinstance(dem, (int, float)):
        dem_heights = np.full(n_points, float(dem), dtype=np.float64)
    elif dem.ndim == 1:
        dem_heights = np.asarray(dem, dtype=np.float64)
    else:
        dem_heights = dem[line_arr.astype(np.int64), pixel_arr.astype(np.int64)].astype(np.float64)
    
    sat = np.asarray(satellite_position, dtype=np.float64)
    vel = np.asarray(velocity, dtype=np.float64)
    sx, sy, sz = float(sat[0]), float(sat[1]), float(sat[2])
    vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
    
    slant_range = radar_grid.starting_range_m + pixel_arr * radar_grid.range_pixel_spacing_m
    vmag = float(np.linalg.norm(vel))
    vhat = vel / vmag
    
    pos_norm = float(np.linalg.norm(sat))
    nx, ny, nz = -sx / pos_norm, -sy / pos_norm, -sz / pos_norm
    
    cx = ny * vz - nz * vy
    cy = nz * vx - nx * vz
    cz = nx * vy - ny * vx
    c_norm = float(np.sqrt(cx * cx + cy * cy + cz * cz))
    if c_norm < 1e-10:
        tx, ty, tz = 1.0, 0.0, 0.0
        cx, cy, cz = 0.0, 1.0, 0.0
    else:
        cx, cy, cz = cx / c_norm, cy / c_norm, cz / c_norm
        tx = cy * nz - cz * ny
        ty = cz * nx - cx * nz
        tz = cx * ny - cy * nx
        t_norm = float(np.sqrt(tx * tx + ty * ty + tz * tz))
        tx, ty, tz = tx / t_norm, ty / t_norm, tz / t_norm
    
    n_dot_v = nx * vhat[0] + ny * vhat[1] + nz * vhat[2]
    v_dot_t = vhat[0] * tx + vhat[1] * ty + vhat[2] * tz
    dopfact = 0.5 * wavelength_m * float(doppler) * slant_range / vmag
    
    minor = ellipsoid.b
    eta = 1.0 / np.sqrt((sx / ellipsoid.a) ** 2 + (sy / ellipsoid.a) ** 2 + (sz / minor) ** 2)
    radius = eta * pos_norm
    height = (1.0 - eta) * pos_norm
    
    ctx = Rdr2GeoGpuContext(af)
    
    slant_range_af = asarray(af, slant_range)
    dopfact_af = asarray(af, dopfact)
    dem_af = asarray(af, dem_heights)
    
    sx_af = ctx.get_constant('sx', sx, n_points)
    sy_af = ctx.get_constant('sy', sy, n_points)
    sz_af = ctx.get_constant('sz', sz, n_points)
    tx_af = ctx.get_constant('tx', tx, n_points)
    ty_af = ctx.get_constant('ty', ty, n_points)
    tz_af = ctx.get_constant('tz', tz, n_points)
    cx_af = ctx.get_constant('cx', cx, n_points)
    cy_af = ctx.get_constant('cy', cy, n_points)
    cz_af = ctx.get_constant('cz', cz, n_points)
    nx_af = ctx.get_constant('nx', nx, n_points)
    ny_af = ctx.get_constant('ny', ny, n_points)
    nz_af = ctx.get_constant('nz', nz, n_points)
    
    pos_norm_af = ctx.get_constant('pos_norm', pos_norm, n_points)
    radius_af = ctx.get_constant('radius', radius, n_points)
    n_dot_v_af = ctx.get_constant('n_dot_v', n_dot_v, n_points)
    v_dot_t_af = ctx.get_constant('v_dot_t', v_dot_t, n_points)
    look_sign_af = ctx.get_constant('look_sign', look_side.sign, n_points)
    
    a_af = ctx.get_constant('a', ellipsoid.a, n_points)
    b_af = ctx.get_constant('b', ellipsoid.b, n_points)
    e2_af = ctx.get_constant('e2', ellipsoid.e2, n_points)
    ep2_af = ctx.get_constant('ep2', ellipsoid.ep2, n_points)
    
    h_af = ctx.get_constant('h_init', height, n_points)
    
    cos_theta = 0.5 * (pos_norm_af / slant_range_af + slant_range_af / pos_norm_af - (a_af / pos_norm_af) * (a_af / slant_range_af))
    cos_theta = af.clamp(cos_theta, -1.0, 1.0)
    sin_theta = af.sqrt(1.0 - cos_theta * cos_theta)
    gamma = slant_range_af * cos_theta
    alpha = (dopfact_af - gamma * n_dot_v_af) / v_dot_t_af
    across = slant_range_af * sin_theta
    beta = af.sqrt(af.maxof(across * across - alpha * alpha, 0.0)) * look_sign_af
    
    x = sx_af + alpha * tx_af + beta * cx_af + gamma * nx_af
    y = sy_af + alpha * ty_af + beta * cy_af + gamma * ny_af
    z = sz_af + alpha * tz_af + beta * cz_af + gamma * nz_af
    
    for iteration in range(max_iterations):
        p = af.sqrt(x * x + y * y)
        lon = af.atan2(y, x)
        theta = af.atan2(z * a_af, p * b_af)
        sin_theta = af.sin(theta)
        cos_theta = af.cos(theta)
        lat = af.atan2(z + ep2_af * b_af * sin_theta * sin_theta * sin_theta, p - e2_af * a_af * cos_theta * cos_theta * cos_theta)
        
        sin_lat = af.sin(lat)
        cos_lat = af.cos(lat)
        sin_lon = af.sin(lon)
        cos_lon = af.cos(lon)
        prime_vertical = a_af / af.sqrt(1.0 - e2_af * sin_lat * sin_lat)
        
        x_new = (prime_vertical + dem_af) * cos_lat * cos_lon
        y_new = (prime_vertical + dem_af) * cos_lat * sin_lon
        z_new = (prime_vertical * (1.0 - e2_af) + dem_af) * sin_lat
        
        xyz_norm = af.sqrt(x_new * x_new + y_new * y_new + z_new * z_new)
        h_af = xyz_norm - radius_af
        
        b_val = radius_af + h_af
        cos_theta = 0.5 * (pos_norm_af / slant_range_af + slant_range_af / pos_norm_af - (b_val / pos_norm_af) * (b_val / slant_range_af))
        cos_theta = af.clamp(cos_theta, -1.0, 1.0)
        sin_theta = af.sqrt(1.0 - cos_theta * cos_theta)
        gamma = slant_range_af * cos_theta
        alpha = (dopfact_af - gamma * n_dot_v_af) / v_dot_t_af
        across = slant_range_af * sin_theta
        beta = af.sqrt(af.maxof(across * across - alpha * alpha, 0.0)) * look_sign_af
        
        x = sx_af + alpha * tx_af + beta * cx_af + gamma * nx_af
        y = sy_af + alpha * ty_af + beta * cy_af + gamma * ny_af
        z = sz_af + alpha * tz_af + beta * cz_af + gamma * nz_af
    
    for iteration in range(extra_iterations):
        b_val = radius_af + h_af
        cos_theta = 0.5 * (pos_norm_af / slant_range_af + slant_range_af / pos_norm_af - (b_val / pos_norm_af) * (b_val / slant_range_af))
        cos_theta = af.clamp(cos_theta, -1.0, 1.0)
        sin_theta = af.sqrt(1.0 - cos_theta * cos_theta)
        gamma = slant_range_af * cos_theta
        alpha = (dopfact_af - gamma * n_dot_v_af) / v_dot_t_af
        across = slant_range_af * sin_theta
        beta = af.sqrt(af.maxof(across * across - alpha * alpha, 0.0)) * look_sign_af
        
        x_new = sx_af + alpha * tx_af + beta * cx_af + gamma * nx_af
        y_new = sy_af + alpha * ty_af + beta * cy_af + gamma * ny_af
        z_new = sz_af + alpha * tz_af + beta * cz_af + gamma * nz_af
        
        p_new = af.sqrt(x_new * x_new + y_new * y_new)
        lon_new = af.atan2(y_new, x_new)
        theta = af.atan2(z_new * a_af, p_new * b_af)
        sin_theta = af.sin(theta)
        cos_theta = af.cos(theta)
        lat_new = af.atan2(z_new + ep2_af * b_af * sin_theta * sin_theta * sin_theta, p_new - e2_af * a_af * cos_theta * cos_theta * cos_theta)
        
        sin_lat_new = af.sin(lat_new)
        cos_lat_new = af.cos(lat_new)
        sin_lon_new = af.sin(lon_new)
        cos_lon_new = af.cos(lon_new)
        prime_vertical_new = a_af / af.sqrt(1.0 - e2_af * sin_lat_new * sin_lat_new)
        
        x_ecef_new = (prime_vertical_new + dem_af) * cos_lat_new * cos_lon_new
        y_ecef_new = (prime_vertical_new + dem_af) * cos_lat_new * sin_lon_new
        z_ecef_new = (prime_vertical_new * (1.0 - e2_af) + dem_af) * sin_lat_new
        
        p_old = af.sqrt(x * x + y * y)
        lon_old = af.atan2(y, x)
        theta = af.atan2(z * a_af, p_old * b_af)
        sin_theta = af.sin(theta)
        cos_theta = af.cos(theta)
        lat_old = af.atan2(z + ep2_af * b_af * sin_theta * sin_theta * sin_theta, p_old - e2_af * a_af * cos_theta * cos_theta * cos_theta)
        
        sin_lat_old = af.sin(lat_old)
        cos_lat_old = af.cos(lat_old)
        sin_lon_old = af.sin(lon_old)
        cos_lon_old = af.cos(lon_old)
        prime_vertical_old = a_af / af.sqrt(1.0 - e2_af * sin_lat_old * sin_lat_old)
        
        x_ecef_old = (prime_vertical_old + h_af) * cos_lat_old * cos_lon_old
        y_ecef_old = (prime_vertical_old + h_af) * cos_lat_old * sin_lon_old
        z_ecef_old = (prime_vertical_old * (1.0 - e2_af) + h_af) * sin_lat_old
        
        x_avg = 0.5 * (x_ecef_old + x_ecef_new)
        y_avg = 0.5 * (y_ecef_old + y_ecef_new)
        z_avg = 0.5 * (z_ecef_old + z_ecef_new)
        
        xyz_avg_norm = af.sqrt(x_avg * x_avg + y_avg * y_avg + z_avg * z_avg)
        h_af = xyz_avg_norm - radius_af
        
        x = x_avg
        y = y_avg
        z = z_avg
    
    p = af.sqrt(x * x + y * y)
    lon = af.atan2(y, x)
    theta = af.atan2(z * a_af, p * b_af)
    sin_theta = af.sin(theta)
    cos_theta = af.cos(theta)
    lat = af.atan2(z + ep2_af * b_af * sin_theta * sin_theta * sin_theta, p - e2_af * a_af * cos_theta * cos_theta * cos_theta)
    
    sin_lat = af.sin(lat)
    cos_lat = af.cos(lat)
    prime_vertical = a_af / af.sqrt(1.0 - e2_af * sin_lat * sin_lat)
    h_af = p / cos_lat - prime_vertical
    
    af.eval(lat, lon, h_af)
    af.sync()
    
    lat_np = to_numpy(lat).ravel() * (180.0 / np.pi)
    lon_np = to_numpy(lon).ravel() * (180.0 / np.pi)
    h_np = to_numpy(h_af).ravel()
    
    ctx.clear_cache()
    
    result = np.stack([lat_np, lon_np, h_np], axis=0)
    if n_points == 1:
        return result[:, 0]
    return result


def rdr2geo_arrayfire_adaptive(
    line: Union[int, float, np.ndarray],
    pixel: Union[int, float, np.ndarray],
    radar_grid: RadarGrid,
    satellite_position: np.ndarray,
    velocity: np.ndarray,
    doppler: float,
    dem: Union[float, np.ndarray] = 0.0,
    ellipsoid: Ellipsoid = WGS84,
    wavelength_m: float = 0.0565642,
    look_side: LookSide = LookSide.RIGHT,
    max_iterations: int = 50,
    threshold: float = 1e-8,
) -> np.ndarray:
    """优化版 rdr2geo，支持动态收敛检测"""
    backend = ArrayFireBackend()
    if not backend.available:
        raise RuntimeError(backend.reason)
    af = backend.module
    
    line_arr = np.atleast_1d(np.asarray(line, dtype=np.float64))
    pixel_arr = np.atleast_1d(np.asarray(pixel, dtype=np.float64))
    n_points = len(line_arr)
    
    if isinstance(dem, (int, float)):
        dem_heights = np.full(n_points, float(dem), dtype=np.float64)
    elif dem.ndim == 1:
        dem_heights = np.asarray(dem, dtype=np.float64)
    else:
        dem_heights = dem[line_arr.astype(np.int64), pixel_arr.astype(np.int64)].astype(np.float64)
    
    sat = np.asarray(satellite_position, dtype=np.float64)
    vel = np.asarray(velocity, dtype=np.float64)
    sx, sy, sz = float(sat[0]), float(sat[1]), float(sat[2])
    vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
    
    slant_range = radar_grid.starting_range_m + pixel_arr * radar_grid.range_pixel_spacing_m
    vmag = float(np.linalg.norm(vel))
    vhat = vel / vmag
    
    pos_norm = float(np.linalg.norm(sat))
    nx, ny, nz = -sx / pos_norm, -sy / pos_norm, -sz / pos_norm
    
    cx = ny * vz - nz * vy
    cy = nz * vx - nx * vz
    cz = nx * vy - ny * vx
    c_norm = float(np.sqrt(cx * cx + cy * cy + cz * cz))
    if c_norm < 1e-10:
        tx, ty, tz = 1.0, 0.0, 0.0
        cx, cy, cz = 0.0, 1.0, 0.0
    else:
        cx, cy, cz = cx / c_norm, cy / c_norm, cz / c_norm
        tx = cy * nz - cz * ny
        ty = cz * nx - cx * nz
        tz = cx * ny - cy * nx
        t_norm = float(np.sqrt(tx * tx + ty * ty + tz * tz))
        tx, ty, tz = tx / t_norm, ty / t_norm, tz / t_norm
    
    n_dot_v = nx * vhat[0] + ny * vhat[1] + nz * vhat[2]
    v_dot_t = vhat[0] * tx + vhat[1] * ty + vhat[2] * tz
    dopfact = 0.5 * wavelength_m * float(doppler) * slant_range / vmag
    
    slant_range_af = asarray(af, slant_range)
    dopfact_af = asarray(af, dopfact)
    dem_af = asarray(af, dem_heights)
    
    sx_af = af.constant(sx, n_points, dtype=af.Dtype.f64)
    sy_af = af.constant(sy, n_points, dtype=af.Dtype.f64)
    sz_af = af.constant(sz, n_points, dtype=af.Dtype.f64)
    tx_af = af.constant(tx, n_points, dtype=af.Dtype.f64)
    ty_af = af.constant(ty, n_points, dtype=af.Dtype.f64)
    tz_af = af.constant(tz, n_points, dtype=af.Dtype.f64)
    cx_af = af.constant(cx, n_points, dtype=af.Dtype.f64)
    cy_af = af.constant(cy, n_points, dtype=af.Dtype.f64)
    cz_af = af.constant(cz, n_points, dtype=af.Dtype.f64)
    nx_af = af.constant(nx, n_points, dtype=af.Dtype.f64)
    ny_af = af.constant(ny, n_points, dtype=af.Dtype.f64)
    nz_af = af.constant(nz, n_points, dtype=af.Dtype.f64)
    
    minor = ellipsoid.b
    eta = 1.0 / np.sqrt((sx / ellipsoid.a) ** 2 + (sy / ellipsoid.a) ** 2 + (sz / minor) ** 2)
    radius = eta * pos_norm
    
    pos_norm_af = af.constant(pos_norm, n_points, dtype=af.Dtype.f64)
    radius_af = af.constant(radius, n_points, dtype=af.Dtype.f64)
    n_dot_v_af = af.constant(n_dot_v, n_points, dtype=af.Dtype.f64)
    v_dot_t_af = af.constant(v_dot_t, n_points, dtype=af.Dtype.f64)
    look_sign_af = af.constant(look_side.sign, n_points, dtype=af.Dtype.f64)
    
    a_af = af.constant(ellipsoid.a, n_points, dtype=af.Dtype.f64)
    b_af = af.constant(ellipsoid.b, n_points, dtype=af.Dtype.f64)
    e2_af = af.constant(ellipsoid.e2, n_points, dtype=af.Dtype.f64)
    ep2_af = af.constant(ellipsoid.ep2, n_points, dtype=af.Dtype.f64)
    
    cos_theta = 0.5 * (pos_norm_af / slant_range_af + slant_range_af / pos_norm_af - (a_af / pos_norm_af) * (a_af / slant_range_af))
    cos_theta = af.clamp(cos_theta, -1.0, 1.0)
    sin_theta = af.sqrt(1.0 - cos_theta * cos_theta)
    gamma = slant_range_af * cos_theta
    alpha = (dopfact_af - gamma * n_dot_v_af) / v_dot_t_af
    across = slant_range_af * sin_theta
    beta = af.sqrt(af.maxof(across * across - alpha * alpha, 0.0)) * look_sign_af
    
    x = sx_af + alpha * tx_af + beta * cx_af + gamma * nx_af
    y = sy_af + alpha * ty_af + beta * cy_af + gamma * ny_af
    z = sz_af + alpha * tz_af + beta * cz_af + gamma * nz_af
    
    xyz_norm = af.sqrt(x * x + y * y + z * z)
    h_af = xyz_norm - radius_af
    
    converged = af.constant(False, n_points, dtype=af.Dtype.b8)
    
    for iteration in range(max_iterations):
        p = af.sqrt(x * x + y * y)
        lon = af.atan2(y, x)
        theta = af.atan2(z * a_af, p * b_af)
        sin_theta = af.sin(theta)
        cos_theta = af.cos(theta)
        lat = af.atan2(z + ep2_af * b_af * sin_theta * sin_theta * sin_theta, p - e2_af * a_af * cos_theta * cos_theta * cos_theta)
        
        sin_lat = af.sin(lat)
        cos_lat = af.cos(lat)
        sin_lon = af.sin(lon)
        cos_lon = af.cos(lon)
        prime_vertical = a_af / af.sqrt(1.0 - e2_af * sin_lat * sin_lat)
        
        x_new = (prime_vertical + dem_af) * cos_lat * cos_lon
        y_new = (prime_vertical + dem_af) * cos_lat * sin_lon
        z_new = (prime_vertical * (1.0 - e2_af) + dem_af) * sin_lat
        
        xyz_norm = af.sqrt(x_new * x_new + y_new * y_new + z_new * z_new)
        h_new = xyz_norm - radius_af
        
        delta_x = af.abs(x_new - x)
        delta_y = af.abs(y_new - y)
        delta_z = af.abs(z_new - z)
        delta_h = af.abs(h_new - h_af)
        
        max_delta = af.max(af.join(0, delta_x, delta_y, delta_z, delta_h))
        
        if max_delta < threshold:
            break
        
        h_af = h_new
        x = x_new
        y = y_new
        z = z_new
        
        b_val = radius_af + h_af
        cos_theta = 0.5 * (pos_norm_af / slant_range_af + slant_range_af / pos_norm_af - (b_val / pos_norm_af) * (b_val / slant_range_af))
        cos_theta = af.clamp(cos_theta, -1.0, 1.0)
        sin_theta = af.sqrt(1.0 - cos_theta * cos_theta)
        gamma = slant_range_af * cos_theta
        alpha = (dopfact_af - gamma * n_dot_v_af) / v_dot_t_af
        across = slant_range_af * sin_theta
        beta = af.sqrt(af.maxof(across * across - alpha * alpha, 0.0)) * look_sign_af
        
        x = sx_af + alpha * tx_af + beta * cx_af + gamma * nx_af
        y = sy_af + alpha * ty_af + beta * cy_af + gamma * ny_af
        z = sz_af + alpha * tz_af + beta * cz_af + gamma * nz_af
    
    p = af.sqrt(x * x + y * y)
    lon = af.atan2(y, x)
    theta = af.atan2(z * a_af, p * b_af)
    sin_theta = af.sin(theta)
    cos_theta = af.cos(theta)
    lat = af.atan2(z + ep2_af * b_af * sin_theta * sin_theta * sin_theta, p - e2_af * a_af * cos_theta * cos_theta * cos_theta)
    
    sin_lat = af.sin(lat)
    cos_lat = af.cos(lat)
    prime_vertical = a_af / af.sqrt(1.0 - e2_af * sin_lat * sin_lat)
    h_af = p / cos_lat - prime_vertical
    
    af.eval(lat, lon, h_af)
    af.sync()
    
    lat_np = to_numpy(lat).ravel() * (180.0 / np.pi)
    lon_np = to_numpy(lon).ravel() * (180.0 / np.pi)
    h_np = to_numpy(h_af).ravel()
    
    result = np.stack([lat_np, lon_np, h_np], axis=0)
    if n_points == 1:
        return result[:, 0]
    return result


def rdr2geo_arrayfire_chunked(
    line: np.ndarray,
    pixel: np.ndarray,
    radar_grid: RadarGrid,
    satellite_position: np.ndarray,
    velocity: np.ndarray,
    doppler: float,
    dem: np.ndarray,
    ellipsoid: Ellipsoid = WGS84,
    wavelength_m: float = 0.0565642,
    look_side: LookSide = LookSide.RIGHT,
    max_iterations: int = 25,
    extra_iterations: int = 15,
    threshold: float = 1e-8,
    chunk_size: int = 65536,
) -> np.ndarray:
    backend = ArrayFireBackend()
    if not backend.available:
        raise RuntimeError(backend.reason)
    
    n_points = len(line)
    result_lat = np.zeros(n_points, dtype=np.float64)
    result_lon = np.zeros(n_points, dtype=np.float64)
    result_hgt = np.zeros(n_points, dtype=np.float64)
    
    num_chunks = (n_points + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_points)
        
        chunk_result = rdr2geo_arrayfire_optimized(
            line=line[start_idx:end_idx],
            pixel=pixel[start_idx:end_idx],
            radar_grid=radar_grid,
            satellite_position=satellite_position,
            velocity=velocity,
            doppler=doppler,
            dem=dem[start_idx:end_idx] if dem.ndim == 1 else dem,
            ellipsoid=ellipsoid,
            wavelength_m=wavelength_m,
            look_side=look_side,
            max_iterations=max_iterations,
            extra_iterations=extra_iterations,
            threshold=threshold,
        )
        
        result_lat[start_idx:end_idx] = chunk_result[0]
        result_lon[start_idx:end_idx] = chunk_result[1]
        result_hgt[start_idx:end_idx] = chunk_result[2]
    
    return np.stack([result_lat, result_lon, result_hgt], axis=0)


def rdr2geo_arrayfire_bracket(
    aztime: Union[int, float, np.ndarray],
    slant_range: Union[int, float, np.ndarray],
    doppler: float,
    radar_grid: RadarGrid,
    satellite_position: np.ndarray,
    velocity: np.ndarray,
    dem: Union[float, np.ndarray] = 0.0,
    ellipsoid: Ellipsoid = WGS84,
    wavelength_m: float = 0.0565642,
    look_side: LookSide = LookSide.RIGHT,
    tol_height: float = 1e-5,
    look_min: float = 0.0,
    look_max: float = 1.5707963267948966,
) -> np.ndarray:
    from i2sar.math.root_find import find_zero_brent
    
    backend = ArrayFireBackend()
    if not backend.available:
        raise RuntimeError(backend.reason)
    
    aztime_arr = np.atleast_1d(np.asarray(aztime, dtype=np.float64))
    slant_range_arr = np.atleast_1d(np.asarray(slant_range, dtype=np.float64))
    n_points = len(aztime_arr)
    
    if n_points == 0:
        return np.array([[], [], []], dtype=np.float64)
    
    if isinstance(dem, (int, float)):
        dem_heights = np.full(n_points, float(dem), dtype=np.float64)
    elif dem.ndim == 1:
        dem_heights = np.asarray(dem, dtype=np.float64)
    else:
        raise ValueError("dem must be scalar or 1D array for rdr2geo_arrayfire_bracket")
    
    sat_pos = np.asarray(satellite_position, dtype=np.float64)
    sat_vel = np.asarray(velocity, dtype=np.float64)
    
    speed = float(np.linalg.norm(sat_vel))
    if speed < 1e-10:
        raise ValueError("Velocity magnitude too small")
    
    along_track = sat_vel / speed
    
    vel_cross_rad = np.cross(sat_vel, sat_pos)
    vel_cross_rad_norm = float(np.linalg.norm(vel_cross_rad))
    if vel_cross_rad_norm < 1e-10:
        raise ValueError("Satellite position and velocity are collinear")
    
    right = vel_cross_rad / vel_cross_rad_norm
    down = np.cross(along_track, right)
    
    if look_side == LookSide.LEFT:
        horizontal = -right
    else:
        horizontal = right
    
    sin_squint = doppler * wavelength_m / (2.0 * speed)
    cos_squint = np.sqrt(max(0.0, 1.0 - sin_squint * sin_squint))
    
    center_x = sat_pos[0] + sin_squint * slant_range_arr * along_track[0]
    center_y = sat_pos[1] + sin_squint * slant_range_arr * along_track[1]
    center_z = sat_pos[2] + sin_squint * slant_range_arr * along_track[2]
    
    radius_arr = cos_squint * slant_range_arr
    
    tol_look = tol_height / np.mean(radius_arr)
    
    a_ellipsoid = ellipsoid.a
    b_ellipsoid = ellipsoid.b
    e2_ellipsoid = ellipsoid.e2
    
    results_lon = np.full(n_points, np.nan)
    results_lat = np.full(n_points, np.nan)
    results_hgt = np.full(n_points, np.nan)
    
    for i in range(n_points):
        cx, cy, cz = center_x[i], center_y[i], center_z[i]
        radius = radius_arr[i]
        dem_h = dem_heights[i]
        
        def get_xyz(look_angle):
            hx = horizontal[0]
            hy = horizontal[1]
            hz = horizontal[2]
            dx = down[0]
            dy = down[1]
            dz = down[2]
            
            x = cx + radius * (np.sin(look_angle) * hx + np.cos(look_angle) * dx)
            y = cy + radius * (np.sin(look_angle) * hy + np.cos(look_angle) * dy)
            z = cz + radius * (np.sin(look_angle) * hz + np.cos(look_angle) * dz)
            return np.array([x, y, z])
        
        def xyz_to_llh(xyz):
            x, y, z = xyz[0], xyz[1], xyz[2]
            
            p = np.sqrt(x * x + y * y)
            lon = np.arctan2(y, x)
            
            theta = np.arctan2(z * a_ellipsoid, p * b_ellipsoid)
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            
            lat = np.arctan2(
                z + ellipsoid.ep2 * b_ellipsoid * sin_theta**3,
                p - e2_ellipsoid * a_ellipsoid * cos_theta**3
            )
            
            sin_lat = np.sin(lat)
            prime_vertical = a_ellipsoid / np.sqrt(1.0 - e2_ellipsoid * sin_lat * sin_lat)
            height = p / np.cos(lat) - prime_vertical
            
            return np.array([lat, lon, height])
        
        def height_error(look_angle):
            xyz = get_xyz(look_angle)
            llh = xyz_to_llh(xyz)
            return llh[2] - dem_h
        
        try:
            err, look_solution = find_zero_brent(look_min, look_max, height_error, tol=tol_look)
        except Exception:
            look_solution = (look_min + look_max) / 2.0
        
        xyz_solution = get_xyz(look_solution)
        llh_solution = xyz_to_llh(xyz_solution)
        
        results_lat[i] = np.degrees(llh_solution[0])
        results_lon[i] = np.degrees(llh_solution[1])
        results_hgt[i] = llh_solution[2]
    
    result = np.stack([results_lat, results_lon, results_hgt], axis=0)
    
    if n_points == 1:
        return result[:, 0]
    
    return result


rdr2geo_arrayfire_core = rdr2geo_arrayfire_optimized
rdr2geo_arrayfire_fast = rdr2geo_arrayfire_adaptive


class Geo2RdrGpuProcessor:
    """持久化的 Geo2Rdr GPU 处理器，支持数组复用和异步传输"""
    
    def __init__(self):
        self._backend = None
        self._af = None
        self._initialized = False
        self._cached_arrays = {}
        self._last_n_points = 0
    
    @property
    def available(self):
        if self._backend is None:
            self._backend = ArrayFireBackend()
        return self._backend.available
    
    def _init_if_needed(self):
        if not self._initialized:
            self._backend = ArrayFireBackend()
            if not self._backend.available:
                raise RuntimeError(self._backend.reason)
            self._af = self._backend.module
            self._initialized = True
    
    def _ensure_array_size(self, n_points):
        """确保缓存的数组大小足够"""
        if self._last_n_points >= n_points:
            return
        
        af = self._af
        
        arrays_to_create = [
            ('sat_x', n_points), ('sat_y', n_points), ('sat_z', n_points),
            ('vel_x', n_points), ('vel_y', n_points), ('vel_z', n_points),
            ('fdop', n_points), ('mid_time', n_points),
        ]
        
        for name, size in arrays_to_create:
            self._cached_arrays[name] = af.constant(0.0, size, dtype=af.Dtype.f64)
        
        self._last_n_points = n_points
    
    def process(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        height: np.ndarray,
        radar_grid: RadarGrid,
        satellite_position: np.ndarray,
        velocity: np.ndarray,
        doppler: float = 0.0,
        ellipsoid: Ellipsoid = WGS84,
        wavelength_m: float = 0.0565642,
        max_iterations: int = 50,
        threshold: float = 1e-8,
        look_side_right: bool = True,
    ) -> np.ndarray:
        self._init_if_needed()
        af = self._af
        
        lat_arr = np.asarray(lat, dtype=np.float64)
        lon_arr = np.asarray(lon, dtype=np.float64)
        h_arr = np.asarray(height, dtype=np.float64)
        n_points = len(lat_arr)
        
        if n_points == 0:
            return np.array([[], []], dtype=np.float64)
        
        self._ensure_array_size(n_points)
        
        lat_rad = lat_arr * np.pi / 180.0
        lon_rad = lon_arr * np.pi / 180.0
        
        a = ellipsoid.a
        e2 = ellipsoid.e2
        
        re = a / np.sqrt(1.0 - e2 * np.sin(lat_rad) ** 2)
        x_ecef = (re + h_arr) * np.cos(lat_rad) * np.cos(lon_rad)
        y_ecef = (re + h_arr) * np.cos(lat_rad) * np.sin(lon_rad)
        z_ecef = (re * (1.0 - e2) + h_arr) * np.sin(lat_rad)
        
        x_ecef_af = asarray(af, x_ecef)
        y_ecef_af = asarray(af, y_ecef)
        z_ecef_af = asarray(af, z_ecef)
        
        sat_x_af = self._cached_arrays['sat_x']
        sat_y_af = self._cached_arrays['sat_y']
        sat_z_af = self._cached_arrays['sat_z']
        vel_x_af = self._cached_arrays['vel_x']
        vel_y_af = self._cached_arrays['vel_y']
        vel_z_af = self._cached_arrays['vel_z']
        
        af.write(sat_x_af, af.constant(float(satellite_position[0]), n_points, dtype=af.Dtype.f64))
        af.write(sat_y_af, af.constant(float(satellite_position[1]), n_points, dtype=af.Dtype.f64))
        af.write(sat_z_af, af.constant(float(satellite_position[2]), n_points, dtype=af.Dtype.f64))
        af.write(vel_x_af, af.constant(float(velocity[0]), n_points, dtype=af.Dtype.f64))
        af.write(vel_y_af, af.constant(float(velocity[1]), n_points, dtype=af.Dtype.f64))
        af.write(vel_z_af, af.constant(float(velocity[2]), n_points, dtype=af.Dtype.f64))
        
        dr_x = x_ecef_af - sat_x_af
        dr_y = y_ecef_af - sat_y_af
        dr_z = z_ecef_af - sat_z_af
        
        slant_range_af = af.sqrt(dr_x * dr_x + dr_y * dr_y + dr_z * dr_z)
        
        mid_time = float(radar_grid.start_time + radar_grid.number_of_seconds / 2.0)
        aztime_af = af.constant(mid_time, n_points, dtype=af.Dtype.f64)
        
        cross_x_af = dr_y * vel_z_af - dr_z * vel_y_af
        cross_y_af = dr_z * vel_x_af - dr_x * vel_z_af
        cross_z_af = dr_x * vel_y_af - dr_y * vel_x_af
        
        dot_product_af = cross_x_af * sat_x_af + cross_y_af * sat_y_af + cross_z_af * sat_z_af
        is_right_side_af = dot_product_af > 0
        
        if look_side_right:
            valid_mask_af = is_right_side_af
        else:
            valid_mask_af = ~is_right_side_af
        
        fdop_val = 0.5 * wavelength_m * doppler
        fdop_af = af.constant(fdop_val, n_points, dtype=af.Dtype.f64)
        
        vel_dot_vel = vel_x_af * vel_x_af + vel_y_af * vel_y_af + vel_z_af * vel_z_af
        c1_af = -vel_dot_vel
        
        dt_af = af.constant(0.0, n_points, dtype=af.Dtype.f64)
        
        for iteration in range(max_iterations):
            aztime_af = aztime_af - dt_af
            
            dopfact_af = dr_x * vel_x_af + dr_y * vel_y_af + dr_z * vel_z_af
            c2_af = fdop_af / slant_range_af
            fnprime_af = c1_af + c2_af * dopfact_af
            
            fn_af = dopfact_af - fdop_af * slant_range_af
            
            dt_af = af.select(fnprime_af != 0, fn_af / fnprime_af, 0.0)
            dt_af = af.select(valid_mask_af, dt_af, 0.0)
            
            af.eval(dt_af)
            
            if iteration % 10 == 0 or iteration == max_iterations - 1:
                max_dt = float(af.max(af.abs(dt_af)))
                if max_dt < threshold:
                    break
        
        af.eval(aztime_af, slant_range_af)
        af.sync()
        
        aztime_np = to_numpy(aztime_af)
        range_np = to_numpy(slant_range_af)
        
        result = np.stack([aztime_np, range_np], axis=0)
        
        if n_points == 1:
            return result[:, 0]
        
        return result
    
    def clear_cache(self):
        self._cached_arrays.clear()
        self._last_n_points = 0


_geo2rdr_processor = None


def get_geo2rdr_processor() -> Geo2RdrGpuProcessor:
    """获取全局的 Geo2Rdr GPU 处理器实例"""
    global _geo2rdr_processor
    if _geo2rdr_processor is None:
        _geo2rdr_processor = Geo2RdrGpuProcessor()
    return _geo2rdr_processor


def geo2rdr_arrayfire_core_optimized(
    lat: Union[int, float, np.ndarray],
    lon: Union[int, float, np.ndarray],
    height: Union[int, float, np.ndarray],
    radar_grid: RadarGrid,
    satellite_position: Union[np.ndarray, OrbitInterpolator],
    velocity: Union[np.ndarray, None] = None,
    doppler: float = 0.0,
    ellipsoid: Ellipsoid = WGS84,
    wavelength_m: float = 0.0565642,
    max_iterations: int = 50,
    threshold: float = 1e-8,
    delta_range: float = 10.0,
    look_side_right: bool = True,
    use_processor: bool = True,
) -> np.ndarray:
    """优化版 geo2rdr，支持持久化处理器和异步传输"""
    
    if isinstance(satellite_position, OrbitInterpolator):
        return _geo2rdr_arrayfire_with_orbit_new(
            lat, lon, height, radar_grid, satellite_position,
            doppler, wavelength_m, max_iterations, threshold,
            look_side_right
        )
    
    if use_processor:
        processor = get_geo2rdr_processor()
        if processor.available:
            return processor.process(
                lat=np.atleast_1d(np.asarray(lat, dtype=np.float64)),
                lon=np.atleast_1d(np.asarray(lon, dtype=np.float64)),
                height=np.atleast_1d(np.asarray(height, dtype=np.float64)),
                radar_grid=radar_grid,
                satellite_position=np.asarray(satellite_position, dtype=np.float64),
                velocity=np.asarray(velocity, dtype=np.float64),
                doppler=doppler,
                ellipsoid=ellipsoid,
                wavelength_m=wavelength_m,
                max_iterations=max_iterations,
                threshold=threshold,
                look_side_right=look_side_right,
            )
    
    return geo2rdr_arrayfire_core_fallback(
        lat, lon, height, radar_grid, satellite_position, velocity,
        doppler, ellipsoid, wavelength_m, max_iterations, threshold,
        delta_range, look_side_right
    )


def geo2rdr_arrayfire_core_fallback(
    lat: Union[int, float, np.ndarray],
    lon: Union[int, float, np.ndarray],
    height: Union[int, float, np.ndarray],
    radar_grid: RadarGrid,
    satellite_position: Union[np.ndarray, OrbitInterpolator],
    velocity: Union[np.ndarray, None] = None,
    doppler: float = 0.0,
    ellipsoid: Ellipsoid = WGS84,
    wavelength_m: float = 0.0565642,
    max_iterations: int = 50,
    threshold: float = 1e-8,
    delta_range: float = 10.0,
    look_side_right: bool = True,
) -> np.ndarray:
    """原始实现作为 fallback"""
    backend = ArrayFireBackend()
    if not backend.available:
        raise RuntimeError(backend.reason)
    af = backend.module
    
    lat_arr = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    lon_arr = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    h_arr = np.atleast_1d(np.asarray(height, dtype=np.float64))
    n_points = len(lat_arr)
    
    if n_points == 0:
        return np.array([[], []], dtype=np.float64)
    
    lat_rad = lat_arr * np.pi / 180.0
    lon_rad = lon_arr * np.pi / 180.0
    
    a = ellipsoid.a
    e2 = ellipsoid.e2
    
    re = a / np.sqrt(1.0 - e2 * np.sin(lat_rad) ** 2)
    x_ecef = (re + h_arr) * np.cos(lat_rad) * np.cos(lon_rad)
    y_ecef = (re + h_arr) * np.cos(lat_rad) * np.sin(lon_rad)
    z_ecef = (re * (1.0 - e2) + h_arr) * np.sin(lat_rad)
    
    x_ecef_af = asarray(af, x_ecef)
    y_ecef_af = asarray(af, y_ecef)
    z_ecef_af = asarray(af, z_ecef)
    
    sat_pos_x_af = af.constant(float(satellite_position[0]), n_points, dtype=af.Dtype.f64)
    sat_pos_y_af = af.constant(float(satellite_position[1]), n_points, dtype=af.Dtype.f64)
    sat_pos_z_af = af.constant(float(satellite_position[2]), n_points, dtype=af.Dtype.f64)
    
    vel_x_af = af.constant(float(velocity[0]), n_points, dtype=af.Dtype.f64)
    vel_y_af = af.constant(float(velocity[1]), n_points, dtype=af.Dtype.f64)
    vel_z_af = af.constant(float(velocity[2]), n_points, dtype=af.Dtype.f64)
    
    dr_x = x_ecef_af - sat_pos_x_af
    dr_y = y_ecef_af - sat_pos_y_af
    dr_z = z_ecef_af - sat_pos_z_af
    
    slant_range_af = af.sqrt(dr_x * dr_x + dr_y * dr_y + dr_z * dr_z)
    
    mid_time = float(radar_grid.start_time + radar_grid.number_of_seconds / 2.0)
    aztime_af = af.constant(mid_time, n_points, dtype=af.Dtype.f64)
    
    cross_x_af = dr_y * vel_z_af - dr_z * vel_y_af
    cross_y_af = dr_z * vel_x_af - dr_x * vel_z_af
    cross_z_af = dr_x * vel_y_af - dr_y * vel_x_af
    
    dot_product_af = cross_x_af * sat_pos_x_af + cross_y_af * sat_pos_y_af + cross_z_af * sat_pos_z_af
    is_right_side_af = dot_product_af > 0
    
    if look_side_right:
        valid_mask_af = is_right_side_af
    else:
        valid_mask_af = ~is_right_side_af
    
    fdop_val = 0.5 * wavelength_m * doppler
    fdop_af = af.constant(fdop_val, n_points, dtype=af.Dtype.f64)
    
    vel_dot_vel = vel_x_af * vel_x_af + vel_y_af * vel_y_af + vel_z_af * vel_z_af
    c1_af = -vel_dot_vel
    
    dt_af = af.constant(0.0, n_points, dtype=af.Dtype.f64)
    
    for iteration in range(max_iterations):
        aztime_af = aztime_af - dt_af
        
        dopfact_af = dr_x * vel_x_af + dr_y * vel_y_af + dr_z * vel_z_af
        c2_af = fdop_af / slant_range_af
        fnprime_af = c1_af + c2_af * dopfact_af
        
        fn_af = dopfact_af - fdop_af * slant_range_af
        
        dt_af = af.select(fnprime_af != 0, fn_af / fnprime_af, 0.0)
        dt_af = af.select(valid_mask_af, dt_af, 0.0)
        
        af.eval(dt_af)
        
        if iteration % 10 == 0 or iteration == max_iterations - 1:
            max_dt = float(af.max(af.abs(dt_af)))
            if max_dt < threshold:
                break
    
    af.eval(aztime_af, slant_range_af)
    af.sync()
    
    aztime_np = to_numpy(aztime_af)
    range_np = to_numpy(slant_range_af)
    
    result = np.stack([aztime_np, range_np], axis=0)
    
    if n_points == 1:
        return result[:, 0]
    
    return result


def _geo2rdr_arrayfire_with_orbit_new(
    lat, lon, height, radar_grid: RadarGrid,
    orbit: OrbitInterpolator, doppler: float,
    wavelength_m: float, max_iterations: int,
    threshold: float, look_side_right: bool = True
) -> np.ndarray:
    """优化版轨道插值器处理，减少数据传输"""
    backend = ArrayFireBackend()
    if not backend.available:
        raise RuntimeError(backend.reason)
    af = backend.module
    
    lat_arr = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    lon_arr = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    h_arr = np.atleast_1d(np.asarray(height, dtype=np.float64))
    n_points = len(lat_arr)
    
    if n_points == 0:
        return np.array([[], []], dtype=np.float64)
    
    lat_rad = lat_arr * np.pi / 180.0
    lon_rad = lon_arr * np.pi / 180.0
    
    a = WGS84.a
    e2 = WGS84.e2
    
    re = a / np.sqrt(1.0 - e2 * np.sin(lat_rad) ** 2)
    x_ecef = (re + h_arr) * np.cos(lat_rad) * np.cos(lon_rad)
    y_ecef = (re + h_arr) * np.cos(lat_rad) * np.sin(lon_rad)
    z_ecef = (re * (1.0 - e2) + h_arr) * np.sin(lat_rad)
    
    x_ecef_af = asarray(af, x_ecef)
    y_ecef_af = asarray(af, y_ecef)
    z_ecef_af = asarray(af, z_ecef)
    
    sat_x_af = af.constant(0.0, n_points, dtype=af.Dtype.f64)
    sat_y_af = af.constant(0.0, n_points, dtype=af.Dtype.f64)
    sat_z_af = af.constant(0.0, n_points, dtype=af.Dtype.f64)
    vel_x_af = af.constant(0.0, n_points, dtype=af.Dtype.f64)
    vel_y_af = af.constant(0.0, n_points, dtype=af.Dtype.f64)
    vel_z_af = af.constant(0.0, n_points, dtype=af.Dtype.f64)
    
    t_az_af = af.constant(
        radar_grid.start_time + radar_grid.number_of_seconds / 2.0,
        n_points, dtype=af.Dtype.f64
    )
    
    fdop_val = 0.5 * wavelength_m * doppler
    fdop_af = af.constant(fdop_val, n_points, dtype=af.Dtype.f64)
    
    t_az_np = to_numpy(t_az_af)
    orbit_state = orbit.state_at(t_az_np, allow_extrapolation=True)
    sat_pos = orbit_state.position
    sat_vel = orbit_state.velocity
    
    af.write(sat_x_af, asarray(af, sat_pos[:, 0]))
    af.write(sat_y_af, asarray(af, sat_pos[:, 1]))
    af.write(sat_z_af, asarray(af, sat_pos[:, 2]))
    af.write(vel_x_af, asarray(af, sat_vel[:, 0]))
    af.write(vel_y_af, asarray(af, sat_vel[:, 1]))
    af.write(vel_z_af, asarray(af, sat_vel[:, 2]))
    
    dr_x = x_ecef_af - sat_x_af
    dr_y = y_ecef_af - sat_y_af
    dr_z = z_ecef_af - sat_z_af
    
    slant_range_af = af.sqrt(dr_x * dr_x + dr_y * dr_y + dr_z * dr_z)
    
    cross_x_af = dr_y * vel_z_af - dr_z * vel_y_af
    cross_y_af = dr_z * vel_x_af - dr_x * vel_z_af
    cross_z_af = dr_x * vel_y_af - dr_y * vel_x_af
    
    dot_product_af = cross_x_af * sat_x_af + cross_y_af * sat_y_af + cross_z_af * sat_z_af
    is_right_side_af = dot_product_af > 0
    
    if look_side_right:
        valid_mask_af = is_right_side_af
    else:
        valid_mask_af = ~is_right_side_af
    
    vel_mag_sq_af = vel_x_af * vel_x_af + vel_y_af * vel_y_af + vel_z_af * vel_z_af
    c1_af = -vel_mag_sq_af
    
    for iteration in range(max_iterations):
        dot_dr_vel = dr_x * vel_x_af + dr_y * vel_y_af + dr_z * vel_z_af
        
        c2_af = fdop_af / slant_range_af
        fnprime_af = c1_af + c2_af * dot_dr_vel
        
        fn_af = dot_dr_vel - fdop_af * slant_range_af
        
        dt_af = af.select(fnprime_af != 0, fn_af / fnprime_af, 0.0)
        dt_af = af.select(valid_mask_af, dt_af, 0.0)
        
        t_az_af = t_az_af - dt_af
        
        af.eval(dt_af)
        
        if iteration % 10 == 0 or iteration == max_iterations - 1:
            max_dt = float(af.max(af.abs(dt_af)))
            if max_dt < threshold:
                break
        
        t_az_np = to_numpy(t_az_af)
        orbit_state = orbit.state_at(t_az_np, allow_extrapolation=True)
        sat_pos = orbit_state.position
        sat_vel = orbit_state.velocity
        
        af.write(sat_x_af, asarray(af, sat_pos[:, 0]))
        af.write(sat_y_af, asarray(af, sat_pos[:, 1]))
        af.write(sat_z_af, asarray(af, sat_pos[:, 2]))
        af.write(vel_x_af, asarray(af, sat_vel[:, 0]))
        af.write(vel_y_af, asarray(af, sat_vel[:, 1]))
        af.write(vel_z_af, asarray(af, sat_vel[:, 2]))
        
        dr_x = x_ecef_af - sat_x_af
        dr_y = y_ecef_af - sat_y_af
        dr_z = z_ecef_af - sat_z_af
        
        slant_range_af = af.sqrt(dr_x * dr_x + dr_y * dr_y + dr_z * dr_z)
    
    af.eval(t_az_af, slant_range_af)
    af.sync()
    
    aztime_np = to_numpy(t_az_af)
    range_np = to_numpy(slant_range_af)
    
    result = np.stack([aztime_np, range_np], axis=0)
    
    if n_points == 1:
        return result[:, 0]
    
    return result


def geo2rdr_arrayfire_chunked_optimized(
    lat: np.ndarray,
    lon: np.ndarray,
    height: np.ndarray,
    radar_grid: RadarGrid,
    satellite_position: Union[np.ndarray, OrbitInterpolator],
    velocity: Union[np.ndarray, None] = None,
    doppler: float = 0.0,
    ellipsoid: Ellipsoid = WGS84,
    wavelength_m: float = 0.0565642,
    max_iterations: int = 50,
    threshold: float = 1e-8,
    chunk_size: int = 65536,
) -> np.ndarray:
    """优化版分块处理，使用持久化处理器"""
    if isinstance(satellite_position, OrbitInterpolator):
        return geo2rdr_arrayfire_chunked_fallback(
            lat, lon, height, radar_grid, satellite_position, velocity,
            doppler, ellipsoid, wavelength_m, max_iterations, threshold, chunk_size
        )
    
    processor = get_geo2rdr_processor()
    if not processor.available:
        return geo2rdr_arrayfire_chunked_fallback(
            lat, lon, height, radar_grid, satellite_position, velocity,
            doppler, ellipsoid, wavelength_m, max_iterations, threshold, chunk_size
        )
    
    n_points = len(lat)
    result_az = np.zeros(n_points, dtype=np.float64)
    result_range = np.zeros(n_points, dtype=np.float64)
    
    num_chunks = (n_points + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_points)
        
        chunk_result = processor.process(
            lat=lat[start_idx:end_idx],
            lon=lon[start_idx:end_idx],
            height=height[start_idx:end_idx],
            radar_grid=radar_grid,
            satellite_position=np.asarray(satellite_position, dtype=np.float64),
            velocity=np.asarray(velocity, dtype=np.float64),
            doppler=doppler,
            ellipsoid=ellipsoid,
            wavelength_m=wavelength_m,
            max_iterations=max_iterations,
            threshold=threshold,
            look_side_right=True,
        )
        
        result_az[start_idx:end_idx] = chunk_result[0]
        result_range[start_idx:end_idx] = chunk_result[1]
    
    return np.stack([result_az, result_range], axis=0)


def geo2rdr_arrayfire_chunked_fallback(
    lat: np.ndarray,
    lon: np.ndarray,
    height: np.ndarray,
    radar_grid: RadarGrid,
    satellite_position: Union[np.ndarray, OrbitInterpolator],
    velocity: Union[np.ndarray, None] = None,
    doppler: float = 0.0,
    ellipsoid: Ellipsoid = WGS84,
    wavelength_m: float = 0.0565642,
    max_iterations: int = 50,
    threshold: float = 1e-8,
    chunk_size: int = 65536,
) -> np.ndarray:
    """分块处理的 fallback 版本"""
    backend = ArrayFireBackend()
    if not backend.available:
        raise RuntimeError(backend.reason)
    
    n_points = len(lat)
    result_az = np.zeros(n_points, dtype=np.float64)
    result_range = np.zeros(n_points, dtype=np.float64)
    
    num_chunks = (n_points + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_points)
        
        chunk_result = geo2rdr_arrayfire_core_fallback(
            lat=lat[start_idx:end_idx],
            lon=lon[start_idx:end_idx],
            height=height[start_idx:end_idx],
            radar_grid=radar_grid,
            satellite_position=satellite_position,
            velocity=velocity,
            doppler=doppler,
            ellipsoid=ellipsoid,
            wavelength_m=wavelength_m,
            max_iterations=max_iterations,
            threshold=threshold,
        )
        
        result_az[start_idx:end_idx] = chunk_result[0]
        result_range[start_idx:end_idx] = chunk_result[1]
    
    return np.stack([result_az, result_range], axis=0)
geo2rdr_arrayfire_core = geo2rdr_arrayfire_core_fallback
