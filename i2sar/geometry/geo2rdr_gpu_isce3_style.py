"""
ISCE3风格的GPU geo2rdr实现，支持设备端轨道插值
"""
import numpy as np
from typing import Union, Tuple
from i2sar.accel.arrayfire_backend import ArrayFireBackend, asarray, to_numpy
from i2sar.core.enums import LookSide
from i2sar.geometry.ellipsoid import WGS84
from i2sar.geometry.radar_grid import RadarGrid
from i2sar.orbit.interpolate import OrbitInterpolator


def _find_closest_valid_orbit_points(
    x_ecef: np.ndarray, y_ecef: np.ndarray, z_ecef: np.ndarray,
    sat_positions: np.ndarray, sat_velocities: np.ndarray,
    sat_times: np.ndarray, look_side_right: bool
) -> np.ndarray:
    """
    向量化查找最近的有效轨道点（考虑观察侧）
    
    Args:
        x_ecef, y_ecef, z_ecef: 目标点 ECEF 坐标数组 (N,)
        sat_positions: 卫星位置数组 (M, 3)
        sat_velocities: 卫星速度数组 (M, 3)
        sat_times: 卫星时间数组 (M,)
        look_side_right: 是否为右侧观察
    
    Returns:
        closest_times: 最近有效轨道点的时间数组 (N,)
    """
    n_points = len(x_ecef)
    n_sat = len(sat_times)
    
    x_ecef_2d = x_ecef[:, np.newaxis]
    y_ecef_2d = y_ecef[:, np.newaxis]
    z_ecef_2d = z_ecef[:, np.newaxis]
    
    dx = x_ecef_2d - sat_positions[:, 0]
    dy = y_ecef_2d - sat_positions[:, 1]
    dz = z_ecef_2d - sat_positions[:, 2]
    
    dist_sq = dx ** 2 + dy ** 2 + dz ** 2
    
    cross_x = dy * sat_velocities[:, 2] - dz * sat_velocities[:, 1]
    cross_y = dz * sat_velocities[:, 0] - dx * sat_velocities[:, 2]
    cross_z = dx * sat_velocities[:, 0] - dy * sat_velocities[:, 1]
    
    dot_prod = cross_x * sat_positions[:, 0] + cross_y * sat_positions[:, 1] + cross_z * sat_positions[:, 2]
    is_right = dot_prod > 0
    
    valid_mask = is_right if look_side_right else ~is_right
    
    invalid_dist = np.full(dist_sq.shape, np.inf)
    valid_dist_sq = np.where(valid_mask, dist_sq, invalid_dist)
    
    closest_idx = np.argmin(valid_dist_sq, axis=1)
    
    has_valid = np.any(valid_mask, axis=1)
    fallback_time = (sat_times[0] + sat_times[-1]) / 2.0
    closest_times = np.where(has_valid, sat_times[closest_idx], fallback_time)
    
    return closest_times


def geo2rdr_gpu_isce3_style(
    lat: np.ndarray,
    lon: np.ndarray,
    height: np.ndarray,
    radar_grid: RadarGrid,
    satellite_position: OrbitInterpolator,
    doppler: float = 0.0,
    wavelength_m: float = 0.0565642,
    max_iterations: int = 50,
    threshold: float = 1e-8,
    look_side: LookSide = LookSide.RIGHT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ISCE3风格的GPU geo2rdr实现
    
    关键特点：
    1. 使用向量化方式找到最近轨道点作为初始估计（考虑观察侧）
    2. 在GPU上执行牛顿迭代的计算密集部分
    
    Args:
        lat: 纬度数组 (度)
        lon: 经度数组 (度)
        height: 高度数组 (米)
        radar_grid: 雷达网格参数
        satellite_position: 轨道插值器
        doppler: 多普勒中心频率 (Hz)
        wavelength_m: 波长 (米)
        max_iterations: 最大迭代次数
        threshold: 收敛阈值
        look_side: 观察侧
    
    Returns:
        aztimes: 方位时间数组
        slant_ranges: 斜距数组
    """
    backend = ArrayFireBackend()
    if not backend.available:
        raise RuntimeError(backend.reason)
    af = backend.module

    n_points = len(lat)
    if n_points == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    lat_rad = lat * np.pi / 180.0
    lon_rad = lon * np.pi / 180.0

    a = WGS84.a
    e2 = WGS84.e2

    re = a / np.sqrt(1.0 - e2 * np.sin(lat_rad) ** 2)
    x_ecef = (re + height) * np.cos(lat_rad) * np.cos(lon_rad)
    y_ecef = (re + height) * np.cos(lat_rad) * np.sin(lon_rad)
    z_ecef = (re * (1.0 - e2) + height) * np.sin(lat_rad)

    x_ecef_af = asarray(af, x_ecef)
    y_ecef_af = asarray(af, y_ecef)
    z_ecef_af = asarray(af, z_ecef)

    sat_times = satellite_position.time
    sat_positions = satellite_position.position
    sat_velocities = satellite_position.velocity

    look_side_right = (look_side == LookSide.RIGHT)
    
    t_az_np = _find_closest_valid_orbit_points(
        x_ecef, y_ecef, z_ecef,
        sat_positions, sat_velocities,
        sat_times, look_side_right
    )

    look_sign = 1.0 if look_side == LookSide.RIGHT else -1.0
    fdop_val = 0.5 * wavelength_m * doppler

    fdop_af = af.constant(fdop_val, n_points, dtype=af.Dtype.f64)
    look_sign_af = af.constant(look_sign, n_points, dtype=af.Dtype.f64)

    for iteration in range(max_iterations):
        orbit_states = satellite_position.state_at(t_az_np, allow_extrapolation=True)
        sat_pos_np = orbit_states.position
        sat_vel_np = orbit_states.velocity

        sat_x_af = asarray(af, sat_pos_np[:, 0])
        sat_y_af = asarray(af, sat_pos_np[:, 1])
        sat_z_af = asarray(af, sat_pos_np[:, 2])
        vel_x_af = asarray(af, sat_vel_np[:, 0])
        vel_y_af = asarray(af, sat_vel_np[:, 1])
        vel_z_af = asarray(af, sat_vel_np[:, 2])

        dr_x_af = x_ecef_af - sat_x_af
        dr_y_af = y_ecef_af - sat_y_af
        dr_z_af = z_ecef_af - sat_z_af

        slant_range_af = af.sqrt(dr_x_af * dr_x_af + dr_y_af * dr_y_af + dr_z_af * dr_z_af)

        cross_x_af = dr_y_af * vel_z_af - dr_z_af * vel_y_af
        cross_y_af = dr_z_af * vel_x_af - dr_x_af * vel_z_af
        cross_z_af = dr_x_af * vel_y_af - dr_y_af * vel_x_af

        dot_look_af = cross_x_af * sat_x_af + cross_y_af * sat_y_af + cross_z_af * sat_z_af
        look_valid_af = (dot_look_af * look_sign_af) > 0

        dopfact_af = dr_x_af * vel_x_af + dr_y_af * vel_y_af + dr_z_af * vel_z_af

        vel_dot_vel_af = vel_x_af * vel_x_af + vel_y_af * vel_y_af + vel_z_af * vel_z_af
        c1_af = -vel_dot_vel_af
        c2_af = fdop_af / slant_range_af
        fnprime_af = c1_af + c2_af * dopfact_af

        fn_af = dopfact_af - fdop_af * slant_range_af

        dt_af = af.select(af.abs(fnprime_af) > 1e-12, fn_af / fnprime_af, 0.0)
        dt_af = af.select(look_valid_af, dt_af, 0.0)

        t_az_af = asarray(af, t_az_np) - dt_af
        t_az_np = to_numpy(t_az_af)

        t_az_np = np.clip(t_az_np, sat_times[0], sat_times[-1])

        af.eval(t_az_af)

        max_dt = float(af.max(af.abs(dt_af)))
        if max_dt < threshold:
            break

    orbit_states = satellite_position.state_at(t_az_np, allow_extrapolation=True)
    sat_pos_np = orbit_states.position

    dr_x = x_ecef - sat_pos_np[:, 0]
    dr_y = y_ecef - sat_pos_np[:, 1]
    dr_z = z_ecef - sat_pos_np[:, 2]
    slant_range_np = np.sqrt(dr_x**2 + dr_y**2 + dr_z**2)

    af.sync()

    return t_az_np, slant_range_np