import numpy as np
from typing import Union, Optional, Tuple
from i2sar.core.enums import LookSide
from i2sar.geometry.radar_grid import RadarGrid
from i2sar.orbit.orbit_data import OrbitData
from i2sar.orbit.interpolate import OrbitInterpolator


def llh_to_ecef(lat: np.ndarray, lon: np.ndarray, h: np.ndarray, 
                a: float = 6378137.0, b: float = 6356752.314245) -> np.ndarray:
    """大地坐标转 ECEF"""
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    N = a / np.sqrt(1 - (1 - (b/a)**2) * sin_lat**2)
    
    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (b/a)**2 + h) * sin_lat
    
    return np.stack([x, y, z], axis=-1)


def ecef_to_llh(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                a: float = 6378137.0, b: float = 6356752.314245) -> np.ndarray:
    """ECEF 转大地坐标"""
    e2 = 1 - (b/a)**2
    
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * a, p * b)
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    lat = np.arctan2(z + e2 * b * sin_theta**3, p - e2 * a * cos_theta**3)
    lon = np.arctan2(y, x)
    
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    h = p / np.cos(lat) - N
    
    return np.stack([np.rad2deg(lat), np.rad2deg(lon), h], axis=-1)


def _check_look_side(rvec: np.ndarray, vel: np.ndarray, pos: np.ndarray, 
                     look_side_right: bool) -> bool:
    """检查观察侧是否匹配"""
    cross_prod = np.cross(rvec, vel)
    dot_prod = np.dot(cross_prod, pos)
    is_right = dot_prod > 0
    return is_right == look_side_right


def _compute_doppler_residual(t_az: np.ndarray, target_ecef: np.ndarray, 
                              orbit_data: OrbitData, doppler: float, 
                              wavelength: float, interp_method: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算多普勒残差和相关量"""
    sat_pos, sat_vel = orbit_data.interpolate(t_az, method=interp_method)
    
    rvec = target_ecef - sat_pos
    slant_range = np.linalg.norm(rvec, axis=-1)
    
    dopfact = np.einsum('...i,...i', rvec, sat_vel)
    fdop = 0.5 * wavelength * doppler
    
    fn = dopfact - fdop * slant_range
    c1 = -np.einsum('...i,...i', sat_vel, sat_vel)
    c2 = fdop / (slant_range + 1e-12)
    fnprime = c1 + c2 * dopfact
    
    return fn, fnprime, slant_range, sat_pos, sat_vel


def _find_closest_aztime_batch(target_ecef: np.ndarray, sat_positions: np.ndarray, 
                                sat_velocities: np.ndarray, sat_times: np.ndarray,
                                look_side_right: bool) -> np.ndarray:
    """
    批量查找最近轨道点的时间
    
    Args:
        target_ecef: 目标点 ECEF 坐标 (N, 3)
        sat_positions: 卫星位置数组 (M, 3)
        sat_velocities: 卫星速度数组 (M, 3)
        sat_times: 卫星时间数组 (M,)
        look_side_right: 是否为右侧观察
    
    Returns:
        closest_times: 最近点时间数组 (N,)
    """
    n_points = len(target_ecef)
    n_sat = len(sat_positions)
    
    closest_times = np.full(n_points, (sat_times[0] + sat_times[-1]) / 2.0)
    
    for i in range(n_points):
        target = target_ecef[i]
        r_closest = 1e16
        idx_closest = 0
        valid_found = False
        
        for k in range(n_sat):
            pos = sat_positions[k]
            vel = sat_velocities[k]
            rvec = target - pos
            r = np.linalg.norm(rvec)
            
            cross_prod = np.cross(rvec, vel)
            dot_prod = np.dot(cross_prod, pos)
            is_valid = (dot_prod > 0) == look_side_right
            
            if is_valid:
                valid_found = True
                if r < r_closest:
                    r_closest = r
                    idx_closest = k
        
        if valid_found:
            closest_times[i] = sat_times[idx_closest]
    
    return closest_times


def _geo2rdr_single_batch(lat_arr: np.ndarray, lon_arr: np.ndarray, h_arr: np.ndarray,
                          orbit_data: OrbitData, radar_grid: RadarGrid,
                          doppler: float, wavelength: float, look_side: LookSide,
                          max_iterations: int = 50, threshold: float = 1e-8,
                          interp_method: str = "hermite") -> Tuple[np.ndarray, np.ndarray]:
    """
    批量 geo2rdr 计算（纯 Python 实现，用于参考）
    
    Args:
        lat_arr: 纬度数组 (N,)
        lon_arr: 经度数组 (N,)
        h_arr: 高度数组 (N,)
        orbit_data: 轨道数据对象
        radar_grid: 雷达网格参数
        doppler: 多普勒中心频率 (Hz)
        wavelength: 波长 (m)
        look_side: 观察侧
        max_iterations: 最大迭代次数
        threshold: 收敛阈值
        interp_method: 插值方法 ('linear', 'hermite')
    
    Returns:
        aztimes: 方位时间数组 (N,)
        slant_ranges: 斜距数组 (N,)
    """
    n_points = len(lat_arr)
    
    target_ecef = llh_to_ecef(lat_arr, lon_arr, h_arr)
    
    look_side_right = (look_side == LookSide.RIGHT)
    t_az = _find_closest_aztime_batch(
        target_ecef, orbit_data.positions, orbit_data.velocities, 
        orbit_data.times, look_side_right
    )
    
    aztimes = np.full(n_points, np.nan)
    slant_ranges = np.full(n_points, np.nan)
    valid_mask = np.ones(n_points, dtype=bool)
    
    for iteration in range(max_iterations):
        fn, fnprime, slant_range, sat_pos, sat_vel = _compute_doppler_residual(
            t_az[valid_mask], target_ecef[valid_mask], orbit_data, 
            doppler, wavelength, interp_method
        )
        
        dt = np.where(fnprime != 0, fn / fnprime, 0.0)
        
        t_az[valid_mask] -= dt
        
        t_az = np.clip(t_az, orbit_data.t_min, orbit_data.t_max)
        
        converged = np.abs(dt) < threshold
        newly_converged = valid_mask & converged
        
        if np.any(newly_converged):
            aztimes[newly_converged] = t_az[newly_converged]
            slant_ranges[newly_converged] = slant_range[converged[newly_converged.nonzero()[0]]]
            valid_mask[newly_converged] = False
        
        if not np.any(valid_mask):
            break
    
    if np.any(valid_mask):
        fn, fnprime, slant_range, sat_pos, sat_vel = _compute_doppler_residual(
            t_az[valid_mask], target_ecef[valid_mask], orbit_data, 
            doppler, wavelength, interp_method
        )
        aztimes[valid_mask] = t_az[valid_mask]
        slant_ranges[valid_mask] = slant_range
    
    return aztimes, slant_ranges


def geo2rdr_unified(lat: Union[float, np.ndarray], 
                    lon: Union[float, np.ndarray],
                    height: Union[float, np.ndarray],
                    radar_grid: RadarGrid,
                    satellite_position: Union[np.ndarray, OrbitInterpolator],
                    velocity: Optional[np.ndarray] = None,
                    doppler: float = 0.0,
                    wavelength_m: float = 0.0565642,
                    look_side: LookSide = LookSide.RIGHT,
                    max_iterations: int = 50,
                    threshold: float = 1e-8,
                    interp_method: str = "hermite",
                    method: str = "python") -> Tuple[np.ndarray, np.ndarray]:
    """
    统一的 geo2rdr 接口，支持批量计算和多种插值算法
    
    Args:
        lat: 纬度 (度)，支持标量或数组
        lon: 经度 (度)，支持标量或数组
        height: 高度 (米)，支持标量或数组
        radar_grid: 雷达网格参数
        satellite_position: 卫星位置（静态位置或轨道插值器）
        velocity: 卫星速度（仅静态轨道时使用）
        doppler: 多普勒中心频率 (Hz)
        wavelength_m: 波长 (m)，默认 Sentinel-1 C波段
        look_side: 观察侧 (LEFT/RIGHT)
        max_iterations: 最大迭代次数
        threshold: 收敛阈值 (秒)
        interp_method: 轨道插值方法 ('linear', 'hermite')
        method: 计算后端 ('python', 'numba')
    
    Returns:
        aztimes: 方位时间数组
        slant_ranges: 斜距数组
    
    Examples:
        >>> # 单点计算
        >>> aztime, slant_range = geo2rdr_unified(30.0, 120.0, 100.0, radar_grid, orbit)
        
        >>> # 批量计算
        >>> lats = np.array([30.0, 31.0, 32.0])
        >>> lons = np.array([120.0, 121.0, 122.0])
        >>> heights = np.array([100.0, 200.0, 300.0])
        >>> aztimes, slant_ranges = geo2rdr_unified(lats, lons, heights, radar_grid, orbit)
    """
    lat_arr = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    lon_arr = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    h_arr = np.atleast_1d(np.asarray(height, dtype=np.float64))
    
    if len(lat_arr) != len(lon_arr) or len(lat_arr) != len(h_arr):
        raise ValueError("lat, lon, height must have the same length")
    
    if isinstance(satellite_position, OrbitInterpolator):
        orbit_data = OrbitData.from_orbit_interpolator(satellite_position)
    else:
        if velocity is None:
            raise ValueError("velocity must be provided for static orbit")
        orbit_data = OrbitData.from_static_orbit(satellite_position, velocity)
    
    look_side_right = (look_side == LookSide.RIGHT)
    
    if method.lower() == "python":
        aztimes, slant_ranges = _geo2rdr_single_batch(
            lat_arr, lon_arr, h_arr, orbit_data, radar_grid,
            doppler, wavelength_m, look_side,
            max_iterations, threshold, interp_method
        )
    elif method.lower() == "numba":
        try:
            from .geo2rdr_numba import geo2rdr_numba_parallel
            aztimes, slant_ranges = geo2rdr_numba_parallel(
                lat_arr=lat_arr,
                lon_arr=lon_arr,
                h_arr=h_arr,
                sat_positions=orbit_data.positions,
                sat_velocities=orbit_data.velocities,
                sat_times=orbit_data.times,
                doppler=float(doppler),
                wavelength=float(wavelength_m),
                a=6378137.0,
                b=6356752.314245,
                max_iterations=max_iterations,
                threshold=threshold,
                sensing_start_s=float(radar_grid.sensing_start_s),
                prf_hz=float(radar_grid.prf_hz),
                length=int(radar_grid.length),
                look_side_right=look_side_right
            )
        except ImportError:
            aztimes, slant_ranges = _geo2rdr_single_batch(
                lat_arr, lon_arr, h_arr, orbit_data, radar_grid,
                doppler, wavelength_m, look_side,
                max_iterations, threshold, interp_method
            )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if len(lat_arr) == 1:
        return aztimes[0], slant_ranges[0]
    
    return aztimes, slant_ranges