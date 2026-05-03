from __future__ import annotations

import multiprocessing as mp
from typing import Union, Tuple, Optional

import numpy as np

from i2sar.core.enums import LookSide
from i2sar.geometry.ellipsoid import Ellipsoid, WGS84, ecef_to_llh, llh_to_ecef, lon_lat_to_xyz
from i2sar.geometry.radar_grid import RadarGrid
from i2sar.geometry.doppler_lut import DopplerLUT2d
from i2sar.geometry.look_side import check_look_side
from i2sar.orbit import OrbitInterpolator
from i2sar.math.root_find import find_zero_brent


def _validate_geo_coordinates(lat_arr: np.ndarray, lon_arr: np.ndarray, h_arr: np.ndarray) -> np.ndarray:
    """
    验证地理坐标的有效性，返回有效索引数组
    """
    valid_mask = np.ones(len(lat_arr), dtype=bool)
    
    valid_mask = np.logical_and(valid_mask, np.isfinite(lat_arr))
    valid_mask = np.logical_and(valid_mask, np.isfinite(lon_arr))
    valid_mask = np.logical_and(valid_mask, np.isfinite(h_arr))
    
    valid_mask = np.logical_and(valid_mask, lat_arr >= -90.0)
    valid_mask = np.logical_and(valid_mask, lat_arr <= 90.0)
    valid_mask = np.logical_and(valid_mask, lon_arr >= -180.0)
    valid_mask = np.logical_and(valid_mask, lon_arr <= 180.0)
    valid_mask = np.logical_and(valid_mask, h_arr >= -1000.0)
    valid_mask = np.logical_and(valid_mask, h_arr <= 10000.0)
    
    return valid_mask


def _make_doppler_lut(doppler_value: float, radar_grid: RadarGrid) -> DopplerLUT2d:
    return DopplerLUT2d(
        azimuth_axis=np.array([radar_grid.start_time, radar_grid.end_time]),
        range_axis=np.array([radar_grid.start_range, radar_grid.end_range]),
        data=np.full((2, 2), doppler_value),
    )


def _compute_doppler_aztime_diff(
    rvec: np.ndarray,
    vel: np.ndarray,
    doppler: DopplerLUT2d,
    wavelength: float,
    azimuth_time: float,
    slant_range: float,
    delta_range: float,
) -> float:
    dopfact = np.dot(rvec, vel)
    fdop = 0.5 * wavelength * doppler.eval(azimuth_time, slant_range)

    fdop_der = (0.5 * wavelength * doppler.eval(azimuth_time, slant_range + delta_range) - fdop) / delta_range

    fn = dopfact - fdop * slant_range
    c1 = -np.dot(vel, vel)
    c2 = (fdop / slant_range) + fdop_der
    fnprime = c1 + c2 * dopfact

    return fn / fnprime if fnprime != 0 else 0.0


def _update_aztime(
    target_ecef: np.ndarray,
    orbit: OrbitInterpolator,
    radar_grid: RadarGrid,
    look_side: LookSide,
    min_azimuth_time: float,
    max_azimuth_time: float,
) -> float:
    num_aztime_test = 15
    tstart = orbit.reference_epoch
    tend = tstart + orbit.number_of_seconds
    dt = (tend - tstart) / (num_aztime_test - 1)

    r_closest = 1e16
    t_closest = tstart + (tend - tstart) / 2.0

    for k in range(num_aztime_test):
        tt = tstart + k * dt
        if tt < tstart or tt > tend:
            continue

        orbit_state = orbit.state_at(tt, allow_extrapolation=True)
        pos = orbit_state.position
        vel = orbit_state.velocity

        rvec = target_ecef - pos
        r = np.linalg.norm(rvec)

        if k == 0:
            if not check_look_side(rvec, vel, pos, look_side):
                continue

        if r < r_closest:
            r_closest = r
            t_closest = tt

    return t_closest


def _geo2rdr_bracket(
    target_ecef: np.ndarray,
    orbit: OrbitInterpolator,
    doppler: DopplerLUT2d,
    wavelength: float,
    look_side: LookSide,
    time_start: float,
    time_end: float,
    tol_aztime: float = 1e-7,
) -> Tuple[float, float]:
    def doppler_error(t: float) -> float:
        orbit_state = orbit.state_at(t, allow_extrapolation=True)
        xp = orbit_state.position
        v = orbit_state.velocity
        r = target_ecef - xp
        rnorm = np.linalg.norm(r)
        fd = doppler.eval(t, rnorm)
        return 2.0 / wavelength * np.dot(v, r) / rnorm - fd

    err, aztime = find_zero_brent(time_start, time_end, doppler_error, tol=tol_aztime)
    
    if err != 0:
        orbit_state = orbit.state_at((time_start + time_end) / 2.0, allow_extrapolation=True)
        xp = orbit_state.position
        r = target_ecef - xp
        return ((time_start + time_end) / 2.0, np.linalg.norm(r))

    orbit_state = orbit.state_at(aztime, allow_extrapolation=True)
    xp = orbit_state.position
    v = orbit_state.velocity
    r = target_ecef - xp
    range_val = np.linalg.norm(r)

    cross_prod = np.cross(r, v)
    dot_prod = np.dot(cross_prod, xp)
    
    if (look_side == LookSide.RIGHT) ^ (dot_prod > 0):
        orbit_state = orbit.state_at((time_start + time_end) / 2.0, allow_extrapolation=True)
        xp = orbit_state.position
        r = target_ecef - xp
        return ((time_start + time_end) / 2.0, np.linalg.norm(r))

    return (aztime, range_val)


def geo2rdr(
    lat: Union[int, float, np.ndarray],
    lon: Union[int, float, np.ndarray],
    height: Union[int, float, np.ndarray],
    radar_grid: RadarGrid,
    satellite_position: Union[np.ndarray, OrbitInterpolator],
    velocity: np.ndarray,
    doppler: Union[float, DopplerLUT2d],
    ellipsoid: Ellipsoid = WGS84,
    wavelength_m: float = 0.0565642,
    look_side: LookSide = LookSide.RIGHT,
    max_iterations: int = 50,
    threshold: float = 1e-8,
    delta_range: float = 10.0,
    use_bracket: bool = True,
) -> np.ndarray:
    if isinstance(doppler, (int, float)):
        doppler_lut = _make_doppler_lut(float(doppler), radar_grid)
    else:
        doppler_lut = doppler

    lat_arr = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    lon_arr = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    h_arr = np.atleast_1d(np.asarray(height, dtype=np.float64))

    use_orbit_interpolator = isinstance(satellite_position, OrbitInterpolator)

    n_points = len(lat_arr)
    results_aztime = np.full(n_points, np.nan)
    results_range = np.full(n_points, np.nan)

    valid_mask = _validate_geo_coordinates(lat_arr, lon_arr, h_arr)
    valid_indices = np.where(valid_mask)[0]

    for i in valid_indices:
        lat_i = lat_arr[i]
        lon_i = lon_arr[i]
        h_i = h_arr[i]

        target_ecef = llh_to_ecef(lat_i, lon_i, h_i, ellipsoid=ellipsoid)

        if use_orbit_interpolator:
            if use_bracket:
                time_start = max(satellite_position.reference_epoch, radar_grid.start_time)
                time_end = min(satellite_position.reference_epoch + satellite_position.number_of_seconds, radar_grid.end_time)
                
                if hasattr(doppler_lut, 'y_start') and hasattr(doppler_lut, 'y_end'):
                    time_start = max(time_start, doppler_lut.y_start())
                    time_end = min(time_end, doppler_lut.y_end())
                
                t_az, slant_range = _geo2rdr_bracket(
                    target_ecef,
                    satellite_position,
                    doppler_lut,
                    wavelength_m,
                    look_side,
                    time_start,
                    time_end,
                )
            else:
                t_az = _update_aztime(
                    target_ecef,
                    satellite_position,
                    radar_grid,
                    look_side,
                    radar_grid.start_time,
                    radar_grid.end_time,
                )
        else:
            t_az = radar_grid.start_time + radar_grid.number_of_seconds / 2.0

        dt = 0.0

        for iteration in range(max_iterations):
            t_az = t_az - dt

            if use_orbit_interpolator:
                orbit_state = satellite_position.state_at(t_az, allow_extrapolation=True)
                sat_pos_i = orbit_state.position
                vel_i = velocity if velocity is not None else orbit_state.velocity
            else:
                sat_pos_i = np.asarray(satellite_position, dtype=np.float64)
                vel_i = velocity

            rvec = target_ecef - sat_pos_i
            slant_range = np.linalg.norm(rvec)

            if iteration == 0:
                if not check_look_side(rvec, vel_i, sat_pos_i, look_side):
                    break

            dt = _compute_doppler_aztime_diff(
                rvec, vel_i, doppler_lut, wavelength_m, t_az, slant_range, delta_range
            )

            if abs(dt) < threshold:
                break

        results_aztime[i] = t_az
        results_range[i] = slant_range

    if n_points == 1:
        return np.array([results_aztime[0], results_range[0]])

    return np.stack([results_aztime, results_range], axis=0)


def geo2rdr_arrayfire(
    lat: Union[int, float, np.ndarray],
    lon: Union[int, float, np.ndarray],
    height: Union[int, float, np.ndarray],
    radar_grid: RadarGrid,
    satellite_position: np.ndarray,
    velocity: np.ndarray,
    doppler: float,
    ellipsoid: Ellipsoid = WGS84,
    wavelength_m: float = 0.0565642,
    max_iterations: int = 50,
    threshold: float = 1e-8,
    look_side: LookSide = LookSide.RIGHT,
) -> np.ndarray:
    from i2sar.geometry.geometry_arrayfire import geo2rdr_arrayfire_core

    if isinstance(satellite_position, OrbitInterpolator) or not isinstance(doppler, (int, float)):
        raise NotImplementedError("ArrayFire geo2rdr currently supports static orbit and constant Doppler only")

    return geo2rdr_arrayfire_core(
        lat=lat,
        lon=lon,
        height=height,
        radar_grid=radar_grid,
        satellite_position=np.asarray(satellite_position, dtype=np.float64),
        velocity=np.asarray(velocity, dtype=np.float64),
        doppler=float(doppler),
        ellipsoid=ellipsoid,
        wavelength_m=wavelength_m,
        max_iterations=max_iterations,
        threshold=threshold,
        look_side_right=(look_side == LookSide.RIGHT),
    )


def compute_geo2rdr_mapping(
    lats: np.ndarray,
    lons: np.ndarray,
    radar_grid: RadarGrid,
    satellite_position: Union[np.ndarray, OrbitInterpolator],
    velocity: np.ndarray,
    doppler: Union[float, DopplerLUT2d],
    ellipsoid: Ellipsoid = WGS84,
    wavelength_m: float = 0.0565642,
    look_side: LookSide = LookSide.RIGHT,
    method: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(doppler, (int, float)):
        doppler_lut = _make_doppler_lut(float(doppler), radar_grid)
    else:
        doppler_lut = doppler

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    height_grid = np.zeros_like(lat_grid, dtype=np.float64)
    n_points = lat_grid.size
    method_for_call = method
    can_use_numba = (
        not isinstance(satellite_position, OrbitInterpolator)
        and velocity is not None
        and isinstance(doppler, (int, float))
    )
    if method_for_call == "auto":
        method_for_call = "numba" if can_use_numba and n_points >= 2000 else "original"

    if method_for_call == "numba":
        if not can_use_numba:
            raise NotImplementedError(
                "numba geo2rdr mapping currently supports static orbit, explicit velocity, and constant Doppler only"
            )
        from i2sar.geometry.accelerated_geometry import geo2rdr_fast

        result = geo2rdr_fast(
            lat=lat_grid.ravel(),
            lon=lon_grid.ravel(),
            height=height_grid.ravel(),
            radar_grid=radar_grid,
            satellite_position=satellite_position,
            velocity=velocity,
            doppler=float(doppler),
            method="numba",
        )
    elif method_for_call == "original":
        result = geo2rdr(
            lat=lat_grid.ravel(),
            lon=lon_grid.ravel(),
            height=height_grid.ravel(),
            radar_grid=radar_grid,
            satellite_position=satellite_position,
            velocity=velocity,
            doppler=doppler_lut,
            ellipsoid=ellipsoid,
            wavelength_m=wavelength_m,
            look_side=look_side,
        )
    else:
        raise ValueError(f"unknown geo2rdr mapping method: {method}")

    return (
        result[0].reshape(lat_grid.shape),
        result[1].reshape(lat_grid.shape),
    )


def geo2rdr_to_pixel(
    aztime: float,
    slant_range: float,
    radar_grid: RadarGrid,
) -> Tuple[float, float]:
    line = radar_grid.azimuth_time_to_line(aztime)
    pixel = radar_grid.slant_range_to_pixel(slant_range)
    return line, pixel


def _geo2rdr_worker(args):
    """geo2rdr 多进程工作函数"""
    (
        i, lat, lon, height, radar_grid, satellite_position, velocity, doppler,
        ellipsoid, wavelength_m, look_side, max_iterations, threshold,
        delta_range, use_bracket
    ) = args
    
    result = geo2rdr(
        lat=lat,
        lon=lon,
        height=height,
        radar_grid=radar_grid,
        satellite_position=satellite_position,
        velocity=velocity,
        doppler=doppler,
        ellipsoid=ellipsoid,
        wavelength_m=wavelength_m,
        look_side=look_side,
        max_iterations=max_iterations,
        threshold=threshold,
        delta_range=delta_range,
        use_bracket=use_bracket,
    )
    return i, result


def geo2rdr_parallel(
    lat: Union[int, float, np.ndarray],
    lon: Union[int, float, np.ndarray],
    height: Union[int, float, np.ndarray],
    radar_grid: RadarGrid,
    satellite_position: Union[np.ndarray, OrbitInterpolator],
    velocity: np.ndarray,
    doppler: Union[float, DopplerLUT2d],
    ellipsoid: Ellipsoid = WGS84,
    wavelength_m: float = 0.0565642,
    look_side: LookSide = LookSide.RIGHT,
    max_iterations: int = 50,
    threshold: float = 1e-8,
    delta_range: float = 10.0,
    use_bracket: bool = True,
    num_workers: int = None,
    min_points_for_parallel: int = 500,
) -> np.ndarray:
    """
    并行版本的 geo2rdr 函数
    
    当数据量较小时（< min_points_for_parallel），使用单进程执行以避免进程开销
    当数据量较大时，使用多进程并行处理以提高性能
    
    参数:
        num_workers: 进程数，默认为 CPU 核心数的 80%
        min_points_for_parallel: 启用并行的最小数据量阈值
    
    返回:
        np.ndarray: [aztime, slant_range] 数组
    """
    lat_arr = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    lon_arr = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    h_arr = np.atleast_1d(np.asarray(height, dtype=np.float64))
    
    n_points = len(lat_arr)
    
    if n_points < min_points_for_parallel:
        return geo2rdr(
            lat=lat,
            lon=lon,
            height=height,
            radar_grid=radar_grid,
            satellite_position=satellite_position,
            velocity=velocity,
            doppler=doppler,
            ellipsoid=ellipsoid,
            wavelength_m=wavelength_m,
            look_side=look_side,
            max_iterations=max_iterations,
            threshold=threshold,
            delta_range=delta_range,
            use_bracket=use_bracket,
        )
    
    if num_workers is None:
        num_workers = max(1, int(mp.cpu_count() * 0.8))
    
    args_list = [
        (
            i, lat_arr[i], lon_arr[i], h_arr[i], radar_grid, satellite_position,
            velocity, doppler, ellipsoid, wavelength_m, look_side,
            max_iterations, threshold, delta_range, use_bracket
        )
        for i in range(n_points)
    ]
    
    with mp.Pool(num_workers) as pool:
        results = pool.map(_geo2rdr_worker, args_list)
    
    results.sort(key=lambda x: x[0])
    aztimes = np.array([r[1][0] for r in results])
    ranges = np.array([r[1][1] for r in results])
    
    if n_points == 1:
        return np.array([aztimes[0], ranges[0]])
    
    return np.stack([aztimes, ranges], axis=0)
