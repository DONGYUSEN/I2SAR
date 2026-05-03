from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Union

import numpy as np

from i2sar.core.enums import LookSide
from i2sar.dem import DEMInterpolator
from i2sar.geometry.ellipsoid import Ellipsoid, WGS84, ecef_to_llh, llh_to_ecef, xyz_to_lon_lat
from i2sar.geometry.pixel import Pixel
from i2sar.geometry.radar_grid import RadarGrid
from i2sar.geometry.tcn_basis import TCNBasis
from i2sar.orbit import OrbitInterpolator
from i2sar.math.root_find import find_zero_brent


def _validate_input_coordinates(line_arr: np.ndarray, pixel_arr: np.ndarray, radar_grid: RadarGrid) -> np.ndarray:
    """
    验证输入坐标的有效性，返回有效索引数组
    """
    valid_mask = np.ones(len(line_arr), dtype=bool)
    
    valid_mask = np.logical_and(valid_mask, np.isfinite(line_arr))
    valid_mask = np.logical_and(valid_mask, np.isfinite(pixel_arr))
    
    valid_mask = np.logical_and(valid_mask, line_arr >= 0)
    valid_mask = np.logical_and(valid_mask, line_arr < radar_grid.length)
    valid_mask = np.logical_and(valid_mask, pixel_arr >= 0)
    valid_mask = np.logical_and(valid_mask, pixel_arr < radar_grid.width)
    
    return valid_mask


def _mark_invalid_results(results: np.ndarray, invalid_mask: np.ndarray) -> np.ndarray:
    """
    将无效位置标记为 NaN
    """
    results[:, invalid_mask] = np.nan
    return results


@dataclass(frozen=True)
class Rdr2GeoParams:
    radar_grid: RadarGrid
    satellite_position: Union[np.ndarray, OrbitInterpolator]
    velocity: np.ndarray
    doppler: float
    dem: DEMInterpolator
    ellipsoid: Ellipsoid = WGS84
    wavelength_m: float = 0.0565642
    look_side: LookSide = LookSide.RIGHT
    max_iterations: int = 25
    extra_iterations: int = 15
    threshold: float = 1e-8


def _get_elevation_for_point(dem_elevation, line_idx, pixel_idx, dem_ndim):
    if isinstance(dem_elevation, (int, float)):
        return float(dem_elevation)
    if dem_ndim == 0:
        return float(dem_elevation)
    if dem_ndim == 1:
        return dem_elevation[int(line_idx)]
    return dem_elevation[int(line_idx), int(pixel_idx)]


def _update_llh(
    sat_pos: np.ndarray,
    radius: float,
    pixel: Pixel,
    tcn_basis: TCNBasis,
    n_dot_v: float,
    v_dot_t: float,
    side: LookSide,
    ellipsoid: Ellipsoid,
    h: float,
) -> tuple[float, float, float]:
    a = np.linalg.norm(sat_pos)
    b = radius + h

    cos_theta = 0.5 * (a / pixel.range + pixel.range / a - (b / a) * (b / pixel.range))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

    gamma = pixel.range * cos_theta
    alpha = (pixel.dopfact - gamma * n_dot_v) / v_dot_t

    x = pixel.range * sin_theta
    beta_sq = x * x - alpha * alpha
    if beta_sq < 0:
        beta = 0.0
    else:
        beta = np.sqrt(beta_sq)
    beta = beta * side.sign

    delta = alpha * tcn_basis.t + beta * tcn_basis.c + gamma * tcn_basis.n
    xyz = sat_pos + delta

    return xyz_to_lon_lat(xyz, ellipsoid=ellipsoid)


def _rdr2geo_bracket(
    sat_pos: np.ndarray,
    vel: np.ndarray,
    slant_range: float,
    doppler: float,
    wavelength: float,
    look_side: LookSide,
    dem: DEMInterpolator,
    ellipsoid: Ellipsoid,
    look_min: float = -np.pi / 2,
    look_max: float = np.pi / 2,
    tol_height: float = 0.1,
) -> np.ndarray:
    speed = np.linalg.norm(vel)
    along_track = vel / speed
    right = np.cross(along_track, sat_pos)
    right = right / np.linalg.norm(right)
    down = np.cross(along_track, right)
    horizontal = right if look_side == LookSide.RIGHT else -right

    sin_squint = doppler * wavelength / (2 * speed)
    sin_squint = np.clip(sin_squint, -1.0, 1.0)
    cos_squint = np.sqrt(1.0 - sin_squint * sin_squint)

    center = sat_pos + sin_squint * slant_range * along_track
    radius = cos_squint * slant_range

    def get_xyz(look):
        return center + radius * np.sin(look) * horizontal + radius * np.cos(look) * down

    def dh(look):
        xyz = get_xyz(look)
        lon_lat_h = xyz_to_lon_lat(xyz, ellipsoid=ellipsoid)
        dem_height = dem.interpolate_at_lonlat(lon_lat_h[0], lon_lat_h[1])
        return lon_lat_h[2] - dem_height

    tol_look = tol_height / radius

    err, look_solution = find_zero_brent(look_min, look_max, dh, tol=tol_look)

    if err != 0:
        return get_xyz(0.0)

    return get_xyz(look_solution)


def rdr2geo(
    line: Union[int, float, np.ndarray],
    pixel: Union[int, float, np.ndarray],
    radar_grid: RadarGrid,
    satellite_position: Union[np.ndarray, OrbitInterpolator],
    velocity: np.ndarray,
    doppler: float,
    dem: Union[float, np.ndarray, DEMInterpolator] = 0.0,
    ellipsoid: Ellipsoid = WGS84,
    wavelength_m: float = 0.0565642,
    look_side: LookSide = LookSide.RIGHT,
    max_iterations: int = 25,
    extra_iterations: int = 15,
    threshold: float = 1e-8,
    dem_elevation: Union[float, np.ndarray, None] = None,
    use_bracket: bool = False,
    look_min: float = -np.pi / 2,
    look_max: float = np.pi / 2,
    tol_height: float = 0.1,
) -> np.ndarray:
    if dem_elevation is not None:
        dem = dem_elevation

    dem_elevation_arr = np.asarray(dem)
    dem_ndim = dem_elevation_arr.ndim
    use_dem_interpolator = isinstance(dem, DEMInterpolator)

    line_arr = np.atleast_1d(np.asarray(line, dtype=np.float64))
    pixel_arr = np.atleast_1d(np.asarray(pixel, dtype=np.float64))

    use_orbit_interpolator = isinstance(satellite_position, OrbitInterpolator)

    n_points = len(line_arr)
    results_lat = np.full(n_points, np.nan)
    results_lon = np.full(n_points, np.nan)
    results_h = np.full(n_points, np.nan)

    valid_mask = _validate_input_coordinates(line_arr, pixel_arr, radar_grid)
    valid_indices = np.where(valid_mask)[0]

    for i in valid_indices:
        l_i = line_arr[i]
        p_i = pixel_arr[i]

        if use_orbit_interpolator:
            t_az = radar_grid.line_to_azimuth_time(l_i)
            orbit_state = satellite_position.state_at(t_az, allow_extrapolation=True)
            sat_pos_i = orbit_state.position
            vel_i = velocity if velocity is not None else orbit_state.velocity
        else:
            sat_pos_i = np.asarray(satellite_position, dtype=np.float64)
            vel_i = velocity

        slant_range = radar_grid.pixel_to_slant_range(p_i)

        if use_bracket and use_dem_interpolator:
            xyz = _rdr2geo_bracket(
                sat_pos=sat_pos_i,
                vel=vel_i,
                slant_range=slant_range,
                doppler=doppler,
                wavelength=wavelength_m,
                look_side=look_side,
                dem=dem,
                ellipsoid=ellipsoid,
                look_min=look_min,
                look_max=look_max,
                tol_height=tol_height,
            )
            llh = xyz_to_lon_lat(xyz, ellipsoid=ellipsoid)
            results_lat[i] = llh[0]
            results_lon[i] = llh[1]
            results_h[i] = llh[2]
            continue

        vmag = np.linalg.norm(vel_i)
        vhat = vel_i / vmag

        tcn_basis = TCNBasis.from_position_velocity(sat_pos_i, vel_i)

        n_dot_v = np.dot(tcn_basis.n, vhat)
        v_dot_t = np.dot(vhat, tcn_basis.t)

        dopfact = 0.5 * wavelength_m * doppler * slant_range / vmag
        pixel_obj = Pixel(range=slant_range, dopfact=dopfact)

        major = ellipsoid.a
        minor = major * np.sqrt(1.0 - ellipsoid.eccentricity_squared)
        sat_dist = np.linalg.norm(sat_pos_i)
        eta = 1.0 / np.sqrt(
            (sat_pos_i[0] / major) ** 2
            + (sat_pos_i[1] / major) ** 2
            + (sat_pos_i[2] / minor) ** 2
        )
        radius = eta * sat_dist
        height = (1.0 - eta) * sat_dist

        llh_new = xyz_to_lon_lat(sat_pos_i, ellipsoid=ellipsoid)
        h = height

        llh_old = None
        for iteration in range(max_iterations + extra_iterations):
            if height - h >= slant_range:
                break

            llh_new = _update_llh(
                sat_pos_i, radius, pixel_obj, tcn_basis, n_dot_v, v_dot_t, look_side, ellipsoid, h
            )

            if use_dem_interpolator:
                h_new = dem.interpolate_at_lonlat(llh_new[0], llh_new[1])
            else:
                h_new = _get_elevation_for_point(dem_elevation_arr, i, p_i, dem_ndim)
            llh_new = (llh_new[0], llh_new[1], h_new)

            xyz_new = llh_to_ecef(llh_new[0], llh_new[1], llh_new[2], ellipsoid=ellipsoid)
            h = np.linalg.norm(xyz_new) - radius

            rng = np.linalg.norm(sat_pos_i - xyz_new)
            dr = abs(pixel_obj.range - rng)

            if dr < threshold:
                break

            if iteration > max_iterations:
                xyz_old = llh_to_ecef(llh_old[0], llh_old[1], llh_old[2], ellipsoid=ellipsoid)
                xyz_avg = 0.5 * (xyz_old + xyz_new)
                llh_new = xyz_to_lon_lat(xyz_avg, ellipsoid=ellipsoid)
                h = np.linalg.norm(xyz_avg) - radius

            llh_old = llh_new

        final_llh = _update_llh(
            sat_pos_i, radius, pixel_obj, tcn_basis, n_dot_v, v_dot_t, look_side, ellipsoid, h
        )

        results_lat[i] = final_llh[0]
        results_lon[i] = final_llh[1]
        results_h[i] = final_llh[2]

    if n_points == 1:
        return np.array([results_lat[0], results_lon[0], results_h[0]])

    return np.stack([results_lat, results_lon, results_h], axis=0)


def rdr2geo_arrayfire(
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
    from i2sar.geometry.geometry_arrayfire import rdr2geo_arrayfire_core

    if isinstance(satellite_position, OrbitInterpolator) or isinstance(dem, DEMInterpolator):
        raise NotImplementedError("ArrayFire rdr2geo currently supports static orbit and scalar/array DEM only")

    return rdr2geo_arrayfire_core(
        line=line,
        pixel=pixel,
        radar_grid=radar_grid,
        satellite_position=np.asarray(satellite_position, dtype=np.float64),
        velocity=np.asarray(velocity, dtype=np.float64),
        doppler=float(doppler),
        dem=dem,
        ellipsoid=ellipsoid,
        wavelength_m=wavelength_m,
        look_side=look_side,
        max_iterations=max_iterations,
        extra_iterations=extra_iterations,
        threshold=threshold,
    )


def compute_rdr2geo_mapping(
    radar_grid: RadarGrid,
    satellite_position: Union[np.ndarray, OrbitInterpolator],
    velocity: np.ndarray,
    doppler: float,
    dem: Union[float, np.ndarray, DEMInterpolator] = 0.0,
    ellipsoid: Ellipsoid = WGS84,
    wavelength_m: float = 0.0565642,
    look_side: LookSide = LookSide.RIGHT,
    dem_elevation: Union[float, np.ndarray, None] = None,
    method: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dem_elevation is not None:
        dem = dem_elevation
    lines, pixels = np.meshgrid(
        np.arange(radar_grid.length, dtype=np.float64),
        np.arange(radar_grid.width, dtype=np.float64),
        indexing="ij",
    )
    dem_for_call = dem
    if isinstance(dem, np.ndarray) and dem.ndim == 2 and dem.shape == (radar_grid.length, radar_grid.width):
        dem_for_call = dem.ravel()

    n_points = lines.size
    method_for_call = method
    can_use_numba = (
        not isinstance(satellite_position, OrbitInterpolator)
        and velocity is not None
        and isinstance(doppler, (int, float))
        and not isinstance(dem_for_call, DEMInterpolator)
    )
    if method_for_call == "auto":
        method_for_call = "numba" if can_use_numba and n_points >= 2000 else "original"

    if method_for_call == "numba":
        if not can_use_numba:
            raise NotImplementedError(
                "numba rdr2geo mapping currently supports static orbit, explicit velocity, constant Doppler, "
                "and scalar/array DEM only"
            )
        from i2sar.geometry.accelerated_geometry import rdr2geo_fast

        result = rdr2geo_fast(
            line=lines.ravel(),
            pixel=pixels.ravel(),
            radar_grid=radar_grid,
            satellite_position=satellite_position,
            velocity=velocity,
            doppler=float(doppler),
            dem=dem_for_call,
            method="numba",
        )
    elif method_for_call == "original":
        result = rdr2geo(
            line=lines.ravel(),
            pixel=pixels.ravel(),
            radar_grid=radar_grid,
            satellite_position=satellite_position,
            velocity=velocity,
            doppler=doppler,
            dem=dem_for_call,
            ellipsoid=ellipsoid,
            wavelength_m=wavelength_m,
            look_side=look_side,
        )
    else:
        raise ValueError(f"unknown rdr2geo mapping method: {method}")

    return (
        result[0].reshape(radar_grid.length, radar_grid.width),
        result[1].reshape(radar_grid.length, radar_grid.width),
        result[2].reshape(radar_grid.length, radar_grid.width),
    )


def _rdr2geo_worker(args):
    """rdr2geo 多进程工作函数"""
    (
        i, line, pixel, radar_grid, satellite_position, velocity, doppler, dem,
        ellipsoid, wavelength_m, look_side, max_iterations, extra_iterations,
        threshold, use_bracket, look_min, look_max, tol_height
    ) = args
    
    result = rdr2geo(
        line=line,
        pixel=pixel,
        radar_grid=radar_grid,
        satellite_position=satellite_position,
        velocity=velocity,
        doppler=doppler,
        dem=dem,
        ellipsoid=ellipsoid,
        wavelength_m=wavelength_m,
        look_side=look_side,
        max_iterations=max_iterations,
        extra_iterations=extra_iterations,
        threshold=threshold,
        use_bracket=use_bracket,
        look_min=look_min,
        look_max=look_max,
        tol_height=tol_height,
    )
    return i, result


def rdr2geo_parallel(
    line: Union[int, float, np.ndarray],
    pixel: Union[int, float, np.ndarray],
    radar_grid: RadarGrid,
    satellite_position: Union[np.ndarray, OrbitInterpolator],
    velocity: np.ndarray,
    doppler: float,
    dem: Union[float, np.ndarray, DEMInterpolator] = 0.0,
    ellipsoid: Ellipsoid = WGS84,
    wavelength_m: float = 0.0565642,
    look_side: LookSide = LookSide.RIGHT,
    max_iterations: int = 25,
    extra_iterations: int = 15,
    threshold: float = 1e-8,
    use_bracket: bool = False,
    look_min: float = -np.pi / 2,
    look_max: float = np.pi / 2,
    tol_height: float = 0.1,
    num_workers: int = None,
    min_points_for_parallel: int = 500,
) -> np.ndarray:
    """
    并行版本的 rdr2geo 函数
    
    当数据量较小时（< min_points_for_parallel），使用单进程执行以避免进程开销
    当数据量较大时，使用多进程并行处理以提高性能
    
    参数:
        num_workers: 进程数，默认为 CPU 核心数
        min_points_for_parallel: 启用并行的最小数据量阈值
    
    返回:
        np.ndarray: [lat, lon, height] 数组
    """
    line_arr = np.atleast_1d(np.asarray(line, dtype=np.float64))
    pixel_arr = np.atleast_1d(np.asarray(pixel, dtype=np.float64))
    n_points = len(line_arr)
    
    if n_points < min_points_for_parallel:
        return rdr2geo(
            line=line,
            pixel=pixel,
            radar_grid=radar_grid,
            satellite_position=satellite_position,
            velocity=velocity,
            doppler=doppler,
            dem=dem,
            ellipsoid=ellipsoid,
            wavelength_m=wavelength_m,
            look_side=look_side,
            max_iterations=max_iterations,
            extra_iterations=extra_iterations,
            threshold=threshold,
            use_bracket=use_bracket,
            look_min=look_min,
            look_max=look_max,
            tol_height=tol_height,
        )
    
    if num_workers is None:
        num_workers = max(1, int(mp.cpu_count() * 0.8))
    
    args_list = [
        (
            i, line_arr[i], pixel_arr[i], radar_grid, satellite_position,
            velocity, doppler, dem, ellipsoid, wavelength_m, look_side,
            max_iterations, extra_iterations, threshold, use_bracket,
            look_min, look_max, tol_height
        )
        for i in range(n_points)
    ]
    
    with mp.Pool(num_workers) as pool:
        results = pool.map(_rdr2geo_worker, args_list)
    
    results.sort(key=lambda x: x[0])
    lats = np.array([r[1][0] for r in results])
    lons = np.array([r[1][1] for r in results])
    heights = np.array([r[1][2] for r in results])
    
    if n_points == 1:
        return np.array([lats[0], lons[0], heights[0]])
    
    return np.stack([lats, lons, heights], axis=0)
