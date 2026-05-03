from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from i2sar.core.enums import LookSide
from i2sar.geometry.ellipsoid import Ellipsoid, WGS84, ecef_to_llh, llh_to_ecef
from i2sar.geometry.radar_grid import RadarGrid
from i2sar.geometry.tcn_basis import TCNBasis
from i2sar.dem.interpolator import DEMInterpolator
from i2sar.orbit.interpolate import OrbitInterpolator


class Pixel:
    def __init__(self, range: float, dopfact: float):
        self.range = range
        self.dopfact = dopfact


def _validate_input_coordinates(line_arr: np.ndarray, pixel_arr: np.ndarray, radar_grid: RadarGrid) -> np.ndarray:
    valid = np.isfinite(line_arr) & np.isfinite(pixel_arr)
    valid &= (line_arr >= 0) & (line_arr < radar_grid.length)
    valid &= (pixel_arr >= 0) & (pixel_arr < radar_grid.width)
    return valid


def _get_elevation_for_point(dem_elevation_arr: np.ndarray, idx: int, pixel: float, dem_ndim: int) -> float:
    if dem_ndim == 0:
        return float(dem_elevation_arr)
    elif dem_ndim == 1:
        return float(dem_elevation_arr[idx])
    else:
        return float(dem_elevation_arr[int(np.floor(pixel)), int(np.floor(pixel))])


def _update_llh(
    sat_pos: np.ndarray,
    radius: float,
    pixel_obj: Pixel,
    tcn_basis: TCNBasis,
    n_dot_v: float,
    v_dot_t: float,
    look_side: LookSide,
    ellipsoid: Ellipsoid,
    h: float,
) -> Tuple[float, float, float]:
    slant_range = pixel_obj.range
    dopfact = pixel_obj.dopfact
    
    pos_norm = np.linalg.norm(sat_pos)
    
    cos_theta = 0.5 * (pos_norm / slant_range + slant_range / pos_norm - (radius / pos_norm) * (radius / slant_range))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
    
    gamma = slant_range * cos_theta
    alpha = (dopfact - gamma * n_dot_v) / v_dot_t
    across = slant_range * sin_theta
    beta = np.sqrt(max(across * across - alpha * alpha, 0.0)) * look_side.sign
    
    x = sat_pos[0] + alpha * tcn_basis.t[0] + beta * tcn_basis.c[0] + gamma * tcn_basis.n[0]
    y = sat_pos[1] + alpha * tcn_basis.t[1] + beta * tcn_basis.c[1] + gamma * tcn_basis.n[1]
    z = sat_pos[2] + alpha * tcn_basis.t[2] + beta * tcn_basis.c[2] + gamma * tcn_basis.n[2]
    
    llh = ecef_to_llh(x, y, z, ellipsoid=ellipsoid)
    return llh


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
    from i2sar.math.root_find import find_zero_brent
    
    speed = np.linalg.norm(vel)
    if speed < 1e-10:
        raise ValueError("Velocity magnitude too small")
    
    along_track = vel / speed
    
    vel_cross_rad = np.cross(vel, sat_pos)
    vel_cross_rad_norm = np.linalg.norm(vel_cross_rad)
    if vel_cross_rad_norm < 1e-10:
        raise ValueError("Satellite position and velocity are collinear")
    
    right = vel_cross_rad / vel_cross_rad_norm
    down = np.cross(along_track, right)
    
    if look_side == LookSide.LEFT:
        horizontal = -right
    else:
        horizontal = right
    
    sin_squint = doppler * wavelength / (2.0 * speed)
    cos_squint = np.sqrt(max(0.0, 1.0 - sin_squint * sin_squint))
    
    center = sat_pos + sin_squint * slant_range * along_track
    radius = cos_squint * slant_range
    
    def get_xyz(look_angle):
        return center + radius * (np.sin(look_angle) * horizontal + np.cos(look_angle) * down)
    
    def height_error(look_angle):
        xyz = get_xyz(look_angle)
        llh = ecef_to_llh(xyz[0], xyz[1], xyz[2], ellipsoid=ellipsoid)
        dem_h = dem.interpolate_at_lonlat(llh[1], llh[0])
        return llh[2] - dem_h
    
    try:
        _, look_solution = find_zero_brent(look_min, look_max, height_error, tol=tol_height / radius)
    except Exception:
        look_solution = (look_min + look_max) / 2.0
    
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
            llh = ecef_to_llh(xyz[0], xyz[1], xyz[2], ellipsoid=ellipsoid)
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

        llh_new = ecef_to_llh(sat_pos_i[0], sat_pos_i[1], sat_pos_i[2], ellipsoid=ellipsoid)
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
                llh_new = ecef_to_llh(xyz_avg[0], xyz_avg[1], xyz_avg[2], ellipsoid=ellipsoid)
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


class Rdr2GeoParams:
    """rdr2geo 参数容器"""
    
    def __init__(
        self,
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
    ):
        self.radar_grid = radar_grid
        self.satellite_position = satellite_position
        self.velocity = velocity
        self.doppler = doppler
        self.dem = dem
        self.ellipsoid = ellipsoid
        self.wavelength_m = wavelength_m
        self.look_side = look_side
        self.max_iterations = max_iterations
        self.extra_iterations = extra_iterations
        self.threshold = threshold


class Rdr2GeoResult:
    """ISCE3风格的rdr2geo输出结果容器"""
    
    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        height: np.ndarray,
        convergence: Optional[np.ndarray] = None,
        incidence_angle: Optional[np.ndarray] = None,
        heading_angle: Optional[np.ndarray] = None,
        local_incidence_angle: Optional[np.ndarray] = None,
        layover_mask: Optional[np.ndarray] = None,
        shadow_mask: Optional[np.ndarray] = None,
        range_spacing: Optional[np.ndarray] = None,
        azimuth_spacing: Optional[np.ndarray] = None,
    ):
        self.lat = lat
        self.lon = lon
        self.height = height
        self.convergence = convergence
        self.incidence_angle = incidence_angle
        self.heading_angle = heading_angle
        self.local_incidence_angle = local_incidence_angle
        self.layover_mask = layover_mask
        self.shadow_mask = shadow_mask
        self.range_spacing = range_spacing
        self.azimuth_spacing = azimuth_spacing
    
    def to_array(self) -> np.ndarray:
        """返回基础输出数组 [lat, lon, height]"""
        return np.stack([self.lat, self.lon, self.height], axis=0)
    
    def to_dict(self) -> dict:
        """返回包含所有输出项的字典"""
        result = {
            'lat': self.lat,
            'lon': self.lon,
            'height': self.height,
        }
        if self.convergence is not None:
            result['convergence'] = self.convergence
        if self.incidence_angle is not None:
            result['incidence_angle'] = self.incidence_angle
        if self.heading_angle is not None:
            result['heading_angle'] = self.heading_angle
        if self.local_incidence_angle is not None:
            result['local_incidence_angle'] = self.local_incidence_angle
        if self.layover_mask is not None:
            result['layover_mask'] = self.layover_mask
        if self.shadow_mask is not None:
            result['shadow_mask'] = self.shadow_mask
        if self.range_spacing is not None:
            result['range_spacing'] = self.range_spacing
        if self.azimuth_spacing is not None:
            result['azimuth_spacing'] = self.azimuth_spacing
        return result


def rdr2geo_full(
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
    compute_incidence: bool = False,
    compute_layover_shadow: bool = False,
    compute_spacing: bool = False,
) -> Rdr2GeoResult:
    """
    ISCE3风格的完整rdr2geo实现，支持扩展输出项
    
    Parameters
    ----------
    line : int, float, or np.ndarray
        方位向像素坐标
    pixel : int, float, or np.ndarray
        距离向像素坐标
    radar_grid : RadarGrid
        雷达网格参数
    satellite_position : np.ndarray or OrbitInterpolator
        卫星位置或轨道插值器
    velocity : np.ndarray
        卫星速度
    doppler : float
        多普勒中心频率
    dem : float, np.ndarray, or DEMInterpolator, optional
        数字高程模型，默认为0.0
    ellipsoid : Ellipsoid, optional
        椭球参数，默认为WGS84
    wavelength_m : float, optional
        波长（米），默认为C波段
    look_side : LookSide, optional
        观测侧，默认为RIGHT
    max_iterations : int, optional
        最大迭代次数，默认为25
    extra_iterations : int, optional
        额外迭代次数（用于收敛优化），默认为15
    threshold : float, optional
        收敛阈值，默认为1e-8
    dem_elevation : float, np.ndarray, or None, optional
        DEM高程（已弃用，使用dem参数）
    compute_incidence : bool, optional
        是否计算入射角和航向角，默认为False
    compute_layover_shadow : bool, optional
        是否计算叠掩和阴影掩码，默认为False
    compute_spacing : bool, optional
        是否计算像素间距，默认为False
    
    Returns
    -------
    Rdr2GeoResult
        包含以下字段的结果对象：
        - lat: 纬度（度）
        - lon: 经度（度）
        - height: 高度（米）
        - convergence: 收敛迭代次数（可选）
        - incidence_angle: 入射角（度，可选）
        - heading_angle: 航向角（度，可选）
        - local_incidence_angle: 局部入射角（度，可选）
        - layover_mask: 叠掩掩码（可选）
        - shadow_mask: 阴影掩码（可选）
        - range_spacing: 距离像素间距（米，可选）
        - azimuth_spacing: 方位像素间距（米，可选）
    """
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
    results_convergence = np.full(n_points, -1) if compute_incidence else None
    results_incidence = np.full(n_points, np.nan) if compute_incidence else None
    results_heading = np.full(n_points, np.nan) if compute_incidence else None
    results_local_incidence = np.full(n_points, np.nan) if compute_incidence else None
    results_layover = np.full(n_points, False) if compute_layover_shadow else None
    results_shadow = np.full(n_points, False) if compute_layover_shadow else None
    results_range_spacing = np.full(n_points, np.nan) if compute_spacing else None
    results_az_spacing = np.full(n_points, np.nan) if compute_spacing else None

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

        llh_new = ecef_to_llh(sat_pos_i[0], sat_pos_i[1], sat_pos_i[2], ellipsoid=ellipsoid)
        h = height

        llh_old = None
        iteration_count = 0
        
        for iteration in range(max_iterations + extra_iterations):
            iteration_count = iteration + 1
            
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
                llh_new = ecef_to_llh(xyz_avg[0], xyz_avg[1], xyz_avg[2], ellipsoid=ellipsoid)
                h = np.linalg.norm(xyz_avg) - radius

            llh_old = llh_new

        final_llh = _update_llh(
            sat_pos_i, radius, pixel_obj, tcn_basis, n_dot_v, v_dot_t, look_side, ellipsoid, h
        )

        results_lat[i] = final_llh[0]
        results_lon[i] = final_llh[1]
        results_h[i] = final_llh[2]
        
        if compute_incidence:
            results_convergence[i] = iteration_count
            
            xyz_final = llh_to_ecef(final_llh[0], final_llh[1], final_llh[2], ellipsoid=ellipsoid)
            r_vec = xyz_final - sat_pos_i
            r_norm = np.linalg.norm(r_vec)
            r_unit = r_vec / r_norm
            
            incidence_angle = np.arccos(np.dot(r_unit, tcn_basis.n))
            results_incidence[i] = np.rad2deg(incidence_angle)
            
            heading_angle = np.arctan2(vel_i[1], vel_i[0])
            results_heading[i] = np.rad2deg(heading_angle)
            
            if compute_layover_shadow:
                local_incidence = np.arccos(np.dot(-r_unit, tcn_basis.n))
                results_local_incidence[i] = np.rad2deg(local_incidence)
                
                layover_condition = np.dot(r_unit, tcn_basis.t) > 0
                results_layover[i] = layover_condition
                
                shadow_condition = np.dot(r_unit, tcn_basis.n) < 0
                results_shadow[i] = shadow_condition
        
        if compute_spacing:
            ground_range_spacing = radar_grid.range_pixel_spacing_m * np.cos(np.deg2rad(results_incidence[i]))
            results_range_spacing[i] = ground_range_spacing
            
            azimuth_spacing = vmag / radar_grid.prf_hz
            results_az_spacing[i] = azimuth_spacing

    return Rdr2GeoResult(
        lat=results_lat if n_points > 1 else results_lat[0],
        lon=results_lon if n_points > 1 else results_lon[0],
        height=results_h if n_points > 1 else results_h[0],
        convergence=results_convergence if (compute_incidence and n_points > 1) else (results_convergence[0] if compute_incidence else None),
        incidence_angle=results_incidence if (compute_incidence and n_points > 1) else (results_incidence[0] if compute_incidence else None),
        heading_angle=results_heading if (compute_incidence and n_points > 1) else (results_heading[0] if compute_incidence else None),
        local_incidence_angle=results_local_incidence if (compute_incidence and n_points > 1) else (results_local_incidence[0] if compute_incidence else None),
        layover_mask=results_layover if (compute_layover_shadow and n_points > 1) else (results_layover[0] if compute_layover_shadow else None),
        shadow_mask=results_shadow if (compute_layover_shadow and n_points > 1) else (results_shadow[0] if compute_layover_shadow else None),
        range_spacing=results_range_spacing if (compute_spacing and n_points > 1) else (results_range_spacing[0] if compute_spacing else None),
        azimuth_spacing=results_az_spacing if (compute_spacing and n_points > 1) else (results_az_spacing[0] if compute_spacing else None),
    )


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
        np.arange(radar_grid.length),
        np.arange(radar_grid.width),
        indexing='ij'
    )
    
    line_flat = lines.flatten()
    pixel_flat = pixels.flatten()
    
    if method == "arrayfire" or (method == "auto" and _is_arrayfire_available()):
        try:
            result = rdr2geo_arrayfire(
                line=line_flat,
                pixel=pixel_flat,
                radar_grid=radar_grid,
                satellite_position=satellite_position,
                velocity=velocity,
                doppler=doppler,
                dem=dem,
                ellipsoid=ellipsoid,
                wavelength_m=wavelength_m,
                look_side=look_side,
            )
        except (ImportError, RuntimeError):
            result = rdr2geo(
                line=line_flat,
                pixel=pixel_flat,
                radar_grid=radar_grid,
                satellite_position=satellite_position,
                velocity=velocity,
                doppler=doppler,
                dem=dem,
                ellipsoid=ellipsoid,
                wavelength_m=wavelength_m,
                look_side=look_side,
            )
    else:
        result = rdr2geo(
            line=line_flat,
            pixel=pixel_flat,
            radar_grid=radar_grid,
            satellite_position=satellite_position,
            velocity=velocity,
            doppler=doppler,
            dem=dem,
            ellipsoid=ellipsoid,
            wavelength_m=wavelength_m,
            look_side=look_side,
        )
    
    lat_grid = result[0].reshape(radar_grid.length, radar_grid.width)
    lon_grid = result[1].reshape(radar_grid.length, radar_grid.width)
    hgt_grid = result[2].reshape(radar_grid.length, radar_grid.width)
    
    return lat_grid, lon_grid, hgt_grid


def _is_arrayfire_available() -> bool:
    try:
        from i2sar.geometry.geometry_arrayfire import rdr2geo_arrayfire_core
        from i2sar.accel.arrayfire_backend import ArrayFireBackend
        backend = ArrayFireBackend()
        return backend.available
    except ImportError:
        return False


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
    n_workers: int = 4,
) -> np.ndarray:
    from concurrent.futures import ThreadPoolExecutor
    
    line_arr = np.atleast_1d(np.asarray(line, dtype=np.float64))
    pixel_arr = np.atleast_1d(np.asarray(pixel, dtype=np.float64))
    n_points = len(line_arr)
    
    chunk_size = (n_points + n_workers - 1) // n_workers
    
    results = []
    
    def process_chunk(start, end):
        return rdr2geo(
            line=line_arr[start:end],
            pixel=pixel_arr[start:end],
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
        )
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i in range(n_workers):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_points)
            if start < end:
                futures.append(executor.submit(process_chunk, start, end))
        
        for future in futures:
            results.append(future.result())
    
    if len(results) == 0:
        return np.array([[], [], []])
    
    lat = np.concatenate([r[0] for r in results])
    lon = np.concatenate([r[1] for r in results])
    h = np.concatenate([r[2] for r in results])
    
    if n_points == 1:
        return np.array([lat[0], lon[0], h[0]])
    
    return np.stack([lat, lon, h], axis=0)