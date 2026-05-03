"""调试 geo2rdr 函数 - 添加详细输出"""

import numpy as np
from typing import Union, Tuple

from i2sar.geometry import RadarGrid, LookSide, llh_to_ecef, check_look_side
from i2sar.orbit.interpolate import OrbitInterpolator
from i2sar.geometry.geo2rdr import _validate_geo_coordinates, _make_doppler_lut, _geo2rdr_bracket, _compute_doppler_aztime_diff


def geo2rdr_debug(
    lat: Union[int, float, np.ndarray],
    lon: Union[int, float, np.ndarray],
    height: Union[int, float, np.ndarray],
    radar_grid: RadarGrid,
    satellite_position: Union[np.ndarray, OrbitInterpolator],
    velocity: np.ndarray,
    doppler: Union[float, object],
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
    
    print(f'\n验证坐标:')
    print(f'  valid_mask: {valid_mask}')
    print(f'  valid_indices: {valid_indices}')

    for i in valid_indices:
        lat_i = lat_arr[i]
        lon_i = lon_arr[i]
        h_i = h_arr[i]

        target_ecef = llh_to_ecef(lat_i, lon_i, h_i)
        
        print(f'\n处理点 {i}:')
        print(f'  lat={lat_i:.6f}, lon={lon_i:.6f}, h={h_i:.2f}')

        if use_orbit_interpolator:
            if use_bracket:
                time_start = max(satellite_position.reference_epoch, radar_grid.start_time)
                time_end = min(satellite_position.reference_epoch + satellite_position.number_of_seconds, radar_grid.end_time)
                
                print(f'  调用 _geo2rdr_bracket: time_start={time_start}, time_end={time_end}')
                
                t_az, slant_range = _geo2rdr_bracket(
                    target_ecef,
                    satellite_position,
                    doppler_lut,
                    wavelength_m,
                    look_side,
                    time_start,
                    time_end,
                )
                
                print(f'  _geo2rdr_bracket 返回: t_az={t_az:.10f}, slant_range={slant_range:.6f}')
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
        look_side_failed = False

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
                is_valid = check_look_side(rvec, vel_i, sat_pos_i, look_side)
                print(f'  迭代 {iteration}: check_look_side 返回 {is_valid}')
                if not is_valid:
                    look_side_failed = True
                    print(f'  look side 检查失败')
                    break

            dt = _compute_doppler_aztime_diff(
                rvec, vel_i, doppler_lut, wavelength_m, t_az, slant_range, delta_range
            )

            if abs(dt) < threshold:
                print(f'  迭代 {iteration}: 收敛, dt={dt:.6e}')
                break

        if look_side_failed:
            print(f'  结果: look side 检查失败')
        else:
            print(f'  结果: t_az={t_az:.10f}, slant_range={slant_range:.6f}')
            results_aztime[i] = t_az
            results_range[i] = slant_range

    if n_points == 1:
        return np.array([results_aztime[0], results_range[0]])

    return np.stack([results_aztime, results_range], axis=0)


# 测试
import xml.etree.ElementTree as ET
from i2sar.geometry import rdr2geo


def parse_isce3_orbit_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    times = []
    positions = []
    velocities = []
    
    for state_vector in root.findall('.//orbitStateVector'):
        gps_time = float(state_vector.find('gps').text)
        x = float(state_vector.find('x').text)
        y = float(state_vector.find('y').text)
        z = float(state_vector.find('z').text)
        vx = float(state_vector.find('vx').text)
        vy = float(state_vector.find('vy').text)
        vz = float(state_vector.find('vz').text)
        
        times.append(gps_time)
        positions.append([x, y, z])
        velocities.append([vx, vy, vz])
    
    return np.array(times), np.array(positions), np.array(velocities)


def test():
    orbit_xml_path = '/home/ysdong/Software/isce/isce3/tests/data/orbit.xml'
    times, positions, velocities = parse_isce3_orbit_xml(orbit_xml_path)
    
    orbit = OrbitInterpolator(times, positions, velocities)
    
    radar_grid = RadarGrid(
        length=1000,
        width=500,
        sensing_start_s=times[0] + (times[-1] - times[0]) * 0.2,
        prf_hz=1666.67,
        starting_range_m=700000.0,
        range_pixel_spacing_m=2.33,
    )
    
    mid_line = radar_grid.length / 2.0
    mid_pixel = radar_grid.width / 2.0
    
    lat, lon, hgt = rdr2geo(
        line=mid_line,
        pixel=mid_pixel,
        radar_grid=radar_grid,
        satellite_position=orbit,
        velocity=None,
        doppler=125.0,
        look_side=LookSide.RIGHT,
    )
    
    print('=' * 80)
    print('测试修改版 geo2rdr_debug')
    print('=' * 80)
    
    result = geo2rdr_debug(
        lat=lat,
        lon=lon,
        height=hgt,
        radar_grid=radar_grid,
        satellite_position=orbit,
        velocity=None,
        doppler=125.0,
        look_side=LookSide.RIGHT,
    )
    
    print(f'\n最终结果:')
    print(f'  aztime={result[0]}')
    print(f'  range={result[1]}')


if __name__ == '__main__':
    test()
