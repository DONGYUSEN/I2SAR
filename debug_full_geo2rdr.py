"""调试完整的 geo2rdr 函数"""

import numpy as np
import xml.etree.ElementTree as ET

from i2sar.geometry import RadarGrid, rdr2geo, llh_to_ecef, LookSide, check_look_side
from i2sar.orbit.interpolate import OrbitInterpolator
from i2sar.geometry.geo2rdr import _geo2rdr_bracket, _make_doppler_lut, _compute_doppler_aztime_diff


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


def debug_full_geo2rdr():
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
    
    target_ecef = llh_to_ecef(lat, lon, hgt)
    
    doppler_lut = _make_doppler_lut(125.0, radar_grid)
    wavelength_m = 0.0565642
    look_side = LookSide.RIGHT
    max_iterations = 50
    threshold = 1e-8
    delta_range = 10.0
    
    print('=' * 80)
    print('调试完整的 geo2rdr 函数')
    print('=' * 80)
    
    print(f'\n输入参数:')
    print(f'  lat={lat:.6f}, lon={lon:.6f}, hgt={hgt:.2f}')
    print(f'  look_side={look_side}')
    
    # 模拟 geo2rdr 的执行流程
    print(f'\n步骤 1: 调用 _geo2rdr_bracket...')
    time_start = max(orbit.reference_epoch, radar_grid.start_time)
    time_end = min(orbit.reference_epoch + orbit.number_of_seconds, radar_grid.end_time)
    
    t_az, slant_range = _geo2rdr_bracket(
        target_ecef,
        orbit,
        doppler_lut,
        wavelength_m,
        look_side,
        time_start,
        time_end,
    )
    
    print(f'  返回: t_az={t_az:.10f}, slant_range={slant_range:.6f}')
    
    print(f'\n步骤 2: 牛顿迭代...')
    dt = 0.0
    for iteration in range(max_iterations):
        t_az_new = t_az - dt
        
        orbit_state = orbit.state_at(t_az_new, allow_extrapolation=True)
        sat_pos_i = orbit_state.position
        vel_i = orbit_state.velocity
        
        rvec = target_ecef - sat_pos_i
        slant_range = np.linalg.norm(rvec)
        
        if iteration == 0:
            print(f'  迭代 {iteration}: 检查 look side...')
            is_valid = check_look_side(rvec, vel_i, sat_pos_i, look_side)
            print(f'    check_look_side 返回: {is_valid}')
            if not is_valid:
                print(f'    look side 检查失败，退出循环')
                break
        
        dt = _compute_doppler_aztime_diff(
            rvec, vel_i, doppler_lut, wavelength_m, t_az_new, slant_range, delta_range
        )
        
        print(f'  迭代 {iteration}: t_az={t_az_new:.12f}, dt={dt:.6e}')
        
        if abs(dt) < threshold:
            print(f'  收敛!')
            t_az = t_az_new
            break
        
        t_az = t_az_new
    
    print(f'\n最终结果:')
    print(f'  t_az={t_az:.10f}')
    print(f'  slant_range={slant_range:.6f}')
    
    # 现在调用真正的 geo2rdr
    print(f'\n{"="*80}')
    print('调用真正的 geo2rdr 函数')
    print('=' * 80)
    
    from i2sar.geometry import geo2rdr
    
    result = geo2rdr(
        lat=lat,
        lon=lon,
        height=hgt,
        radar_grid=radar_grid,
        satellite_position=orbit,
        velocity=None,
        doppler=125.0,
        look_side=LookSide.RIGHT,
    )
    
    print(f'\ngeo2rdr 返回:')
    print(f'  aztime={result[0]}')
    print(f'  range={result[1]}')


if __name__ == '__main__':
    debug_full_geo2rdr()
