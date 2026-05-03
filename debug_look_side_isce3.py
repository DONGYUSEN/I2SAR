"""调试 ISCE3 风格的 look side 判断"""

import numpy as np
import xml.etree.ElementTree as ET

from i2sar.geometry import RadarGrid, rdr2geo, llh_to_ecef, check_look_side, LookSide
from i2sar.orbit.interpolate import OrbitInterpolator


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


def debug_look_side():
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
    
    print('=' * 80)
    print('调试 ISCE3 风格的 look side 判断')
    print('=' * 80)
    
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
    print(f'\n目标点 (rdr2geo结果):')
    print(f'  lat={lat:.6f}, lon={lon:.6f}, hgt={hgt:.2f}')
    print(f'  ECEF: {target_ecef}')
    
    expected_aztime = radar_grid.line_to_azimuth_time(mid_line)
    print(f'\n期望方位时间: {expected_aztime}')
    
    orbit_state = orbit.state_at(expected_aztime)
    satpos = orbit_state.position
    satvel = orbit_state.velocity
    dr = target_ecef - satpos
    
    print(f'\n在期望方位时间的卫星状态:')
    print(f'  satpos: {satpos}')
    print(f'  satvel: {satvel}')
    print(f'  dr = target - satpos: {dr}')
    
    cross_prod = np.cross(dr, satvel)
    dot_prod = np.dot(cross_prod, satpos)
    print(f'\nISCE3 风格计算:')
    print(f'  cross(dr, satvel) = {cross_prod}')
    print(f'  dot(cross(dr, satvel), satpos) = {dot_prod}')
    print(f'  dot_prod > 0: {dot_prod > 0}')
    
    print(f'\nISCE3 判定逻辑:')
    print(f'  LookSide::Right ^ (dot_prod > 0) = {LookSide.RIGHT == LookSide.RIGHT} ^ {dot_prod > 0} = {(LookSide.RIGHT == LookSide.RIGHT) ^ (dot_prod > 0)}')
    print(f'  如果结果为 True，说明 look side 不匹配，返回错误')
    
    print(f'\n使用 check_look_side 函数:')
    result_right = check_look_side(dr, satvel, satpos, LookSide.RIGHT)
    result_left = check_look_side(dr, satvel, satpos, LookSide.LEFT)
    print(f'  check_look_side(RIGHT) = {result_right}')
    print(f'  check_look_side(LEFT) = {result_left}')
    
    print(f'\n结论:')
    if dot_prod > 0:
        print('  dot_prod > 0，ISCE3 认为这是 LEFT 侧')
        print('  正确的 look_side 应该是 LEFT')
    else:
        print('  dot_prod <= 0，ISCE3 认为这是 RIGHT 侧')
        print('  正确的 look_side 应该是 RIGHT')


if __name__ == '__main__':
    debug_look_side()
