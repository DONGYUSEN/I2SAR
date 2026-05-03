"""调试 look_side 判断逻辑"""

import numpy as np
import xml.etree.ElementTree as ET

from i2sar.geometry import RadarGrid, LookSide
from i2sar.orbit.interpolate import OrbitInterpolator


def parse_isce3_orbit_xml(xml_path):
    """解析 ISCE3 轨道 XML 文件"""
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
    """调试 look_side 判断"""
    orbit_xml_path = '/home/ysdong/Software/isce/isce3/tests/data/orbit.xml'
    times, positions, velocities = parse_isce3_orbit_xml(orbit_xml_path)
    
    orbit = OrbitInterpolator(times, positions, velocities)
    
    mid_time = times[0] + (times[-1] - times[0]) / 2.0
    orbit_state = orbit.state_at(mid_time)
    pos = orbit_state.position
    vel = orbit_state.velocity
    
    print(f'卫星位置 (ECEF): {pos}')
    print(f'卫星速度 (ECEF): {vel}')
    
    target_ecef = np.array([0.0, 0.0, 6371000.0])
    rvec = target_ecef - pos
    print(f'目标点 (ECEF): {target_ecef}')
    print(f'目标到卫星向量 rvec: {rvec}')
    
    cross_prod = np.cross(rvec, vel)
    dot_prod = np.dot(cross_prod, pos)
    
    print(f'cross(rvec, vel) = {cross_prod}')
    print(f'dot(cross_prod, pos) = {dot_prod}')
    print(f'dot_prod > 0: {dot_prod > 0}')
    
    if dot_prod > 0:
        computed_side = LookSide.RIGHT
    else:
        computed_side = LookSide.LEFT
    
    print(f'\n计算得到的观察侧: {computed_side}')
    
    for side in [LookSide.LEFT, LookSide.RIGHT]:
        is_right = (side == LookSide.RIGHT)
        check = is_right ^ (dot_prod > 0)
        print(f'检查 {side}: is_right={is_right}, dot_prod>0={dot_prod>0}, check={check}')
        if check:
            print(f'  -> 观察侧不匹配，会返回错误')
        else:
            print(f'  -> 观察侧匹配')


def test_with_rdr2geo_result():
    """使用 rdr2geo 的结果测试"""
    from i2sar.geometry import rdr2geo, geo2rdr
    
    orbit_xml_path = '/home/ysdong/Software/isce/isce3/tests/data/orbit.xml'
    times, positions, velocities = parse_isce3_orbit_xml(orbit_xml_path)
    
    orbit = OrbitInterpolator(times, positions, velocities)
    
    radar_grid = RadarGrid(
        length=500,
        width=500,
        sensing_start_s=times[0] + (times[-1] - times[0]) * 0.2,
        prf_hz=1666.67,
        starting_range_m=800000.0,
        range_pixel_spacing_m=2.33,
    )
    
    line = 250
    pixel = 250
    
    expected_aztime = radar_grid.line_to_azimuth_time(line)
    orbit_state = orbit.state_at(expected_aztime)
    pos = orbit_state.position
    vel = orbit_state.velocity
    
    print(f'\n在期望方位时间 {expected_aztime}:')
    print(f'卫星位置: {pos}')
    print(f'卫星速度: {vel}')
    
    lat, lon, hgt = rdr2geo(
        line=line,
        pixel=pixel,
        radar_grid=radar_grid,
        satellite_position=orbit,
        velocity=None,
        doppler=125.0,
        look_side=LookSide.LEFT,
    )
    
    print(f'\nrdr2geo 结果:')
    print(f'lat={lat:.6f}, lon={lon:.6f}, hgt={hgt:.2f}')
    
    from i2sar.geometry import llh_to_ecef
    target_ecef = llh_to_ecef(lat, lon, hgt)
    print(f'目标点 ECEF: {target_ecef}')
    
    rvec = target_ecef - pos
    cross_prod = np.cross(rvec, vel)
    dot_prod = np.dot(cross_prod, pos)
    
    print(f'\nrvec = {rvec}')
    print(f'cross(rvec, vel) = {cross_prod}')
    print(f'dot(cross_prod, pos) = {dot_prod}')
    print(f'dot_prod > 0: {dot_prod > 0}')
    
    for side in [LookSide.LEFT, LookSide.RIGHT]:
        is_right = (side == LookSide.RIGHT)
        check = is_right ^ (dot_prod > 0)
        print(f'\n对于观察侧 {side}:')
        print(f'  is_right={is_right}, dot_prod>0={dot_prod>0}')
        print(f'  check={check} (0=匹配, 1=不匹配)')


if __name__ == '__main__':
    debug_look_side()
    test_with_rdr2geo_result()
