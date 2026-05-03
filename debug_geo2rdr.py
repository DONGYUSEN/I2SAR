"""调试 geo2rdr 对于 ISCE3 真实轨道数据返回 NaN 的问题"""

import numpy as np
import xml.etree.ElementTree as ET

from i2sar.geometry import RadarGrid, geo2rdr, LookSide, llh_to_ecef
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


def debug_geo2rdr_with_isce3_orbit():
    orbit_xml_path = '/home/ysdong/Software/isce/isce3/tests/data/orbit.xml'
    times, positions, velocities = parse_isce3_orbit_xml(orbit_xml_path)
    
    orbit = OrbitInterpolator(times, positions, velocities)
    
    orbit_duration = times[-1] - times[0]
    radar_duration = orbit_duration * 0.6
    radar_start = times[0] + orbit_duration * 0.2
    
    radar_grid = RadarGrid(
        length=500,
        width=500,
        sensing_start_s=radar_start,
        prf_hz=1666.67,
        starting_range_m=800000.0,
        range_pixel_spacing_m=2.33,
    )
    
    print(f'轨道时间范围: {times[0]} 到 {times[-1]}')
    print(f'雷达时间范围: {radar_grid.start_time} 到 {radar_grid.end_time}')
    
    mid_time = radar_grid.start_time + radar_grid.number_of_seconds / 2.0
    orbit_state = orbit.state_at(mid_time)
    pos = orbit_state.position
    vel = orbit_state.velocity
    
    mid_range = (radar_grid.start_range + radar_grid.end_range) / 2.0
    
    target_ecef = llh_to_ecef(0.0, 0.0, 0.0)
    rvec = target_ecef - pos
    rvec_unit = rvec / np.linalg.norm(rvec)
    
    target_ecef = pos + rvec_unit * mid_range
    
    r = target_ecef - pos
    
    det = np.dot(np.cross(r, vel), pos)
    
    print(f'debug: det={det}')
    
    if det > 0:
        look_side = LookSide.RIGHT
    else:
        look_side = LookSide.LEFT
    
    print(f'计算得到的观察侧: {look_side}')
    
    from i2sar.geometry import rdr2geo
    
    line = 250
    pixel = 250
    
    lat, lon, hgt = rdr2geo(
        line=line,
        pixel=pixel,
        radar_grid=radar_grid,
        satellite_position=orbit,
        velocity=None,
        doppler=125.0,
        look_side=look_side,
    )
    
    print(f'\nrdr2geo 结果:')
    print(f'lat={lat:.6f}, lon={lon:.6f}, hgt={hgt:.2f}')
    
    target_ecef = llh_to_ecef(lat, lon, hgt)
    print(f'目标点 ECEF: {target_ecef}')
    
    expected_aztime = radar_grid.line_to_azimuth_time(line)
    orbit_state = orbit.state_at(expected_aztime)
    pos_at_expected = orbit_state.position
    vel_at_expected = orbit_state.velocity
    
    print(f'\n在期望方位时间 {expected_aztime}:')
    print(f'卫星位置: {pos_at_expected}')
    print(f'卫星速度: {vel_at_expected}')
    
    r = target_ecef - pos_at_expected
    det = np.dot(np.cross(r, vel_at_expected), pos_at_expected)
    
    print(f'\nr = {r}')
    print(f'cross(r, vel) = {np.cross(r, vel_at_expected)}')
    print(f'det = {det}')
    
    if det > 0:
        actual_side = LookSide.RIGHT
    else:
        actual_side = LookSide.LEFT
    
    print(f'实际观察侧: {actual_side}')
    print(f'指定观察侧: {look_side}')
    
    print(f'\n尝试调用 geo2rdr:')
    try:
        aztime, slant_range = geo2rdr(
            lat=lat,
            lon=lon,
            height=hgt,
            radar_grid=radar_grid,
            satellite_position=orbit,
            velocity=None,
            doppler=125.0,
            look_side=look_side,
        )
        print(f'成功! aztime={aztime}, slant_range={slant_range}')
    except Exception as e:
        print(f'失败: {e}')


if __name__ == '__main__':
    debug_geo2rdr_with_isce3_orbit()
