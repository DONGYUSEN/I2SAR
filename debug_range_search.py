"""调试范围搜索问题"""

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


def debug_range_search():
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
    print('调试范围搜索问题')
    print('=' * 80)
    
    print(f'\n雷达网格参数:')
    print(f'  range_min = {radar_grid.start_range:.0f} m')
    print(f'  range_max = {radar_grid.end_range:.0f} m')
    
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
    print(f'\n目标点:')
    print(f'  lat={lat:.6f}, lon={lon:.6f}, hgt={hgt:.2f}')
    print(f'  ECEF: {target_ecef}')
    
    tstart = orbit.time[0]
    tend = orbit.time[-1]
    times_test = np.linspace(tstart, tend, 15)
    sat_positions, sat_velocities = orbit.state_at_many(times_test)
    
    print(f'\n轨道时间范围: {tstart} 到 {tend}')
    print(f'测试时间点数: {len(times_test)}')
    
    range_min = radar_grid.start_range
    range_max = radar_grid.end_range
    
    print(f'\n范围搜索详情:')
    slant_range_closest = 1.0e16
    aztime_closest = -1.0
    
    for k in range(len(times_test)):
        tt = times_test[k]
        pos = sat_positions[k]
        rvec = target_ecef - pos
        slant_range = np.linalg.norm(rvec)
        
        in_range = range_min <= slant_range <= range_max
        
        print(f'  k={k}: time={tt:.2f}, slant_range={slant_range:.0f} m, in_range={in_range}')
        
        if in_range and slant_range < slant_range_closest:
            slant_range_closest = slant_range
            aztime_closest = tt
    
    print(f'\n搜索结果:')
    print(f'  aztime_closest = {aztime_closest}')
    print(f'  slant_range_closest = {slant_range_closest:.0f} m')
    
    if aztime_closest < 0:
        print('  警告: 没有找到有效的初始估计!')


if __name__ == '__main__':
    debug_range_search()
