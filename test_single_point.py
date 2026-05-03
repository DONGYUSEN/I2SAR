"""测试单点 geo2rdr"""

import numpy as np
import xml.etree.ElementTree as ET

from i2sar.geometry import RadarGrid, geo2rdr, rdr2geo, llh_to_ecef, LookSide
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


def test_single_point():
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
    print('测试单点 geo2rdr')
    print('=' * 80)
    
    print(f'\n输入坐标:')
    print(f'  lat={lat:.6f}, lon={lon:.6f}, hgt={hgt:.2f}')
    
    print(f'\n期望输出:')
    print(f'  aztime={radar_grid.line_to_azimuth_time(mid_line):.6f}')
    print(f'  range={radar_grid.pixel_to_slant_range(mid_pixel):.2f}')
    
    aztime, slant_range = geo2rdr(
        lat=lat,
        lon=lon,
        height=hgt,
        radar_grid=radar_grid,
        satellite_position=orbit,
        velocity=None,
        doppler=125.0,
        look_side=LookSide.RIGHT,
    )
    
    print(f'\n实际输出:')
    if np.isnan(aztime):
        print('  aztime=NaN, range=NaN')
    else:
        print(f'  aztime={aztime:.10f}')
        print(f'  range={slant_range:.6f}')
        
        aztime_error = abs(aztime - radar_grid.line_to_azimuth_time(mid_line))
        range_error = abs(slant_range - radar_grid.pixel_to_slant_range(mid_pixel))
        print(f'\n误差:')
        print(f'  aztime: {aztime_error*1e9:.2f} ns')
        print(f'  range: {range_error*1000:.2f} mm')


if __name__ == '__main__':
    test_single_point()
