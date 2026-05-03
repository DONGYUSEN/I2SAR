"""调试牛顿迭代过程"""

import numpy as np
import xml.etree.ElementTree as ET

from i2sar.geometry import RadarGrid, rdr2geo, llh_to_ecef, LookSide
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


def debug_newton():
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
    
    print('=' * 80)
    print('调试牛顿迭代过程')
    print('=' * 80)
    
    print(f'\n目标点: lat={lat:.6f}, lon={lon:.6f}, hgt={hgt:.2f}')
    print(f'期望方位时间: {radar_grid.line_to_azimuth_time(mid_line):.6f}')
    print(f'期望斜距: {radar_grid.pixel_to_slant_range(mid_pixel):.2f}')
    
    tstart = orbit.time[0]
    tend = orbit.time[-1]
    
    times_test = np.linspace(tstart, tend, 15)
    sat_positions, sat_velocities = orbit.state_at_many(times_test)
    
    range_min = radar_grid.start_range
    range_max = radar_grid.end_range
    
    slant_range_closest = 1.0e16
    aztime_closest = -1.0
    
    for k in range(len(times_test)):
        tt = times_test[k]
        pos = sat_positions[k]
        rvec = target_ecef - pos
        slant_range = np.linalg.norm(rvec)
        
        if range_min <= slant_range <= range_max and slant_range < slant_range_closest:
            slant_range_closest = slant_range
            aztime_closest = tt
    
    print(f'\n初始估计: aztime={aztime_closest:.6f}, range={slant_range_closest:.2f}')
    
    wavelength_m = 0.0565642
    doppler = 125.0
    delta_range = 10.0
    max_iterations = 50
    threshold = 1e-8
    
    aztime = aztime_closest
    
    print(f'\n牛顿迭代过程:')
    
    for iteration in range(max_iterations):
        orbit_state = orbit.state_at(aztime)
        satpos = orbit_state.position
        satvel = orbit_state.velocity
        
        rvec = target_ecef - satpos
        slant_range = np.linalg.norm(rvec)
        
        fdop = 0.5 * wavelength_m * doppler
        fdop_der = (0.5 * wavelength_m * doppler - fdop) / delta_range
        
        dopfact = np.dot(rvec, satvel)
        fn = dopfact - fdop * slant_range
        
        c1 = -np.dot(satvel, satvel)
        c2 = (fdop / slant_range) + fdop_der
        fnprime = c1 + c2 * dopfact
        
        if fnprime == 0:
            print(f'  迭代 {iteration}: fnprime=0，无法更新')
            break
        
        dt = fn / fnprime
        aztime_new = aztime - dt
        
        print(f'  迭代 {iteration}: aztime={aztime:.12f}, range={slant_range:.6f}, dt={dt:.6e}')
        
        if abs(dt) < threshold:
            print(f'  收敛!')
            break
        
        if aztime_new < tstart or aztime_new > tend:
            print(f'  方位时间超出范围: {aztime_new}')
            break
        
        aztime = aztime_new
    
    print(f'\n最终结果: aztime={aztime:.12f}, range={slant_range:.6f}')


if __name__ == '__main__':
    debug_newton()
