"""调试 _geo2rdr_bracket 方法"""

import numpy as np
import xml.etree.ElementTree as ET

from i2sar.geometry import RadarGrid, rdr2geo, llh_to_ecef, LookSide, check_look_side
from i2sar.orbit.interpolate import OrbitInterpolator
from i2sar.geometry.geo2rdr import _geo2rdr_bracket, _make_doppler_lut


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


def debug_bracket():
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
    
    print('=' * 80)
    print('调试 _geo2rdr_bracket 方法')
    print('=' * 80)
    
    print(f'\n目标点:')
    print(f'  lat={lat:.6f}, lon={lon:.6f}, hgt={hgt:.2f}')
    print(f'  ECEF: {target_ecef}')
    
    print(f'\n雷达网格时间范围:')
    print(f'  start: {radar_grid.start_time}')
    print(f'  end: {radar_grid.end_time}')
    
    print(f'\n轨道时间范围:')
    print(f'  start: {orbit.reference_epoch}')
    print(f'  end: {orbit.reference_epoch + orbit.number_of_seconds}')
    
    time_start = max(orbit.reference_epoch, radar_grid.start_time)
    time_end = min(orbit.reference_epoch + orbit.number_of_seconds, radar_grid.end_time)
    
    print(f'\n搜索时间范围:')
    print(f'  time_start: {time_start}')
    print(f'  time_end: {time_end}')
    
    def doppler_error(t: float) -> float:
        orbit_state = orbit.state_at(t, allow_extrapolation=True)
        xp = orbit_state.position
        v = orbit_state.velocity
        r = target_ecef - xp
        rnorm = np.linalg.norm(r)
        fd = doppler_lut.eval(t, rnorm)
        return 2.0 / wavelength_m * np.dot(v, r) / rnorm - fd
    
    print(f'\n多普勒误差函数值:')
    test_times = np.linspace(time_start, time_end, 10)
    for tt in test_times:
        err = doppler_error(tt)
        print(f'  t={tt:.6f}: error={err:.6f}')
    
    print(f'\n调用 _geo2rdr_bracket...')
    t_az, slant_range = _geo2rdr_bracket(
        target_ecef,
        orbit,
        doppler_lut,
        wavelength_m,
        look_side,
        time_start,
        time_end,
    )
    
    print(f'\n返回结果:')
    print(f'  t_az={t_az:.10f}')
    print(f'  slant_range={slant_range:.6f}')
    
    orbit_state = orbit.state_at(t_az)
    satpos = orbit_state.position
    satvel = orbit_state.velocity
    rvec = target_ecef - satpos
    
    print(f'\n验证 look side:')
    cross_prod = np.cross(rvec, satvel)
    dot_prod = np.dot(cross_prod, satpos)
    print(f'  dot(cross(rvec, satvel), satpos) = {dot_prod}')
    print(f'  look_side=RIGHT: {look_side == LookSide.RIGHT}')
    print(f'  dot_prod > 0: {dot_prod > 0}')
    print(f'  (RIGHT) ^ (dot_prod > 0) = {(look_side == LookSide.RIGHT) ^ (dot_prod > 0)}')
    
    is_valid = check_look_side(rvec, satvel, satpos, look_side)
    print(f'  check_look_side 返回: {is_valid}')


if __name__ == '__main__':
    debug_bracket()
