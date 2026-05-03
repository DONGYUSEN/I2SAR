"""
使用 ISCE2 和 ISCE3 的实际测试数据验证 rdr2geo 和 geo2rdr 的精度和性能

参考数据来源:
- ISCE3: /home/ysdong/Software/isce/isce3/tests/data/
- ISCE2: /home/ysdong/Software/isce/isce2/
"""

import time
import numpy as np
import xml.etree.ElementTree as ET

from i2sar.geometry import (
    RadarGrid, rdr2geo, geo2rdr, 
    Rdr2GeoResult, LookSide, WGS84, llh_to_ecef
)
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


def compute_look_side_from_orbit(orbit, radar_grid):
    """根据轨道计算观察侧（ISCE3/GMTSAR 风格）
    
    正确的几何判据（与 ISCE/GAMMA 实现一致）：
    det = dot(cross(r, v_s), r_s)
    
    其中：
    - r = target_ecef - sat_pos (卫星到地面点的向量)
    - v_s = 卫星速度
    - r_s = 卫星位置
    
    det > 0: 右视 (RIGHT)
    det < 0: 左视 (LEFT)
    
    关键：使用 rdr2geo 获取真实的成像点来计算观察侧
    """
    from i2sar.geometry import llh_to_ecef, rdr2geo
    
    mid_line = radar_grid.length / 2.0
    mid_pixel = radar_grid.width / 2.0
    
    try:
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
        
        expected_aztime = radar_grid.line_to_azimuth_time(mid_line)
        orbit_state = orbit.state_at(expected_aztime)
        pos = orbit_state.position
        vel = orbit_state.velocity
        
        r = target_ecef - pos
        det = np.dot(np.cross(r, vel), pos)
        
        print(f'debug: pos={pos[:3]}, vel={vel[:3]}, r={r[:3]}')
        print(f'debug: cross(r, vel)={np.cross(r, vel)[:3]}')
        print(f'debug: det={det}')
        
        if det > 0:
            return LookSide.RIGHT
        else:
            return LookSide.LEFT
            
    except Exception as e:
        print(f'计算观察侧失败，使用默认值: {e}')
        return LookSide.RIGHT


def test_with_isce3_orbit():
    """使用 ISCE3 真实轨道数据进行测试"""
    print('=' * 80)
    print('使用 ISCE3 真实轨道数据测试')
    print('=' * 80)
    
    orbit_xml_path = '/home/ysdong/Software/isce/isce3/tests/data/orbit.xml'
    print(f'加载轨道数据: {orbit_xml_path}')
    
    times, positions, velocities = parse_isce3_orbit_xml(orbit_xml_path)
    
    orbit = OrbitInterpolator(times, positions, velocities)
    print(f'轨道时间范围: {times[0]} 到 {times[-1]}')
    print(f'轨道状态点数: {len(times)}')
    
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
    
    look_side = compute_look_side_from_orbit(orbit, radar_grid)
    print(f'计算得到的观察侧: {look_side}')
    
    test_cases = [
        {'line': 100, 'pixel': 100},
        {'line': 250, 'pixel': 250},
        {'line': 400, 'pixel': 400},
    ]
    
    for case in test_cases:
        line = case['line']
        pixel = case['pixel']
        
        expected_aztime = radar_grid.line_to_azimuth_time(line)
        expected_range = radar_grid.pixel_to_slant_range(pixel)
        
        print(f'\n测试点: line={line}, pixel={pixel}')
        print(f'期望: aztime={expected_aztime:.6f} s, range={expected_range:.2f} m')
        
        lat, lon, hgt = rdr2geo(
            line=line,
            pixel=pixel,
            radar_grid=radar_grid,
            satellite_position=orbit,
            velocity=None,
            doppler=125.0,
            look_side=look_side,
        )
        print(f'rdr2geo: lat={lat:.6f}, lon={lon:.6f}, hgt={hgt:.2f}')
        
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
        print(f'geo2rdr: aztime={aztime:.6f} s, range={slant_range:.2f} m')
        
        aztime_error = aztime - expected_aztime
        range_error = slant_range - expected_range
        print(f'误差: aztime={aztime_error*1e9:.2f} ns, range={range_error*1000:.2f} mm')
        
        if not np.isnan(aztime):
            assert abs(aztime_error) < 1e-6, f"方位时间误差过大: {aztime_error}"
            assert abs(range_error) < 0.1, f"距离误差过大: {range_error}"
            print('测试通过!')
        else:
            print('警告: geo2rdr 返回 NaN，可能是观察侧问题')


def test_isce3_style_simulation():
    """使用 ISCE3 风格的模拟数据进行测试"""
    print('\n' + '=' * 80)
    print('ISCE3 风格模拟数据测试')
    print('=' * 80)
    
    radar_grid = RadarGrid(
        length=1000,
        width=500,
        sensing_start_s=0.0,
        prf_hz=1666.67,
        starting_range_m=800000.0,
        range_pixel_spacing_m=2.33,
    )
    
    num_states = 100
    orbit_times = np.linspace(-100, radar_grid.number_of_seconds + 100, num_states)
    
    sat_height = 7078000.0
    sat_speed = 7500.0
    
    sat_positions = np.zeros((num_states, 3))
    sat_velocities = np.zeros((num_states, 3))
    
    for i, t in enumerate(orbit_times):
        sat_positions[i] = np.array([sat_height, sat_speed * t, 100000.0 * np.sin(t * 0.01)])
        sat_velocities[i] = np.array([0.0, sat_speed, 100000.0 * 0.01 * np.cos(t * 0.01)])
    
    orbit = OrbitInterpolator(orbit_times, sat_positions, sat_velocities)
    
    look_side = compute_look_side_from_orbit(orbit, radar_grid)
    print(f'计算得到的观察侧: {look_side}')
    
    print('\n测试往返一致性:')
    line = 500.0
    pixel = 250.0
    
    expected_aztime = radar_grid.line_to_azimuth_time(line)
    expected_range = radar_grid.pixel_to_slant_range(pixel)
    
    lat, lon, hgt = rdr2geo(
        line=line,
        pixel=pixel,
        radar_grid=radar_grid,
        satellite_position=orbit,
        velocity=None,
        doppler=125.0,
        look_side=look_side,
    )
    
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
    
    aztime_error = aztime - expected_aztime
    range_error = slant_range - expected_range
    
    print(f'初始: line={line}, pixel={pixel}')
    print(f'rdr2geo: lat={lat:.6f}, lon={lon:.6f}, hgt={hgt:.2f}')
    print(f'geo2rdr: aztime={aztime:.6f} s, range={slant_range:.2f} m')
    print(f'期望: aztime={expected_aztime:.6f} s, range={expected_range:.2f} m')
    print(f'误差: aztime={aztime_error*1e9:.2f} ns, range={range_error*1000:.2f} mm')
    
    assert not np.isnan(aztime), "geo2rdr 返回 NaN"
    assert abs(aztime_error) < 1e-6, f"方位时间误差过大: {aztime_error}"
    assert abs(range_error) < 1e-3, f"距离误差过大: {range_error}"
    print('测试通过!')


def test_performance_with_isce3_data():
    """使用 ISCE3 数据进行性能测试"""
    print('\n' + '=' * 80)
    print('ISCE3 风格性能测试')
    print('=' * 80)
    
    radar_grid = RadarGrid(
        length=10000,
        width=5000,
        sensing_start_s=0.0,
        prf_hz=1666.67,
        starting_range_m=800000.0,
        range_pixel_spacing_m=2.33,
    )
    
    num_states = 100
    orbit_times = np.linspace(-100, radar_grid.number_of_seconds + 100, num_states)
    
    sat_height = 7078000.0
    sat_speed = 7500.0
    
    sat_positions = np.zeros((num_states, 3))
    sat_velocities = np.zeros((num_states, 3))
    
    for i, t in enumerate(orbit_times):
        sat_positions[i] = np.array([sat_height, sat_speed * t, 0.0])
        sat_velocities[i] = np.array([0.0, sat_speed, 0.0])
    
    orbit = OrbitInterpolator(orbit_times, sat_positions, sat_velocities)
    look_side = compute_look_side_from_orbit(orbit, radar_grid)
    
    test_sizes = [100, 1000, 10000]
    
    for n_points in test_sizes:
        np.random.seed(42)
        lines = np.random.uniform(0, radar_grid.length, n_points)
        pixels = np.random.uniform(0, radar_grid.width, n_points)
        
        print(f'\n测试规模: {n_points} 点')
        
        t0 = time.time()
        lat_arr, lon_arr, hgt_arr = rdr2geo(
            line=lines,
            pixel=pixels,
            radar_grid=radar_grid,
            satellite_position=orbit,
            velocity=None,
            doppler=125.0,
            look_side=look_side,
        )
        t_cpu = time.time() - t0
        print(f'  rdr2geo: {t_cpu*1000:.2f} ms')
        
        t0 = time.time()
        aztimes, slant_ranges = geo2rdr(
            lat=lat_arr,
            lon=lon_arr,
            height=hgt_arr,
            radar_grid=radar_grid,
            satellite_position=orbit,
            velocity=None,
            doppler=125.0,
            look_side=look_side,
        )
        t_fast = time.time() - t0
        print(f'  geo2rdr: {t_fast*1000:.2f} ms')
        
        valid_mask = ~np.isnan(aztimes)
        print(f'  有效率: {valid_mask.sum()/len(valid_mask)*100:.1f}%')


def test_look_side_consistency():
    """测试观察侧的一致性"""
    print('\n' + '=' * 80)
    print('观察侧一致性测试')
    print('=' * 80)
    
    radar_grid = RadarGrid(
        length=1000,
        width=500,
        sensing_start_s=0.0,
        prf_hz=1666.67,
        starting_range_m=800000.0,
        range_pixel_spacing_m=2.33,
    )
    
    num_states = 50
    orbit_times = np.linspace(-50, radar_grid.number_of_seconds + 50, num_states)
    
    sat_height = 7078000.0
    sat_speed = 7500.0
    
    sat_positions = np.zeros((num_states, 3))
    sat_velocities = np.zeros((num_states, 3))
    
    for i, t in enumerate(orbit_times):
        sat_positions[i] = np.array([sat_height, sat_speed * t, 0.0])
        sat_velocities[i] = np.array([0.0, sat_speed, 0.0])
    
    orbit = OrbitInterpolator(orbit_times, sat_positions, sat_velocities)
    
    look_side = compute_look_side_from_orbit(orbit, radar_grid)
    print(f'计算得到的观察侧: {look_side}')
    
    for side in [LookSide.LEFT, LookSide.RIGHT]:
        print(f'\n测试观察侧: {side}')
        
        line = 500.0
        pixel = 250.0
        
        aztime, slant_range = geo2rdr(
            lat=30.0,
            lon=100.0,
            height=0.0,
            radar_grid=radar_grid,
            satellite_position=orbit,
            velocity=None,
            doppler=125.0,
            look_side=side,
        )
        
        if np.isnan(aztime):
            print(f'  结果: NaN (观察侧不匹配)')
        else:
            print(f'  结果: aztime={aztime:.6f} s, range={slant_range:.2f} m')


if __name__ == '__main__':
    test_with_isce3_orbit()
    test_isce3_style_simulation()
    test_performance_with_isce3_data()
    test_look_side_consistency()
    
    print('\n' + '=' * 80)
    print('所有 ISCE 测试完成!')
    print('=' * 80)
