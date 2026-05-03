"""
使用 ISCE3 风格的测试数据验证 rdr2geo 和 geo2rdr 的精度和性能
参考 ISCE3 的测试参数和数据格式
"""

import time
import numpy as np

from i2sar.geometry import (
    RadarGrid, rdr2geo, geo2rdr, 
    Rdr2GeoResult, LookSide, WGS84, llh_to_ecef
)
from i2sar.orbit.interpolate import OrbitInterpolator


def generate_isce3_style_orbit(radar_grid):
    """生成 ISCE3 风格的卫星轨道"""
    orbit_duration = radar_grid.number_of_seconds * 1.2
    num_states = 100
    
    times = np.linspace(
        radar_grid.start_time - orbit_duration * 0.1,
        radar_grid.end_time + orbit_duration * 0.1,
        num_states
    )
    
    sat_height = 7078000.0
    sat_speed = 7500.0
    
    positions = np.zeros((num_states, 3))
    velocities = np.zeros((num_states, 3))
    
    for i, t in enumerate(times):
        positions[i] = np.array([sat_height, sat_speed * t, 0.0])
        velocities[i] = np.array([0.0, sat_speed, 0.0])
    
    return OrbitInterpolator(times, positions, velocities)


def determine_look_side(orbit, radar_grid, line=500, pixel=250):
    """确定合适的观察侧"""
    expected_aztime = radar_grid.line_to_azimuth_time(line)
    orbit_state = orbit.state_at(expected_aztime)
    pos = orbit_state.position
    vel = orbit_state.velocity
    
    r = np.array([-100000.0, 0.0, 0.0])
    cross_prod = np.cross(r, vel)
    dot_prod = np.dot(cross_prod, pos)
    
    check = (LookSide.RIGHT == LookSide.LEFT) ^ (dot_prod > 0)
    if check:
        return LookSide.RIGHT
    else:
        return LookSide.LEFT


def test_rdr2geo_geo2rdr_roundtrip():
    """测试 rdr2geo 和 geo2rdr 的往返一致性"""
    print('=' * 80)
    print('ISCE3 风格往返一致性测试')
    print('=' * 80)
    
    radar_grid = RadarGrid(
        length=1000,
        width=500,
        sensing_start_s=0.0,
        prf_hz=1666.67,
        starting_range_m=800000.0,
        range_pixel_spacing_m=2.33,
    )
    
    orbit = generate_isce3_style_orbit(radar_grid)
    
    look_side = determine_look_side(orbit, radar_grid)
    print(f'使用观察侧: {look_side}')
    
    test_cases = [
        {'line': 100, 'pixel': 100},
        {'line': 500, 'pixel': 250},
        {'line': 900, 'pixel': 400},
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
        
        assert not np.isnan(aztime_error), f"geo2rdr 返回 NaN"
        assert abs(aztime_error) < 1e-6, f"方位时间误差过大: {aztime_error}"
        assert abs(range_error) < 1e-3, f"距离误差过大: {range_error}"
    
    print('\n所有往返测试通过!')


def test_performance_comparison():
    """性能对比测试"""
    print('\n' + '=' * 80)
    print('ISCE3 风格性能对比测试')
    print('=' * 80)
    
    radar_grid = RadarGrid(
        length=10000,
        width=5000,
        sensing_start_s=0.0,
        prf_hz=1666.67,
        starting_range_m=800000.0,
        range_pixel_spacing_m=2.33,
    )
    
    orbit = generate_isce3_style_orbit(radar_grid)
    look_side = determine_look_side(orbit, radar_grid)
    
    test_sizes = [100, 1000, 10000]
    
    for n_points in test_sizes:
        np.random.seed(42)
        lines = np.random.uniform(0, radar_grid.length, n_points)
        pixels = np.random.uniform(0, radar_grid.width, n_points)
        
        print(f'\n测试规模: {n_points} 点')
        
        t0 = time.time()
        rdr2geo(
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
        
        lat_arr, lon_arr, hgt_arr = rdr2geo(
            line=lines,
            pixel=pixels,
            radar_grid=radar_grid,
            satellite_position=orbit,
            velocity=None,
            doppler=125.0,
            look_side=look_side,
        )
        
        t0 = time.time()
        geo2rdr(
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


def test_static_orbit():
    """测试静态轨道（无轨道插值器）"""
    print('\n' + '=' * 80)
    print('静态轨道测试')
    print('=' * 80)
    
    radar_grid = RadarGrid(
        length=1000,
        width=500,
        sensing_start_s=0.0,
        prf_hz=1666.67,
        starting_range_m=800000.0,
        range_pixel_spacing_m=2.33,
    )
    
    sat_pos = np.array([7078000.0, 0.0, 0.0])
    vel = np.array([0.0, 7500.0, 0.0])
    
    line = 500.0
    pixel = 250.0
    
    print(f'测试点: line={line}, pixel={pixel}')
    
    lat, lon, hgt = rdr2geo(
        line=line,
        pixel=pixel,
        radar_grid=radar_grid,
        satellite_position=sat_pos,
        velocity=vel,
        doppler=125.0,
    )
    print(f'rdr2geo: lat={lat:.6f}, lon={lon:.6f}, hgt={hgt:.2f}')
    
    aztime, slant_range = geo2rdr(
        lat=lat,
        lon=lon,
        height=hgt,
        radar_grid=radar_grid,
        satellite_position=sat_pos,
        velocity=vel,
        doppler=125.0,
    )
    print(f'geo2rdr: aztime={aztime:.6f} s, range={slant_range:.2f} m')
    
    expected_aztime = radar_grid.start_time + radar_grid.number_of_seconds / 2.0
    print(f'期望方位时间（静态轨道）: {expected_aztime:.6f} s')
    print(f'方位时间误差: {(aztime - expected_aztime)*1e9:.2f} ns')


if __name__ == '__main__':
    test_rdr2geo_geo2rdr_roundtrip()
    test_performance_comparison()
    test_static_orbit()
    
    print('\n' + '=' * 80)
    print('所有 ISCE3 风格测试完成!')
    print('=' * 80)