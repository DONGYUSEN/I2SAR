#!/usr/bin/env python3
"""
Debug Geo2Rdr GPU issue
"""

import numpy as np
from i2sar.geometry import RadarGrid, WGS84, llh_to_ecef
from i2sar.geometry.geo2rdr import geo2rdr
from i2sar.geometry.geometry_arrayfire import geo2rdr_arrayfire_core


def debug_single_point():
    print('=' * 80)
    print('Debug Geo2Rdr GPU - Single Point Analysis')
    print('=' * 80)
    
    # 测试参数
    sat_position = np.array([7078000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7500.0, 0.0])
    
    radar_grid = RadarGrid(
        length=10000,
        width=5000,
        sensing_start_s=0.0,
        prf_hz=1666.67,
        starting_range_m=800000.0,
        range_pixel_spacing_m=2.33,
    )
    
    test_lat = 45.0
    test_lon = 10.0
    test_h = 100.0
    
    print(f'\nTest point: lat={test_lat}, lon={test_lon}, h={test_h}')
    print(f'Satellite position: {sat_position}')
    print(f'Velocity: {velocity}')
    print(f'Radar grid mid time: {radar_grid.start_time + radar_grid.number_of_seconds / 2.0}')
    
    # 计算目标点的ECEF坐标
    target_ecef = llh_to_ecef(test_lat, test_lon, test_h)
    print(f'\nTarget ECEF: {target_ecef}')
    
    # 计算rvec
    rvec = target_ecef - sat_position
    print(f'rvec: {rvec}')
    print(f'slant_range: {np.linalg.norm(rvec)}')
    
    # 计算dopfact
    dopfact = np.dot(rvec, velocity)
    print(f'dopfact = rvec · vel = {dopfact}')
    
    # 计算fn和fnprime（当doppler=0时）
    vel_dot_vel = np.dot(velocity, velocity)
    fn = dopfact  # 当doppler=0时，fdop=0
    fnprime = -vel_dot_vel
    dt = fn / fnprime if fnprime != 0 else 0.0
    print(f'\nFor doppler=0:')
    print(f'  fn = dopfact = {fn}')
    print(f'  fnprime = -|vel|² = {fnprime}')
    print(f'  dt = fn / fnprime = {dt}')
    
    # CPU版本结果
    print('\n--- CPU Version ---')
    result_cpu = geo2rdr(
        lat=test_lat,
        lon=test_lon,
        height=test_h,
        radar_grid=radar_grid,
        satellite_position=sat_position,
        velocity=velocity,
        doppler=0.0,
        use_bracket=False,
    )
    print(f'CPU aztime: {result_cpu[0]:.15f} s')
    print(f'CPU range: {result_cpu[1]:.10f} m')
    
    # GPU版本结果（直接调用核心函数）
    print('\n--- GPU Version (direct call) ---')
    try:
        result_gpu = geo2rdr_arrayfire_core(
            lat=test_lat,
            lon=test_lon,
            height=test_h,
            radar_grid=radar_grid,
            satellite_position=sat_position,
            velocity=velocity,
            doppler=0.0,
        )
        print(f'GPU aztime: {result_gpu[0]:.15f} s')
        print(f'GPU range: {result_gpu[1]:.10f} m')
    except Exception as e:
        print(f'GPU error: {e}')
    
    print('\n' + '=' * 80)


def debug_batch_issue():
    print('\n\n' + '=' * 80)
    print('Debug Geo2Rdr GPU - Batch Processing Issue')
    print('=' * 80)
    
    sat_position = np.array([7078000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7500.0, 0.0])
    
    radar_grid = RadarGrid(
        length=10000,
        width=5000,
        sensing_start_s=0.0,
        prf_hz=1666.67,
        starting_range_m=800000.0,
        range_pixel_spacing_m=2.33,
    )
    
    # 使用固定的随机种子
    np.random.seed(42)
    n_points = 10
    lats = np.random.uniform(-80, 80, n_points)
    lons = np.random.uniform(-180, 180, n_points)
    heights = np.random.uniform(0, 1000, n_points)
    
    print(f'\nTesting {n_points} points')
    
    # CPU版本
    result_cpu = geo2rdr(
        lat=lats,
        lon=lons,
        height=heights,
        radar_grid=radar_grid,
        satellite_position=sat_position,
        velocity=velocity,
        doppler=0.0,
        use_bracket=False,
    )
    
    # GPU版本
    result_gpu = geo2rdr_arrayfire_core(
        lat=lats,
        lon=lons,
        height=heights,
        radar_grid=radar_grid,
        satellite_position=sat_position,
        velocity=velocity,
        doppler=0.0,
    )
    
    print('\nResults comparison:')
    print('-' * 60)
    print(f'{"Point":>6} {"CPU aztime":>18} {"GPU aztime":>18} {"Diff":>12}')
    print('-' * 60)
    
    for i in range(n_points):
        cpu_az = result_cpu[0, i]
        gpu_az = result_gpu[0, i]
        diff = abs(cpu_az - gpu_az)
        print(f'{i:6d} {cpu_az:18.6f} {gpu_az:18.6f} {diff:12.3e}')
    
    print('\nMax differences:')
    print(f'  aztime diff: {np.max(np.abs(result_cpu[0] - result_gpu[0])):.3e} s')
    print(f'  range diff:  {np.max(np.abs(result_cpu[1] - result_gpu[1])):.3e} m')


if __name__ == '__main__':
    debug_single_point()
    debug_batch_issue()