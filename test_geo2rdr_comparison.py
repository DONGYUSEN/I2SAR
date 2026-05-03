#!/usr/bin/env python3
"""
Geo2Rdr Comparison Test (ISCE3-style)
Compare CPU, Numba, and ArrayFire GPU implementations
"""

import numpy as np
import time

from i2sar.geometry import RadarGrid, WGS84
from i2sar.geometry.geo2rdr import geo2rdr
from i2sar.geometry.accelerated_geometry import geo2rdr_fast


def run_test():
    print('=' * 80)
    print('Geo2Rdr Comparison Test (ISCE3-style Testing)')
    print('=' * 80)
    
    np.random.seed(42)
    
    # ISCE3风格的测试参数
    # 卫星轨道参数（模拟 Sentinel-1 风格）
    sat_position = np.array([7078000.0, 0.0, 0.0])  # 约 700km 高度
    velocity = np.array([0.0, 7500.0, 0.0])  # 约 7.5 km/s

    # 雷达网格参数
    radar_grid = RadarGrid(
        length=10000,
        width=5000,
        sensing_start_s=0.0,
        prf_hz=1666.67,  # Sentinel-1 IW 模式典型 PRF
        starting_range_m=800000.0,
        range_pixel_spacing_m=2.33,  # 约 2.33m 距离分辨率
    )

    # 测试点配置
    test_configs = [
        {'name': 'Single Point', 'n_points': 1},
        {'name': 'Small Batch', 'n_points': 10},
        {'name': 'Medium Batch', 'n_points': 100},
        {'name': 'Large Batch', 'n_points': 1000},
        {'name': 'Extra Large Batch', 'n_points': 10000},
    ]

    print('\n--- Test Configuration ---')
    print(f'Satellite Position: {sat_position}')
    print(f'Velocity: {velocity}')
    print(f'Radar Grid: length={radar_grid.length}, width={radar_grid.width}')
    print(f'Sensing Start: {radar_grid.sensing_start_s}s, PRF: {radar_grid.prf_hz}Hz')
    print(f'Starting Range: {radar_grid.starting_range_m}m')
    print()

    for config in test_configs:
        n_points = config['n_points']
        print(f'--- {config["name"]} ({n_points} points) ---')
        
        # 生成测试地理坐标
        lats = np.random.uniform(-80, 80, n_points)
        lons = np.random.uniform(-180, 180, n_points)
        heights = np.random.uniform(0, 1000, n_points)
        
        # CPU版本
        start = time.time()
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
        cpu_time = time.time() - start
        
        # Numba版本
        start = time.time()
        result_numba = geo2rdr_fast(
            lat=lats,
            lon=lons,
            height=heights,
            radar_grid=radar_grid,
            satellite_position=sat_position,
            velocity=velocity,
            doppler=0.0,
            method='numba',
        )
        numba_time = time.time() - start
        
        # GPU版本
        result_gpu = None
        gpu_time = -1
        try:
            start = time.time()
            result_gpu = geo2rdr_fast(
                lat=lats,
                lon=lons,
                height=heights,
                radar_grid=radar_grid,
                satellite_position=sat_position,
                velocity=velocity,
                doppler=0.0,
                method='arrayfire',
            )
            gpu_time = time.time() - start
        except Exception as e:
            print(f'  ArrayFire: Not available - {e}')
        
        # 结果对比
        print(f'\n  Time Results:')
        print(f'    CPU:    {cpu_time*1000:.3f} ms')
        print(f'    Numba:  {numba_time*1000:.3f} ms ({cpu_time/numba_time:.2f}x faster)')
        if result_gpu is not None:
            print(f'    GPU:    {gpu_time*1000:.3f} ms ({cpu_time/gpu_time:.2f}x faster)')
        
        print(f'\n  Result Consistency:')
        if n_points == 1:
            aztime_cpu, range_cpu = result_cpu
            aztime_numba, range_numba = result_numba
            
            aztime_diff = abs(aztime_cpu - aztime_numba)
            range_diff = abs(range_cpu - range_numba)
            
            print(f'    CPU aztime:     {aztime_cpu:.15f} s')
            print(f'    Numba aztime:   {aztime_numba:.15f} s')
            print(f'    aztime_diff:    {aztime_diff:.2e} s')
            print(f'    CPU range:      {range_cpu:.10f} m')
            print(f'    Numba range:    {range_numba:.10f} m')
            print(f'    range_diff:     {range_diff:.2e} m')
            
            if result_gpu is not None:
                aztime_gpu, range_gpu = result_gpu
                aztime_diff_gpu = abs(aztime_cpu - aztime_gpu)
                range_diff_gpu = abs(range_cpu - range_gpu)
                
                print(f'    GPU aztime:     {aztime_gpu:.15f} s')
                print(f'    aztime_diff_gpu:{aztime_diff_gpu:.2e} s')
                print(f'    GPU range:      {range_gpu:.10f} m')
                print(f'    range_diff_gpu: {range_diff_gpu:.2e} m')
        else:
            aztime_diff = np.max(np.abs(result_cpu[0] - result_numba[0]))
            range_diff = np.max(np.abs(result_cpu[1] - result_numba[1]))
            
            print(f'    Max aztime diff (CPU vs Numba): {aztime_diff:.2e} s')
            print(f'    Max range diff  (CPU vs Numba): {range_diff:.2e} m')
            print(f'    All finite (CPU):    {np.all(np.isfinite(result_cpu))}')
            print(f'    All finite (Numba):  {np.all(np.isfinite(result_numba))}')
            
            if result_gpu is not None:
                aztime_diff_gpu = np.max(np.abs(result_cpu[0] - result_gpu[0]))
                range_diff_gpu = np.max(np.abs(result_cpu[1] - result_gpu[1]))
                
                print(f'    Max aztime diff (CPU vs GPU):   {aztime_diff_gpu:.2e} s')
                print(f'    Max range diff  (CPU vs GPU):   {range_diff_gpu:.2e} m')
                print(f'    All finite (GPU):    {np.all(np.isfinite(result_gpu))}')
        
        # 验证通过条件
        numba_pass = aztime_diff < 1e-6 and range_diff < 1e-3
        gpu_pass = aztime_diff_gpu < 1e-6 and range_diff_gpu < 1e-3 if result_gpu is not None else True
        
        print(f'\n  Test Results:')
        print(f'    Numba: {"PASS" if numba_pass else "FAIL"}')
        if result_gpu is not None:
            print(f'    GPU:   {"PASS" if gpu_pass else "FAIL"}')
        
        print()

    # 特殊测试：使用ISCE3风格的单个点验证
    print('\n--- ISCE3-style Single Point Validation ---')
    print()
    print('Using specific test point from ISCE3 test patterns:')
    
    # ISCE3风格的测试点
    test_lat = 45.0
    test_lon = 10.0
    test_h = 100.0
    
    print(f'Input: lat={test_lat}, lon={test_lon}, h={test_h}')
    print()
    
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
    
    result_numba = geo2rdr_fast(
        lat=test_lat,
        lon=test_lon,
        height=test_h,
        radar_grid=radar_grid,
        satellite_position=sat_position,
        velocity=velocity,
        doppler=0.0,
        method='numba',
    )
    
    print(f'CPU:    aztime={result_cpu[0]:.15f} s, slant_range={result_cpu[1]:.10f} m')
    print(f'Numba:  aztime={result_numba[0]:.15f} s, slant_range={result_numba[1]:.10f} m')
    
    try:
        result_gpu = geo2rdr_fast(
            lat=test_lat,
            lon=test_lon,
            height=test_h,
            radar_grid=radar_grid,
            satellite_position=sat_position,
            velocity=velocity,
            doppler=0.0,
            method='arrayfire',
        )
        print(f'GPU:    aztime={result_gpu[0]:.15f} s, slant_range={result_gpu[1]:.10f} m')
    except Exception as e:
        print(f'GPU:    Not available - {e}')
    
    print()
    print('=' * 80)
    print('Test completed!')
    print('=' * 80)


if __name__ == '__main__':
    run_test()