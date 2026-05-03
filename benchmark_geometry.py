#!/usr/bin/env python3
"""
I2SAR Geometry Algorithms Benchmark (ISCE3-style Testing)
"""

import numpy as np
import time
from i2sar.geometry import RadarGrid, WGS84, llh_to_ecef
from i2sar.geometry.rdr2geo import rdr2geo
from i2sar.geometry.geo2rdr import geo2rdr
from i2sar.geometry.accelerated_geometry import rdr2geo_fast, geo2rdr_fast

def main():
    print('='*80)
    print('I2SAR Geometry Algorithms Benchmark (ISCE3-style Testing)')
    print('='*80)

    # 测试配置
    np.random.seed(42)
    n_test_points = [1, 10, 100, 1000, 10000]

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

    print('\n--- 1. Single Point Test (ISCE3 style) ---')
    print()

    # 单个点测试（类似 ISCE3 的 test_point）
    line = 5000.0
    pixel = 2500.0

    print(f'Test point: line={line}, pixel={pixel}')
    print()

    # CPU 版本
    start = time.time()
    result_cpu = rdr2geo(line=line, pixel=pixel, radar_grid=radar_grid,
                        satellite_position=sat_position, velocity=velocity, doppler=0.0)
    cpu_time = time.time() - start

    print(f'CPU:     lat={result_cpu[0]:.10f}, lon={result_cpu[1]:.10f}, h={result_cpu[2]:.6f}, time={cpu_time*1000:.3f}ms')

    # Numba 版本
    start = time.time()
    result_numba = rdr2geo_fast(line=line, pixel=pixel, radar_grid=radar_grid,
                                satellite_position=sat_position, velocity=velocity,
                                doppler=0.0, method='numba')
    numba_time = time.time() - start

    print(f'Numba:   lat={result_numba[0]:.10f}, lon={result_numba[1]:.10f}, h={result_numba[2]:.6f}, time={numba_time*1000:.3f}ms')

    # ArrayFire 版本
    result_gpu = None
    try:
        start = time.time()
        result_gpu = rdr2geo_fast(line=line, pixel=pixel, radar_grid=radar_grid,
                                  satellite_position=sat_position, velocity=velocity,
                                  doppler=0.0, method='arrayfire')
        gpu_time = time.time() - start
        print(f'ArrayFire: lat={result_gpu[0]:.10f}, lon={result_gpu[1]:.10f}, h={result_gpu[2]:.6f}, time={gpu_time*1000:.3f}ms')
    except Exception as e:
        print(f'ArrayFire: Not available - {e}')

    # 验证一致性
    print()
    print('Result consistency check:')
    lat_diff = abs(result_cpu[0] - result_numba[0])
    lon_diff = abs(result_cpu[1] - result_numba[1])
    h_diff = abs(result_cpu[2] - result_numba[2])
    print(f'  CPU vs Numba: lat_diff={lat_diff:.2e}, lon_diff={lon_diff:.2e}, h_diff={h_diff:.2e}')

    if result_gpu is not None:
        lat_diff_gpu = abs(result_cpu[0] - result_gpu[0])
        lon_diff_gpu = abs(result_cpu[1] - result_gpu[1])
        h_diff_gpu = abs(result_cpu[2] - result_gpu[2])
        print(f'  CPU vs ArrayFire: lat_diff={lat_diff_gpu:.2e}, lon_diff={lon_diff_gpu:.2e}, h_diff={h_diff_gpu:.2e}')

    print()
    print('--- 2. Batch Performance Benchmark ---')
    print()
    print('%8s %12s %12s %12s %15s %12s' % ('Points', 'CPU (ms)', 'Numba (ms)', 'GPU (ms)', 'Numba Speedup', 'GPU Speedup'))
    print('-' * 80)

    for n_points in n_test_points:
        # 生成测试数据
        lines = np.random.uniform(0, radar_grid.length-1, n_points)
        pixels = np.random.uniform(0, radar_grid.width-1, n_points)
        
        # CPU
        start = time.time()
        rdr2geo(line=lines, pixel=pixels, radar_grid=radar_grid,
                satellite_position=sat_position, velocity=velocity, doppler=0.0)
        cpu_time = (time.time() - start) * 1000
        
        # Numba
        start = time.time()
        rdr2geo_fast(line=lines, pixel=pixels, radar_grid=radar_grid,
                     satellite_position=sat_position, velocity=velocity,
                     doppler=0.0, method='numba')
        numba_time = (time.time() - start) * 1000
        
        # ArrayFire
        try:
            start = time.time()
            rdr2geo_fast(line=lines, pixel=pixels, radar_grid=radar_grid,
                         satellite_position=sat_position, velocity=velocity,
                         doppler=0.0, method='arrayfire')
            gpu_time = (time.time() - start) * 1000
            gpu_speedup = cpu_time / gpu_time
        except Exception as e:
            gpu_time = -1
            gpu_speedup = -1
        
        numba_speedup = cpu_time / numba_time
        
        print('%8d %12.2f %12.2f %12.2f %15.2fx %12.2fx' % (n_points, cpu_time, numba_time, gpu_time, numba_speedup, gpu_speedup))

    print()
    print('--- 3. geo2rdr Round-trip Test ---')
    print()

    # 生成地理坐标测试点
    lats = np.array([45.0, 45.1, 45.2])
    lons = np.array([10.0, 10.1, 10.2])
    heights = np.array([100.0, 200.0, 300.0])

    print('Input LLH:')
    for i in range(3):
        print(f'  Point {i+1}: lat={lats[i]}, lon={lons[i]}, h={heights[i]}')

    print()

    # geo2rdr CPU
    start = time.time()
    rdr_cpu = geo2rdr(lat=lats, lon=lons, height=heights, radar_grid=radar_grid,
                      satellite_position=sat_position, velocity=velocity, doppler=0.0)
    cpu_time = time.time() - start
    print(f'geo2rdr CPU: aztime={rdr_cpu[0]}, range={rdr_cpu[1]}, time={cpu_time*1000:.3f}ms')

    # geo2rdr Numba
    start = time.time()
    rdr_numba = geo2rdr_fast(lat=lats, lon=lons, height=heights, radar_grid=radar_grid,
                             satellite_position=sat_position, velocity=velocity,
                             doppler=0.0, method='numba')
    numba_time = time.time() - start
    print(f'geo2rdr Numba: aztime={rdr_numba[0]}, range={rdr_numba[1]}, time={numba_time*1000:.3f}ms')

    # geo2rdr ArrayFire
    try:
        start = time.time()
        rdr_gpu = geo2rdr_fast(lat=lats, lon=lons, height=heights, radar_grid=radar_grid,
                               satellite_position=sat_position, velocity=velocity,
                               doppler=0.0, method='arrayfire')
        gpu_time = time.time() - start
        print(f'geo2rdr ArrayFire: aztime={rdr_gpu[0]}, range={rdr_gpu[1]}, time={gpu_time*1000:.3f}ms')
    except Exception as e:
        print(f'geo2rdr ArrayFire: Not available - {e}')

    # Round-trip check: geo2rdr -> rdr2geo
    print()
    print('Round-trip verification (geo2rdr -> rdr2geo):')
    llh_back = rdr2geo(line=rdr_cpu[0], pixel=((rdr_cpu[1] - radar_grid.starting_range_m) / radar_grid.range_pixel_spacing_m),
                       radar_grid=radar_grid, satellite_position=sat_position, 
                       velocity=velocity, doppler=0.0)
    for i in range(3):
        lat_err = abs(lats[i] - llh_back[0][i])
        lon_err = abs(lons[i] - llh_back[1][i])
        h_err = abs(heights[i] - llh_back[2][i])
        print(f'  Point {i+1}: lat_err={lat_err:.6f}deg, lon_err={lon_err:.6f}deg, h_err={h_err:.3f}m')

    print()
    print('='*80)
    print('Benchmark completed successfully!')
    print('='*80)

if __name__ == '__main__':
    main()
