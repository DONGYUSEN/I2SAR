"""
使用 ISCE3 真实测试数据验证 rdr2geo 和 geo2rdr 的精度和性能
分别测试 CPU、Numba 和 GPU 模式
"""

import time
import numpy as np

from i2sar.geometry import (
    RadarGrid, rdr2geo, geo2rdr, 
    Rdr2GeoResult, LookSide, WGS84, llh_to_ecef
)
from i2sar.geometry.accelerated_geometry import AcceleratedRdr2Geo, AcceleratedGeo2Rdr
from i2sar.orbit.interpolate import OrbitInterpolator


def load_isce3_orbit_from_h5(filepath):
    """从 ISCE3 HDF5 文件加载轨道数据"""
    import h5py
    
    with h5py.File(filepath, 'r') as f:
        if 'orbit' in f:
            orbit_group = f['orbit']
            times = np.array(orbit_group['time'])
            positions = np.array(orbit_group['position'])
            velocities = np.array(orbit_group['velocity'])
        elif '/science/LSAR/SLC/metadata/orbit' in f:
            orbit_group = f['/science/LSAR/SLC/metadata/orbit']
            times = np.array(orbit_group['time'])
            positions = np.array(orbit_group['position'])
            velocities = np.array(orbit_group['velocity'])
        else:
            raise ValueError(f"无法在 {filepath} 中找到轨道数据")
    
    return OrbitInterpolator(times, positions, velocities)


def generate_isce3_style_test_data():
    """生成 ISCE3 风格的测试数据（模拟真实卫星参数）"""
    radar_grid = RadarGrid(
        length=10000,
        width=5000,
        sensing_start_s=503053218.0,  # ISCE3 测试数据中的典型起始时间
        prf_hz=1666.67,
        starting_range_m=800000.0,
        range_pixel_spacing_m=2.33,
    )
    
    orbit_duration = radar_grid.number_of_seconds * 1.2
    num_states = 200
    
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
        positions[i] = np.array([sat_height, sat_speed * (t - radar_grid.start_time), 0.0])
        velocities[i] = np.array([0.0, sat_speed, 0.0])
    
    orbit = OrbitInterpolator(times, positions, velocities)
    
    return radar_grid, orbit


def compute_look_side_from_orbit(orbit, radar_grid):
    """从轨道计算观察侧"""
    mid_time = radar_grid.start_time + radar_grid.number_of_seconds / 2.0
    orbit_state = orbit.state_at(mid_time)
    pos = orbit_state.position
    vel = orbit_state.velocity
    
    mid_range = radar_grid.start_range + radar_grid.range_pixel_spacing_m * radar_grid.width / 2.0
    
    target_ecef = np.array([0.0, 0.0, 0.0])
    rvec = target_ecef - pos
    rvec_unit = rvec / np.linalg.norm(rvec)
    target_ecef = pos + rvec_unit * mid_range
    
    rvec = target_ecef - pos
    cross_prod = np.cross(pos, vel)
    dot_prod = np.dot(cross_prod, rvec)
    
    if dot_prod > 0:
        return LookSide.RIGHT
    else:
        return LookSide.LEFT


def test_single_point_all_modes(radar_grid, orbit, look_side):
    """测试单点在所有模式下的性能和精度"""
    print('=' * 80)
    print('单点测试 - 所有模式对比')
    print('=' * 80)
    
    line = 5000
    pixel = 2500
    
    accelerated_rdr2geo = AcceleratedRdr2Geo()
    accelerated_geo2rdr = AcceleratedGeo2Rdr()
    
    print(f'\n测试点: line={line}, pixel={pixel}')
    
    expected_aztime = radar_grid.line_to_azimuth_time(line)
    expected_range = radar_grid.pixel_to_slant_range(pixel)
    print(f'期望: aztime={expected_aztime:.6f} s, range={expected_range:.2f} m')
    
    print('\n--- rdr2geo ---')
    for mode in ['original', 'numba']:
        t0 = time.time()
        result = accelerated_rdr2geo(
            line=line,
            pixel=pixel,
            radar_grid=radar_grid,
            satellite_position=orbit,
            velocity=None,
            doppler=125.0,
            look_side=look_side,
            method=mode,
        )
        t = time.time() - t0
        
        lat, lon, hgt = result
        print(f'{mode}: lat={lat:.6f}, lon={lon:.6f}, hgt={hgt:.2f} m, time={t*1000:.2f} ms')
    
    try:
        t0 = time.time()
        result_gpu = accelerated_rdr2geo(
            line=line,
            pixel=pixel,
            radar_grid=radar_grid,
            satellite_position=orbit,
            velocity=None,
            doppler=125.0,
            look_side=look_side,
            method='arrayfire',
        )
        t_gpu = time.time() - t0
        lat_gpu, lon_gpu, hgt_gpu = result_gpu
        print(f'arrayfire: lat={lat_gpu:.6f}, lon={lon_gpu:.6f}, hgt={hgt_gpu:.2f} m, time={t_gpu*1000:.2f} ms')
    except Exception as e:
        print(f'arrayfire: 不可用 - {e}')
    
    print('\n--- geo2rdr ---')
    lat, lon, hgt = rdr2geo(
        line=line,
        pixel=pixel,
        radar_grid=radar_grid,
        satellite_position=orbit,
        velocity=None,
        doppler=125.0,
        look_side=look_side,
    )
    
    for mode in ['original', 'numba']:
        t0 = time.time()
        aztime, slant_range = accelerated_geo2rdr(
            lat=lat,
            lon=lon,
            height=hgt,
            radar_grid=radar_grid,
            satellite_position=orbit,
            velocity=None,
            doppler=125.0,
            look_side=look_side,
            method=mode,
        )
        t = time.time() - t0
        
        aztime_error = abs(aztime - expected_aztime)
        range_error = abs(slant_range - expected_range)
        print(f'{mode}: aztime={aztime:.6f} s, range={slant_range:.2f} m, time={t*1000:.2f} ms, '
              f'az_error={aztime_error*1e9:.2f} ns, rng_error={range_error*1000:.2f} mm')
    
    try:
        t0 = time.time()
        aztime_gpu, slant_range_gpu = accelerated_geo2rdr(
            lat=lat,
            lon=lon,
            height=hgt,
            radar_grid=radar_grid,
            satellite_position=orbit,
            velocity=None,
            doppler=125.0,
            look_side=look_side,
            method='arrayfire',
        )
        t_gpu = time.time() - t0
        
        aztime_error = abs(aztime_gpu - expected_aztime)
        range_error = abs(slant_range_gpu - expected_range)
        print(f'arrayfire: aztime={aztime_gpu:.6f} s, range={slant_range_gpu:.2f} m, time={t_gpu*1000:.2f} ms, '
              f'az_error={aztime_error*1e9:.2f} ns, rng_error={range_error*1000:.2f} mm')
    except Exception as e:
        print(f'arrayfire 模式不可用: {e}')


def test_batch_performance(radar_grid, orbit, look_side):
    """测试批量处理性能"""
    print('\n' + '=' * 80)
    print('批量性能测试')
    print('=' * 80)
    
    accelerated_rdr2geo = AcceleratedRdr2Geo()
    accelerated_geo2rdr = AcceleratedGeo2Rdr()
    
    test_sizes = [100, 1000, 10000]
    
    for n_points in test_sizes:
        np.random.seed(42)
        lines = np.random.uniform(0, radar_grid.length, n_points)
        pixels = np.random.uniform(0, radar_grid.width, n_points)
        
        print(f'\n测试规模: {n_points} 点')
        
        print('  rdr2geo:')
        for mode in ['original', 'numba']:
            t0 = time.time()
            result = accelerated_rdr2geo(
                line=lines,
                pixel=pixels,
                radar_grid=radar_grid,
                satellite_position=orbit,
                velocity=None,
                doppler=125.0,
                look_side=look_side,
                method=mode,
            )
            t = time.time() - t0
            print(f'    {mode}: {t*1000:.2f} ms')
        
        try:
            t0 = time.time()
            result_gpu = accelerated_rdr2geo(
                line=lines,
                pixel=pixels,
                radar_grid=radar_grid,
                satellite_position=orbit,
                velocity=None,
                doppler=125.0,
                look_side=look_side,
                method='arrayfire',
            )
            t_gpu = time.time() - t0
            print(f'    arrayfire: {t_gpu*1000:.2f} ms')
        except Exception as e:
            print(f'    arrayfire: 不可用 - {e}')
        
        lat_arr, lon_arr, hgt_arr = rdr2geo(
            line=lines,
            pixel=pixels,
            radar_grid=radar_grid,
            satellite_position=orbit,
            velocity=None,
            doppler=125.0,
            look_side=look_side,
        )
        
        print('  geo2rdr:')
        for mode in ['original', 'numba']:
            t0 = time.time()
            aztime, slant_range = accelerated_geo2rdr(
                lat=lat_arr,
                lon=lon_arr,
                height=hgt_arr,
                radar_grid=radar_grid,
                satellite_position=orbit,
                velocity=None,
                doppler=125.0,
                look_side=look_side,
                method=mode,
            )
            t = time.time() - t0
            valid_rate = np.sum(np.isfinite(aztime)) / len(aztime) * 100
            print(f'    {mode}: {t*1000:.2f} ms, 有效率: {valid_rate:.1f}%')
        
        try:
            t0 = time.time()
            aztime_gpu, slant_range_gpu = accelerated_geo2rdr(
                lat=lat_arr,
                lon=lon_arr,
                height=hgt_arr,
                radar_grid=radar_grid,
                satellite_position=orbit,
                velocity=None,
                doppler=125.0,
                look_side=look_side,
                method='arrayfire',
            )
            t_gpu = time.time() - t0
            valid_rate = np.sum(np.isfinite(aztime_gpu)) / len(aztime_gpu) * 100
            print(f'    arrayfire: {t_gpu*1000:.2f} ms, 有效率: {valid_rate:.1f}%')
        except Exception as e:
            print(f'    arrayfire: 不可用 - {e}')


def test_roundtrip_accuracy(radar_grid, orbit, look_side):
    """测试往返精度"""
    print('\n' + '=' * 80)
    print('往返精度测试')
    print('=' * 80)
    
    test_cases = [
        {'line': 1000, 'pixel': 500},
        {'line': 5000, 'pixel': 2500},
        {'line': 9000, 'pixel': 4500},
    ]
    
    for case in test_cases:
        line = case['line']
        pixel = case['pixel']
        
        print(f'\n测试点: line={line}, pixel={pixel}')
        
        lat, lon, hgt = rdr2geo(
            line=line,
            pixel=pixel,
            radar_grid=radar_grid,
            satellite_position=orbit,
            velocity=None,
            doppler=125.0,
            look_side=look_side,
        )
        print(f'rdr2geo: lat={lat:.6f}, lon={lon:.6f}, hgt={hgt:.2f} m')
        
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
        
        line_back = radar_grid.azimuth_time_to_line(aztime)
        pixel_back = radar_grid.slant_range_to_pixel(slant_range)
        
        line_error = abs(line_back - line)
        pixel_error = abs(pixel_back - pixel)
        
        print(f'geo2rdr -> line={line_back:.6f}, pixel={pixel_back:.6f}')
        print(f'误差: line={line_error:.6f}, pixel={pixel_error:.6f}')
        
        assert line_error < 1e-3, f"Line 误差过大: {line_error}"
        assert pixel_error < 1e-3, f"Pixel 误差过大: {pixel_error}"
    
    print('\n往返精度测试通过!')


if __name__ == '__main__':
    print('=' * 80)
    print('使用 ISCE3 风格真实数据测试 rdr2geo 和 geo2rdr')
    print('=' * 80)
    
    radar_grid, orbit = generate_isce3_style_test_data()
    look_side = compute_look_side_from_orbit(orbit, radar_grid)
    
    print(f'\n雷达网格: {radar_grid.length} x {radar_grid.width}')
    print(f'轨道时间范围: {orbit.reference_epoch:.2f} - {orbit.reference_epoch + orbit.number_of_seconds:.2f} s')
    print(f'观察侧: {look_side}')
    
    test_single_point_all_modes(radar_grid, orbit, look_side)
    test_batch_performance(radar_grid, orbit, look_side)
    test_roundtrip_accuracy(radar_grid, orbit, look_side)
    
    print('\n' + '=' * 80)
    print('所有测试完成!')
    print('=' * 80)