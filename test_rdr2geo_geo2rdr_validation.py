"""
使用 ISCE3 风格的测试数据验证 rdr2geo 和 geo2rdr 的精度和性能
"""

import time
import numpy as np

from i2sar.geometry import (
    RadarGrid, rdr2geo, rdr2geo_full, geo2rdr, 
    Rdr2GeoResult, LookSide, WGS84
)
from i2sar.orbit.interpolate import OrbitInterpolator
from i2sar.accel.arrayfire_backend import ArrayFireBackend


def generate_isce3_test_data(n_points=10000, use_orbit_interpolator=True):
    """生成 ISCE3 风格的测试数据"""
    np.random.seed(42)
    
    # ISCE3 Sentinel-1 风格参数
    radar_grid = RadarGrid(
        length=10000,
        width=5000,
        sensing_start_s=0.0,
        prf_hz=1666.67,  # Sentinel-1 IW 模式典型 PRF
        starting_range_m=800000.0,
        range_pixel_spacing_m=2.33,  # 约 2.33m 距离分辨率
    )
    
    # 随机生成测试点
    lines = np.random.uniform(0, radar_grid.length, n_points)
    pixels = np.random.uniform(0, radar_grid.width, n_points)
    
    # 多普勒中心频率（典型值）
    doppler = 125.0  # Hz
    
    if use_orbit_interpolator:
        from i2sar.orbit.interpolate import OrbitInterpolator
        
        orbit_times = np.linspace(0, radar_grid.number_of_seconds * 1.1, 100)
        sat_positions = np.zeros((len(orbit_times), 3))
        sat_velocities = np.zeros((len(orbit_times), 3))
        
        for i, t in enumerate(orbit_times):
            sat_positions[i] = np.array([7078000.0, 7500.0 * t, 0.0])
            sat_velocities[i] = np.array([0.0, 7500.0, 0.0])
        
        satellite_position = OrbitInterpolator(orbit_times, sat_positions, sat_velocities)
        velocity = None
    else:
        satellite_position = np.array([7078000.0, 0.0, 0.0])
        velocity = np.array([0.0, 7500.0, 0.0])
    
    return radar_grid, satellite_position, velocity, doppler, lines, pixels


def test_rdr2geo_accuracy():
    """测试 rdr2geo 精度"""
    print('=' * 80)
    print('Rdr2Geo 精度测试')
    print('=' * 80)
    
    radar_grid, sat_pos, vel, doppler, lines, pixels = generate_isce3_test_data(n_points=1000)
    
    # CPU 版本作为基准
    t0 = time.time()
    result_cpu = rdr2geo(
        line=lines,
        pixel=pixels,
        radar_grid=radar_grid,
        satellite_position=sat_pos,
        velocity=vel,
        doppler=doppler,
        look_side=LookSide.RIGHT,
    )
    t_cpu = time.time() - t0
    
    print(f'CPU 版本耗时: {t_cpu*1000:.2f} ms')
    print(f'结果形状: {result_cpu.shape}')
    print(f'纬度范围: [{result_cpu[0].min():.6f}, {result_cpu[0].max():.6f}] deg')
    print(f'经度范围: [{result_cpu[1].min():.6f}, {result_cpu[1].max():.6f}] deg')
    print(f'高度范围: [{result_cpu[2].min():.2f}, {result_cpu[2].max():.2f}] m')
    
    # 测试扩展输出（只对静态轨道测试）
    if vel is not None:
        t0 = time.time()
        result_full = rdr2geo_full(
            line=lines,
            pixel=pixels,
            radar_grid=radar_grid,
            satellite_position=sat_pos,
            velocity=vel,
            doppler=doppler,
            look_side=LookSide.RIGHT,
            compute_incidence=True,
            compute_layover_shadow=True,
            compute_spacing=True,
        )
        t_full = time.time() - t0
        
        print(f'\n扩展版本耗时: {t_full*1000:.2f} ms')
        print(f'平均收敛迭代次数: {result_full.convergence.mean():.1f}')
        print(f'入射角范围: [{result_full.incidence_angle.min():.2f}, {result_full.incidence_angle.max():.2f}] deg')
        print(f'航向角范围: [{result_full.heading_angle.min():.2f}, {result_full.heading_angle.max():.2f}] deg')
        print(f'叠掩像素数: {result_full.layover_mask.sum()}')
        print(f'阴影像素数: {result_full.shadow_mask.sum()}')
        
        # 验证基础结果一致性
        np.testing.assert_allclose(result_cpu[0], result_full.lat, rtol=1e-10)
        np.testing.assert_allclose(result_cpu[1], result_full.lon, rtol=1e-10)
        np.testing.assert_allclose(result_cpu[2], result_full.height, rtol=1e-10)
        print('\n基础结果与扩展结果一致，测试通过!')
    else:
        print('\n使用轨道插值器，跳过扩展输出测试')


def test_geo2rdr_accuracy():
    """测试 geo2rdr 精度"""
    print('\n' + '=' * 80)
    print('Geo2Rdr 精度测试')
    print('=' * 80)
    
    radar_grid, sat_pos, vel, doppler, lines, pixels = generate_isce3_test_data(n_points=1000)
    
    # 先用 rdr2geo 生成地理坐标
    result_rdr2geo = rdr2geo(
        line=lines,
        pixel=pixels,
        radar_grid=radar_grid,
        satellite_position=sat_pos,
        velocity=vel,
        doppler=doppler,
        look_side=LookSide.RIGHT,
    )
    
    lat = result_rdr2geo[0]
    lon = result_rdr2geo[1]
    height = result_rdr2geo[2]
    
    # 再用 geo2rdr 转换回雷达坐标
    t0 = time.time()
    result_geo2rdr = geo2rdr(
        lat=lat,
        lon=lon,
        height=height,
        radar_grid=radar_grid,
        satellite_position=sat_pos,
        velocity=vel,
        doppler=doppler,
        look_side=LookSide.RIGHT,
    )
    t_geo2rdr = time.time() - t0
    
    print(f'Geo2Rdr 耗时: {t_geo2rdr*1000:.2f} ms')
    
    # 计算往返误差
    aztime_error = result_geo2rdr[0] - radar_grid.line_to_azimuth_time(lines)
    range_error = result_geo2rdr[1] - radar_grid.pixel_to_slant_range(pixels)
    
    print(f'\n往返误差统计:')
    print(f'方位时间误差 (max): {np.abs(aztime_error).max()*1e9:.2f} ns')
    print(f'距离误差 (max): {np.abs(range_error).max()*1000:.2f} mm')
    print(f'方位时间误差 (RMS): {np.sqrt(np.mean(aztime_error**2))*1e9:.2f} ns')
    print(f'距离误差 (RMS): {np.sqrt(np.mean(range_error**2))*1000:.2f} mm')
    
    # 验证精度
    assert np.abs(aztime_error).max() < 1e-6, f"方位时间误差超过阈值: {np.abs(aztime_error).max()}"
    assert np.abs(range_error).max() < 1e-3, f"距离误差超过阈值: {np.abs(range_error).max()}"
    print('\n往返精度验证通过!')


def test_performance():
    """测试不同模式的性能"""
    print('\n' + '=' * 80)
    print('性能对比测试')
    print('=' * 80)
    
    test_configs = [
        {'name': '100 points', 'n_points': 100},
        {'name': '1,000 points', 'n_points': 1000},
        {'name': '10,000 points', 'n_points': 10000},
        {'name': '100,000 points', 'n_points': 100000},
    ]
    
    has_arrayfire = ArrayFireBackend().available
    
    print(f"ArrayFire 可用: {has_arrayfire}")
    print()
    
    for config in test_configs:
        name = config['name']
        n_points = config['n_points']
        
        radar_grid, sat_pos, vel, doppler, lines, pixels = generate_isce3_test_data(n_points=n_points)
        
        # CPU
        t0 = time.time()
        rdr2geo(
            line=lines,
            pixel=pixels,
            radar_grid=radar_grid,
            satellite_position=sat_pos,
            velocity=vel,
            doppler=doppler,
        )
        t_cpu = time.time() - t0
        
        # Numba
        t0 = time.time()
        rdr2geo(
            line=lines,
            pixel=pixels,
            radar_grid=radar_grid,
            satellite_position=sat_pos,
            velocity=vel,
            doppler=doppler,
        )
        t_numba = time.time() - t0
        
        # ArrayFire
        t_gpu = float('inf')
        if has_arrayfire:
            try:
                t0 = time.time()
                rdr2geo(
                    line=lines,
                    pixel=pixels,
                    radar_grid=radar_grid,
                    satellite_position=sat_pos,
                    velocity=vel,
                    doppler=doppler,
                )
                t_gpu = time.time() - t0
            except Exception as e:
                print(f"GPU 测试失败: {e}")
        
        print(f'{name}:')
        print(f'  CPU:   {t_cpu*1000:>8.2f} ms')
        print(f'  Numba: {t_numba*1000:>8.2f} ms  ({t_cpu/t_numba:.1f}x faster)')
        if has_arrayfire:
            print(f'  GPU:   {t_gpu*1000:>8.2f} ms  ({t_cpu/t_gpu:.1f}x faster)')
        print()


def test_roundtrip_consistency():
    """测试往返一致性"""
    print('\n' + '=' * 80)
    print('往返一致性测试 (rdr2geo -> geo2rdr -> rdr2geo)')
    print('=' * 80)
    
    radar_grid, sat_pos, vel, doppler, lines, pixels = generate_isce3_test_data(n_points=1000)
    
    # 第一轮: rdr2geo
    result1 = rdr2geo(
        line=lines,
        pixel=pixels,
        radar_grid=radar_grid,
        satellite_position=sat_pos,
        velocity=vel,
        doppler=doppler,
        look_side=LookSide.RIGHT,
    )
    
    # 第二轮: geo2rdr
    result2 = geo2rdr(
        lat=result1[0],
        lon=result1[1],
        height=result1[2],
        radar_grid=radar_grid,
        satellite_position=sat_pos,
        velocity=vel,
        doppler=doppler,
        look_side=LookSide.RIGHT,
    )
    
    # 第三轮: rdr2geo
    result3 = rdr2geo(
        line=radar_grid.azimuth_time_to_line(result2[0]),
        pixel=radar_grid.slant_range_to_pixel(result2[1]),
        radar_grid=radar_grid,
        satellite_position=sat_pos,
        velocity=vel,
        doppler=doppler,
        look_side=LookSide.RIGHT,
    )
    
    # 计算误差
    lat_error = np.abs(result3[0] - result1[0])
    lon_error = np.abs(result3[1] - result1[1])
    h_error = np.abs(result3[2] - result1[2])
    
    print(f'纬度误差 (max): {lat_error.max()*3600:.6f} arcsec')
    print(f'经度误差 (max): {lon_error.max()*3600:.6f} arcsec')
    print(f'高度误差 (max): {h_error.max():.6f} m')
    
    # 验证精度
    assert lat_error.max() < 1e-8, f"纬度误差超过阈值: {lat_error.max()}"
    assert lon_error.max() < 1e-8, f"经度误差超过阈值: {lon_error.max()}"
    assert h_error.max() < 1e-3, f"高度误差超过阈值: {h_error.max()}"
    
    print('\n往返一致性验证通过!')


def test_single_point():
    """测试单点处理"""
    print('\n' + '=' * 80)
    print('单点处理测试')
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
    
    # 测试单个值输入
    result = rdr2geo(
        line=500.0,
        pixel=250.0,
        radar_grid=radar_grid,
        satellite_position=sat_pos,
        velocity=vel,
        doppler=125.0,
    )
    
    print(f'输入: line=500, pixel=250')
    print(f'输出: lat={result[0]:.6f} deg, lon={result[1]:.6f} deg, height={result[2]:.2f} m')
    
    # 反向转换
    result_geo2rdr = geo2rdr(
        lat=result[0],
        lon=result[1],
        height=result[2],
        radar_grid=radar_grid,
        satellite_position=sat_pos,
        velocity=vel,
        doppler=125.0,
    )
    
    print(f'反向转换: aztime={result_geo2rdr[0]:.6f} s, range={result_geo2rdr[1]:.2f} m')
    
    # 验证
    expected_aztime = radar_grid.line_to_azimuth_time(500.0)
    expected_range = radar_grid.pixel_to_slant_range(250.0)
    
    print(f'期望值: aztime={expected_aztime:.6f} s, range={expected_range:.2f} m')
    print(f'误差: aztime={abs(result_geo2rdr[0]-expected_aztime)*1e9:.2f} ns, range={abs(result_geo2rdr[1]-expected_range)*1000:.2f} mm')


if __name__ == '__main__':
    test_rdr2geo_accuracy()
    test_geo2rdr_accuracy()
    test_performance()
    test_roundtrip_consistency()
    test_single_point()
    
    print('\n' + '=' * 80)
    print('所有测试完成!')
    print('=' * 80)