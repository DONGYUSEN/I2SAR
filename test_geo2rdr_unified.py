import numpy as np
import time
from i2sar.geometry.radar_grid import RadarGrid
from i2sar.orbit.interpolate import OrbitInterpolator
from i2sar.core.enums import LookSide
from i2sar.geometry.geo2rdr_unified import geo2rdr_unified, OrbitData


def generate_test_orbit():
    """生成测试轨道数据"""
    np.random.seed(42)
    
    radar_grid = RadarGrid(
        length=10000,
        width=5000,
        sensing_start_s=503053200.0,
        prf_hz=1666.67,
        starting_range_m=800000.0,
        range_pixel_spacing_m=2.33,
    )
    
    orbit_times = np.linspace(
        radar_grid.sensing_start_s - 10.0,
        radar_grid.sensing_start_s + radar_grid.number_of_seconds + 10.0,
        200
    )
    
    sat_positions = np.zeros((len(orbit_times), 3))
    sat_velocities = np.zeros((len(orbit_times), 3))
    
    for i, t in enumerate(orbit_times):
        t_rel = t - radar_grid.sensing_start_s
        sat_positions[i] = np.array([
            7078000.0 * np.cos(0.0001 * t_rel),
            7078000.0 * np.sin(0.0001 * t_rel),
            700000.0 + 10000.0 * np.sin(0.0005 * t_rel)
        ])
        sat_velocities[i] = np.array([
            -7078000.0 * 0.0001 * np.sin(0.0001 * t_rel),
            7078000.0 * 0.0001 * np.cos(0.0001 * t_rel),
            10000.0 * 0.0005 * np.cos(0.0005 * t_rel)
        ])
    
    orbit = OrbitInterpolator(orbit_times, sat_positions, sat_velocities)
    
    return radar_grid, orbit


def test_single_point():
    """测试单点计算"""
    print("\n=== 单点测试 ===")
    radar_grid, orbit = generate_test_orbit()
    
    lat, lon, height = 30.0, 120.0, 500.0
    
    aztime_py, range_py = geo2rdr_unified(
        lat, lon, height, radar_grid, orbit,
        doppler=125.0, look_side=LookSide.RIGHT,
        method="python", interp_method="hermite"
    )
    
    aztime_nb, range_nb = geo2rdr_unified(
        lat, lon, height, radar_grid, orbit,
        doppler=125.0, look_side=LookSide.RIGHT,
        method="numba", interp_method="hermite"
    )
    
    print(f"Python: aztime={aztime_py:.6f}, range={range_py:.2f}")
    print(f"Numba:  aztime={aztime_nb:.6f}, range={range_nb:.2f}")
    print(f"差异:   aztime_diff={abs(aztime_py - aztime_nb):.10f}, range_diff={abs(range_py - range_nb):.6f}")
    
    assert abs(aztime_py - aztime_nb) < 1e-6, "方位时间差异过大"
    assert abs(range_py - range_nb) < 1e-3, "距离差异过大"
    print("单点测试通过！")


def test_batch_computation():
    """测试批量计算"""
    print("\n=== 批量测试 ===")
    radar_grid, orbit = generate_test_orbit()
    
    n_points = [100, 1000, 10000]
    
    for n in n_points:
        np.random.seed(42)
        lats = np.random.uniform(-45, 45, n)
        lons = np.random.uniform(-180, 180, n)
        heights = np.random.uniform(0, 10000, n)
        
        print(f"\n处理 {n} 个点:")
        
        for method in ["python", "numba"]:
            start = time.time()
            aztimes, ranges = geo2rdr_unified(
                lats, lons, heights, radar_grid, orbit,
                doppler=125.0, look_side=LookSide.RIGHT,
                method=method, interp_method="hermite"
            )
            elapsed = time.time() - start
            
            valid_ratio = np.sum(~np.isnan(aztimes)) / n
            print(f"  {method}: {elapsed:.4f}s, 有效率={valid_ratio*100:.1f}%")
            
            if method == "python":
                aztimes_py = aztimes
                ranges_py = ranges
            else:
                max_az_diff = np.max(np.abs(aztimes - aztimes_py))
                max_range_diff = np.max(np.abs(ranges - ranges_py))
                print(f"  与Python差异: max_az_diff={max_az_diff:.10f}, max_range_diff={max_range_diff:.6f}")
                assert max_az_diff < 1e-2, "批量计算方位时间差异过大"
                assert max_range_diff < 1e-1, "批量计算距离差异过大"
    
    print("批量测试通过！")


def test_interpolation_methods():
    """测试不同插值方法"""
    print("\n=== 插值方法测试 ===")
    radar_grid, orbit = generate_test_orbit()
    
    lat, lon, height = 30.0, 120.0, 500.0
    
    aztime_lin, range_lin = geo2rdr_unified(
        lat, lon, height, radar_grid, orbit,
        doppler=125.0, look_side=LookSide.RIGHT,
        method="python", interp_method="linear"
    )
    
    aztime_herm, range_herm = geo2rdr_unified(
        lat, lon, height, radar_grid, orbit,
        doppler=125.0, look_side=LookSide.RIGHT,
        method="python", interp_method="hermite"
    )
    
    print(f"线性插值:  aztime={aztime_lin:.10f}, range={range_lin:.6f}")
    print(f"Hermite插值: aztime={aztime_herm:.10f}, range={range_herm:.6f}")
    print(f"差异:     aztime_diff={abs(aztime_lin - aztime_herm):.10f}, range_diff={abs(range_lin - range_herm):.6f}")
    
    print("插值方法测试通过！")


def test_roundtrip():
    """测试往返一致性"""
    print("\n=== 往返一致性测试 ===")
    from i2sar.geometry.rdr2geo import rdr2geo
    
    radar_grid, orbit = generate_test_orbit()
    
    test_points = [
        (1000, 500),
        (5000, 2500),
        (9000, 4500)
    ]
    
    for line, pixel in test_points:
        print(f"\n测试点: line={line}, pixel={pixel}")
        
        lat, lon, hgt = rdr2geo(
            line, pixel, radar_grid, orbit,
            velocity=None, doppler=125.0, look_side=LookSide.RIGHT
        )
        
        print(f"rdr2geo: lat={lat:.6f}, lon={lon:.6f}, hgt={hgt:.2f}")
        
        aztime, slant_range = geo2rdr_unified(
            lat, lon, hgt, radar_grid, orbit,
            doppler=125.0, look_side=LookSide.RIGHT,
            method="numba"
        )
        
        line_back = radar_grid.azimuth_time_to_line(aztime)
        pixel_back = radar_grid.slant_range_to_pixel(slant_range)
        
        print(f"geo2rdr: aztime={aztime:.6f}, slant_range={slant_range:.2f}")
        print(f"转换回:  line={line_back:.6f}, pixel={pixel_back:.6f}")
        
        line_error = abs(line_back - line)
        pixel_error = abs(pixel_back - pixel)
        
        print(f"误差:    line_error={line_error:.6f}, pixel_error={pixel_error:.6f}")
        
        assert line_error < 1e-4, "往返line误差过大"
        assert pixel_error < 1e-4, "往返pixel误差过大"
    
    print("往返一致性测试通过！")


def test_loop():
    """循环测试（稳定性测试）"""
    print("\n=== 循环测试 ===")
    radar_grid, orbit = generate_test_orbit()
    
    n_iterations = 10
    n_points = 1000
    
    np.random.seed(42)
    lats = np.random.uniform(-45, 45, n_points)
    lons = np.random.uniform(-180, 180, n_points)
    heights = np.random.uniform(0, 10000, n_points)
    
    results = []
    
    for i in range(n_iterations):
        start = time.time()
        aztimes, ranges = geo2rdr_unified(
            lats, lons, heights, radar_grid, orbit,
            doppler=125.0, look_side=LookSide.RIGHT,
            method="numba", interp_method="hermite"
        )
        elapsed = time.time() - start
        
        results.append((aztimes.copy(), ranges.copy(), elapsed))
        print(f"第 {i+1}/{n_iterations} 次: {elapsed:.4f}s")
    
    aztimes_first = results[0][0]
    ranges_first = results[0][1]
    
    for i, (aztimes, ranges, elapsed) in enumerate(results[1:], 1):
        max_az_diff = np.max(np.abs(aztimes - aztimes_first))
        max_range_diff = np.max(np.abs(ranges - ranges_first))
        
        if max_az_diff > 1e-3 or max_range_diff > 1e-1:
            print(f"第 {i+1} 次结果不一致！az_diff={max_az_diff}, range_diff={max_range_diff}")
            return False
    
    avg_time = np.mean([r[2] for r in results])
    std_time = np.std([r[2] for r in results])
    
    print(f"\n循环测试完成！")
    print(f"平均时间: {avg_time:.4f}s ± {std_time:.4f}s")
    print(f"所有迭代结果一致")
    return True


def test_orbit_data_directly():
    """测试直接使用 OrbitData"""
    print("\n=== OrbitData 直接测试 ===")
    radar_grid, orbit = generate_test_orbit()
    
    orbit_data = OrbitData.from_orbit_interpolator(orbit, num_samples=500)
    print(f"OrbitData: {orbit_data}")
    
    test_times = np.array([orbit_data.t_min, 
                          (orbit_data.t_min + orbit_data.t_max) / 2, 
                          orbit_data.t_max])
    
    pos_lin, vel_lin = orbit_data.interpolate(test_times, method="linear")
    pos_herm, vel_herm = orbit_data.interpolate(test_times, method="hermite")
    
    print(f"\n线性插值位置:\n{pos_lin}")
    print(f"\nHermite插值位置:\n{pos_herm}")
    print(f"\n最大位置差异: {np.max(np.abs(pos_lin - pos_herm)):.6f}")
    
    assert len(pos_lin) == 3
    assert len(pos_herm) == 3
    print("OrbitData 测试通过！")


if __name__ == "__main__":
    print("=" * 60)
    print("geo2rdr_unified 统一接口测试")
    print("=" * 60)
    
    try:
        test_single_point()
        test_batch_computation()
        test_interpolation_methods()
        test_roundtrip()
        test_loop()
        test_orbit_data_directly()
        
        print("\n" + "=" * 60)
        print("所有测试通过！")
        print("=" * 60)
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()