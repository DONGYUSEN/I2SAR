import numpy as np
from i2sar.geometry.radar_grid import RadarGrid
from i2sar.orbit.interpolate import OrbitInterpolator
from i2sar.orbit.orbit_data import OrbitData


def generate_test_orbit():
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
    
    return radar_grid, orbit, orbit_times, sat_positions, sat_velocities


def test_lagrange():
    print("=== Lagrange 插值测试 ===")
    
    radar_grid, orbit, orbit_times, sat_positions, sat_velocities = generate_test_orbit()
    
    orbit_data = OrbitData.from_orbit_interpolator(orbit, num_samples=100)
    print(f"OrbitData: {orbit_data}")
    
    test_times = np.array([orbit_data.times[10], 
                          (orbit_data.times[10] + orbit_data.times[11]) / 2, 
                          orbit_data.times[50]])
    
    print(f"\n测试时间点: {test_times}")
    
    # 测试线性插值
    pos_lin, vel_lin = orbit_data.interpolate(test_times, method="linear")
    print(f"\n线性插值位置:\n{pos_lin}")
    
    # 测试 Hermite 插值
    pos_herm, vel_herm = orbit_data.interpolate(test_times, method="hermite")
    print(f"\nHermite插值位置:\n{pos_herm}")
    
    # 测试 Lagrange 插值（低阶）
    pos_lagrange, vel_lagrange = orbit_data.interpolate(test_times, method="lagrange", order=3)
    print(f"\nLagrange插值(3阶)位置:\n{pos_lagrange}")
    
    # 比较差异
    print(f"\n线性 vs Hermite 位置差异: {np.max(np.abs(pos_lin - pos_herm)):.6f}")
    print(f"Hermite vs Lagrange 位置差异: {np.max(np.abs(pos_herm - pos_lagrange)):.6f}")


if __name__ == "__main__":
    test_lagrange()