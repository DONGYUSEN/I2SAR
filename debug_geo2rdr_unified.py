import numpy as np
from i2sar.geometry.radar_grid import RadarGrid
from i2sar.orbit.interpolate import OrbitInterpolator
from i2sar.core.enums import LookSide
from i2sar.geometry.geo2rdr_unified import geo2rdr_unified, OrbitData, _find_closest_aztime_batch


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
    
    return radar_grid, orbit


def llh_to_ecef(lat, lon, h, a=6378137.0, b=6356752.314245):
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    N = a / np.sqrt(1 - (1 - (b/a)**2) * sin_lat**2)
    
    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (b/a)**2 + h) * sin_lat
    
    return np.array([x, y, z])


def check_look_side(rvec, vel, pos):
    cross_prod = np.cross(rvec, vel)
    dot_prod = np.dot(cross_prod, pos)
    return dot_prod > 0


def main():
    radar_grid, orbit = generate_test_orbit()
    
    lat, lon, height = 30.0, 120.0, 500.0
    
    orbit_data = OrbitData.from_orbit_interpolator(orbit)
    
    target_ecef = llh_to_ecef(lat, lon, height)
    print(f"目标点 ECEF: {target_ecef}")
    
    look_side_right = True
    
    closest_times = _find_closest_aztime_batch(
        np.array([target_ecef]), 
        orbit_data.positions, 
        orbit_data.velocities, 
        orbit_data.times, 
        look_side_right
    )
    print(f"Python 找到的最近时间: {closest_times[0]}")
    
    print("\n检查各轨道点的 look_side:")
    valid_count = 0
    for i, (pos, vel, t) in enumerate(zip(orbit_data.positions, orbit_data.velocities, orbit_data.times)):
        rvec = target_ecef - pos
        is_valid = check_look_side(rvec, vel, pos)
        if is_valid == look_side_right:
            valid_count += 1
            r = np.linalg.norm(rvec)
            if i % 20 == 0:
                print(f"  t={t:.2f}: valid={is_valid}, r={r:.2f}")
    
    print(f"\n总共有 {valid_count}/{len(orbit_data.times)} 个有效点")
    
    aztime_py, range_py = geo2rdr_unified(
        lat, lon, height, radar_grid, orbit,
        doppler=125.0, look_side=LookSide.RIGHT,
        method="python", interp_method="linear"
    )
    print(f"\nPython 结果: aztime={aztime_py:.6f}, range={range_py:.2f}")
    
    aztime_nb, range_nb = geo2rdr_unified(
        lat, lon, height, radar_grid, orbit,
        doppler=125.0, look_side=LookSide.RIGHT,
        method="numba", interp_method="linear"
    )
    print(f"Numba 结果: aztime={aztime_nb:.6f}, range={range_nb:.2f}")


if __name__ == "__main__":
    main()