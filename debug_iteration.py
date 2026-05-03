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


def linear_interpolate(t, t0, t1, v0, v1):
    if t1 == t0:
        return v0
    alpha = (t - t0) / (t1 - t0)
    return v0 + alpha * (v1 - v0)


def orbit_interpolate(t, sat_times, sat_positions, sat_velocities):
    n = len(sat_times)
    
    if n == 1:
        return sat_positions[0], sat_velocities[0]
    
    idx = 0
    for i in range(n - 1):
        if sat_times[i] <= t <= sat_times[i + 1]:
            idx = i
            break
    
    if idx == n - 1:
        idx = n - 2
    
    t0 = sat_times[idx]
    t1 = sat_times[idx + 1]
    
    pos = linear_interpolate(t, t0, t1, sat_positions[idx], sat_positions[idx + 1])
    vel = linear_interpolate(t, t0, t1, sat_velocities[idx], sat_velocities[idx + 1])
    
    return pos, vel


def python_newton_iteration(target_ecef, sat_times, sat_positions, sat_velocities, 
                            doppler, wavelength, max_iterations, threshold):
    print("\n=== Python 牛顿迭代 ===")
    
    t_az = (sat_times[0] + sat_times[-1]) / 2.0
    print(f"初始 t_az = {t_az}")
    
    dt = 0.0
    for iteration in range(max_iterations):
        t_az = t_az - dt
        
        t_az_clamped = max(sat_times[0], min(t_az, sat_times[-1]))
        sat_pos_i, vel_i = orbit_interpolate(t_az_clamped, sat_times, sat_positions, sat_velocities)
        
        rvec = target_ecef - sat_pos_i
        slant_range = np.linalg.norm(rvec)
        
        dopfact = np.dot(rvec, vel_i)
        fdop = 0.5 * wavelength * doppler
        
        c1 = -np.dot(vel_i, vel_i)
        c2 = fdop / slant_range
        fnprime = c1 + c2 * dopfact
        
        fn = dopfact - fdop * slant_range
        dt = fn / fnprime if fnprime != 0 else 0.0
        
        print(f"迭代 {iteration}: t_az={t_az:.6f}, dt={dt:.10f}, range={slant_range:.2f}")
        
        if abs(dt) < threshold:
            break
    
    return t_az


def numba_newton_iteration(target_ecef, sat_times, sat_positions, sat_velocities, 
                           doppler, wavelength, max_iterations, threshold):
    print("\n=== Numba 风格牛顿迭代 ===")
    
    t_az = (sat_times[0] + sat_times[-1]) / 2.0
    print(f"初始 t_az = {t_az}")
    
    dt = 0.0
    for iteration in range(max_iterations):
        t_az = t_az - dt
        
        t_az_clamped = max(sat_times[0], min(t_az, sat_times[-1]))
        sat_pos_i, vel_i = orbit_interpolate(t_az_clamped, sat_times, sat_positions, sat_velocities)
        
        rvec = target_ecef - sat_pos_i
        slant_range = np.linalg.norm(rvec)
        
        dopfact = np.dot(rvec, vel_i)
        fdop = 0.5 * wavelength * doppler
        
        c1 = -np.dot(vel_i, vel_i)
        c2 = fdop / slant_range
        fnprime = c1 + c2 * dopfact
        
        fn = dopfact - fdop * slant_range
        dt = fn / fnprime if fnprime != 0 else 0.0
        
        print(f"迭代 {iteration}: t_az={t_az:.6f}, dt={dt:.10f}, range={slant_range:.2f}")
        
        if abs(dt) < threshold:
            break
    
    return t_az


def main():
    radar_grid, orbit, orbit_times, sat_positions, sat_velocities = generate_test_orbit()
    
    lat, lon, height = 30.0, 120.0, 500.0
    target_ecef = llh_to_ecef(lat, lon, height)
    print(f"目标点 ECEF: {target_ecef}")
    
    doppler = 125.0
    wavelength = 0.0565642
    max_iterations = 50
    threshold = 1e-8
    
    t_py = python_newton_iteration(target_ecef, orbit_times, sat_positions, sat_velocities,
                                   doppler, wavelength, max_iterations, threshold)
    t_nb = numba_newton_iteration(target_ecef, orbit_times, sat_positions, sat_velocities,
                                  doppler, wavelength, max_iterations, threshold)
    
    print(f"\n最终结果对比:")
    print(f"Python: {t_py:.6f}")
    print(f"Numba:  {t_nb:.6f}")
    print(f"差异:   {abs(t_py - t_nb):.10f}")


if __name__ == "__main__":
    main()