import numpy as np
from i2sar.geometry.radar_grid import RadarGrid
from i2sar.orbit.interpolate import OrbitInterpolator
from i2sar.core.enums import LookSide
from i2sar.geometry.geo2rdr_unified import geo2rdr_unified, OrbitData
from i2sar.geometry.geo2rdr_numba import geo2rdr_numba_parallel


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


def main():
    radar_grid, orbit, orbit_times, sat_positions, sat_velocities = generate_test_orbit()
    
    lat, lon, height = 30.0, 120.0, 500.0
    
    orbit_data = OrbitData.from_orbit_interpolator(orbit)
    
    print(f"轨道时间范围: [{orbit_data.times[0]}, {orbit_data.times[-1]}]")
    print(f"轨道中点时间: {(orbit_data.times[0] + orbit_data.times[-1])/2}")
    
    lat_arr = np.array([lat])
    lon_arr = np.array([lon])
    h_arr = np.array([height])
    
    aztime_py, range_py = geo2rdr_unified(
        lat, lon, height, radar_grid, orbit,
        doppler=125.0, look_side=LookSide.RIGHT,
        method="python", interp_method="linear"
    )
    print(f"\nPython 结果: aztime={aztime_py:.10f}, range={range_py:.2f}")
    
    aztime_nb, range_nb = geo2rdr_numba_parallel(
        lat_arr=lat_arr,
        lon_arr=lon_arr,
        h_arr=h_arr,
        sat_positions=orbit_data.positions,
        sat_velocities=orbit_data.velocities,
        sat_times=orbit_data.times,
        doppler=125.0,
        wavelength=0.0565642,
        a=6378137.0,
        b=6356752.314245,
        max_iterations=50,
        threshold=1e-8,
        sensing_start_s=float(radar_grid.sensing_start_s),
        prf_hz=float(radar_grid.prf_hz),
        length=int(radar_grid.length),
        look_side_right=True
    )
    print(f"Numba 结果: aztime={aztime_nb[0]:.10f}, range={range_nb[0]:.2f}")
    
    print(f"\n差异: aztime_diff={abs(aztime_py - aztime_nb[0]):.10f}, range_diff={abs(range_py - range_nb[0]):.6f}")


if __name__ == "__main__":
    main()