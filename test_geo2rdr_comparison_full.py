import numpy as np
import time
from i2sar.geometry.radar_grid import RadarGrid
from i2sar.geometry.geo2rdr import geo2rdr_arrayfire
from i2sar.geometry.accelerated_geometry import AcceleratedGeo2Rdr
from i2sar.core.enums import LookSide

def test_comparison():
    print('=' * 80)
    print('Geo2Rdr Full Comparison Test')
    print('=' * 80)
    
    np.random.seed(42)
    
    radar_grid = RadarGrid(
        length=10000,
        width=5000,
        sensing_start_s=0.0,
        prf_hz=1666.67,
        starting_range_m=800000.0,
        range_pixel_spacing_m=2.33,
    )
    
    sat_position = np.array([7078000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7500.0, 0.0])
    
    test_configs = [
        {'name': 'Small', 'n_points': 100},
        {'name': 'Medium', 'n_points': 1000},
        {'name': 'Large', 'n_points': 10000},
    ]
    
    print(f"{'Test Case':<12} {'Method':<10} {'Time (ms)':<12} {'Points/s':<12}")
    print('-' * 60)
    
    for config in test_configs:
        n_points = config['n_points']
        
        lat = np.random.uniform(-30, 30, n_points)
        lon = np.random.uniform(110, 120, n_points)
        height = np.random.uniform(0, 1000, n_points)
        
        times = []
        for _ in range(3):
            t0 = time.time()
            result = geo2rdr_arrayfire(
                lat=lat,
                lon=lon,
                height=height,
                radar_grid=radar_grid,
                satellite_position=sat_position,
                velocity=velocity,
                doppler=0.0,
                look_side=LookSide.RIGHT,
            )
            t1 = time.time()
            times.append(t1 - t0)
        
        avg_time = np.mean(times)
        points_per_sec = n_points / avg_time
        print(f"{config['name']:<12} {'GPU':<10} {avg_time * 1000:<12.2f} {points_per_sec:<12.0f}")
        
        geo2rdr_numba = AcceleratedGeo2Rdr()
        times = []
        for _ in range(3):
            t0 = time.time()
            result = geo2rdr_numba(
                lat=lat,
                lon=lon,
                height=height,
                radar_grid=radar_grid,
                satellite_position=sat_position,
                velocity=velocity,
                doppler=0.0,
                method='numba',
                look_side=LookSide.RIGHT,
            )
            t1 = time.time()
            times.append(t1 - t0)
        
        avg_time = np.mean(times)
        points_per_sec = n_points / avg_time
        print(f"{config['name']:<12} {'Numba':<10} {avg_time * 1000:<12.2f} {points_per_sec:<12.0f}")
        
        print()

if __name__ == '__main__':
    test_comparison()