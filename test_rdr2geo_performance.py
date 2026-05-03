#!/usr/bin/env python3
import time
import numpy as np
import os

os.environ['LD_LIBRARY_PATH'] = '/home/ysdong/Software/arrayfire/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

from i2sar.geometry.radar_grid import RadarGrid
from i2sar.geometry.accelerated_geometry import rdr2geo_fast

def test_rdr2geo_performance():
    np.random.seed(42)
    
    radar_grid = RadarGrid(
        sensing_start_s=0.0,
        prf_hz=1666.67,
        starting_range_m=830000.0,
        range_pixel_spacing_m=2.33,
        length=1000,
        width=2000
    )
    
    n_points = 100000
    line = np.random.uniform(0, radar_grid.length, n_points)
    pixel = np.random.uniform(0, radar_grid.width, n_points)
    
    satellite_position = np.array([-5142374.0, -4146449.0, 1154545.0], dtype=np.float64)
    velocity = np.array([-3120.0, -6690.0, -2500.0], dtype=np.float64)
    doppler = 1000.0
    dem = np.zeros(n_points, dtype=np.float64)
    
    print(f"Testing rdr2geo performance with {n_points:,} points")
    print(f"Radar grid: {radar_grid.length} x {radar_grid.width}")
    
    for method in ['numba', 'arrayfire']:
        print(f"\n--- Testing {method} method ---")
        start_time = time.time()
        
        try:
            result = rdr2geo_fast(
                line=line,
                pixel=pixel,
                radar_grid=radar_grid,
                satellite_position=satellite_position,
                velocity=velocity,
                doppler=doppler,
                dem=dem,
                method=method
            )
            
            elapsed_time = time.time() - start_time
            points_per_second = n_points / elapsed_time
            
            print(f"Success! Time: {elapsed_time:.2f}s")
            print(f"Throughput: {points_per_second:.0f} points/s")
            print(f"Result shape: {result.shape}")
            print(f"Sample lat: {result[0, 0]:.6f}, lon: {result[1, 0]:.6f}, h: {result[2, 0]:.2f}")
            
        except Exception as e:
            print(f"Failed: {e}")


if __name__ == "__main__":
    test_rdr2geo_performance()