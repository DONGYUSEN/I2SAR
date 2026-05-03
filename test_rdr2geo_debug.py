#!/usr/bin/env python3
import time
import numpy as np
import os

os.environ['LD_LIBRARY_PATH'] = '/home/ysdong/Software/arrayfire/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

from i2sar.geometry.radar_grid import RadarGrid
from i2sar.geometry.rdr2geo import rdr2geo
from i2sar.geometry.geometry_arrayfire import rdr2geo_arrayfire_optimized

def test_single_point():
    np.random.seed(42)
    
    radar_grid = RadarGrid(
        sensing_start_s=0.0,
        prf_hz=1666.67,
        starting_range_m=830000.0,
        range_pixel_spacing_m=2.33,
        length=1000,
        width=2000
    )
    
    line = np.array([500.0])
    pixel = np.array([1000.0])
    
    satellite_position = np.array([-5142374.0, -4146449.0, 1154545.0], dtype=np.float64)
    velocity = np.array([-3120.0, -6690.0, -2500.0], dtype=np.float64)
    doppler = 1000.0
    dem = np.array([0.0])
    
    print(f"Testing single point: line={line[0]}, pixel={pixel[0]}")
    
    print("\n--- Original implementation ---")
    start_time = time.time()
    result_original = rdr2geo(
        line=line,
        pixel=pixel,
        radar_grid=radar_grid,
        satellite_position=satellite_position,
        velocity=velocity,
        doppler=doppler,
        dem=dem
    )
    elapsed_time = time.time() - start_time
    print(f"Time: {elapsed_time:.4f}s")
    print(f"Result: lat={result_original[0]:.6f}, lon={result_original[1]:.6f}, h={result_original[2]:.2f}")
    
    print("\n--- ArrayFire implementation ---")
    start_time = time.time()
    try:
        result_af = rdr2geo_arrayfire_optimized(
            line=line,
            pixel=pixel,
            radar_grid=radar_grid,
            satellite_position=satellite_position,
            velocity=velocity,
            doppler=doppler,
            dem=dem
        )
        elapsed_time = time.time() - start_time
        print(f"Time: {elapsed_time:.4f}s")
        print(f"Result: lat={result_af[0]:.6f}, lon={result_af[1]:.6f}, h={result_af[2]:.2f}")
        
        print("\n--- Comparison ---")
        print(f"Lat diff: {abs(result_af[0] - result_original[0]):.6f}")
        print(f"Lon diff: {abs(result_af[1] - result_original[1]):.6f}")
        print(f"H diff: {abs(result_af[2] - result_original[2]):.2f}")
        
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_point()