#!/usr/bin/env python3
import os
os.environ['LD_LIBRARY_PATH'] = '/home/ysdong/Software/arrayfire/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

import time
import numpy as np
import arrayfire as af

def test_arrayfire_basic():
    print("Testing ArrayFire GPU...")
    
    # Check backend
    print(f"Available backends: {af.get_available_backends()}")
    print(f"Current backend: {af.get_backend()}")
    
    # Get device info
    dev_info = af.device_info()
    print(f"\nDevice info:")
    for key, value in dev_info.items():
        print(f"  {key}: {value}")
    
    # Test basic GPU operation
    print("\nTesting GPU computation...")
    n = 1000000
    a = af.randu(n, dtype=af.Dtype.f64)
    b = af.randu(n, dtype=af.Dtype.f64)
    
    start = time.time()
    for _ in range(100):
        c = af.sin(a) + af.cos(b)
    af.sync()
    elapsed = time.time() - start
    
    print(f"Time for 100 operations on {n} elements: {elapsed:.4f}s")
    print(f"Estimated operations per second: {100/elapsed:.2f}")

if __name__ == "__main__":
    test_arrayfire_basic()