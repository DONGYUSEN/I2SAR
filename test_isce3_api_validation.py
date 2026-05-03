"""使用 ISCE3 Python API 验证 geo2rdr 实现"""

import numpy as np
import os

try:
    import isce3
    from isce3.core import LookSide, Orbit, Ellipsoid, LUT2d, Poly2d
    from isce3.geometry import geo2rdr as isce3_geo2rdr
    ISCE3_AVAILABLE = True
except ImportError:
    print("ISCE3 Python API 不可用")
    ISCE3_AVAILABLE = False

from i2sar.geometry import RadarGrid, geo2rdr, rdr2geo, llh_to_ecef, LookSide as I2SARLookSide
from i2sar.orbit.interpolate import OrbitInterpolator


def test_with_isce3_api():
    """使用 ISCE3 Python API 验证"""
    if not ISCE3_AVAILABLE:
        print("ISCE3 Python API 不可用，跳过测试")
        return
    
    print('=' * 80)
    print('使用 ISCE3 Python API 验证 geo2rdr')
    print('=' * 80)
    
    orbit_file = '/home/ysdong/Software/isce/isce3/tests/data/orbit.xml'
    
    isce3_orbit = isce3.core.Orbit(orbit_file)
    
    radar_grid = RadarGrid(
        length=1000,
        width=500,
        sensing_start_s=isce3_orbit.start_time,
        prf_hz=1666.67,
        starting_range_m=700000.0,
        range_pixel_spacing_m=2.33,
    )
    
    isce3_ellipsoid = isce3.core.Ellipsoid()
    
    doppler = Poly2d([[125.0]])
    
    look_side = LookSide.Right
    
    test_coords = [
        {'lat': -39.374163, 'lon': 0.104055, 'hgt': 647043.35},
        {'lat': -39.363234, 'lon': 0.096404, 'hgt': 647384.38},
        {'lat': -39.385092, 'lon': 0.111709, 'hgt': 646702.68},
    ]
    
    print(f'\n雷达网格参数:')
    print(f'  sensing_start: {radar_grid.start_time}')
    print(f'  sensing_end: {radar_grid.end_time}')
    print(f'  range_min: {radar_grid.start_range}')
    print(f'  range_max: {radar_grid.end_range}')
    
    for coord in test_coords:
        lat, lon, hgt = coord['lat'], coord['lon'], coord['hgt']
        
        print(f'\n测试坐标: lat={lat:.6f}, lon={lon:.6f}, hgt={hgt:.2f}')
        
        isce3_llh = np.array([lon, lat, hgt])
        
        aztime = 0.0
        slant_range = 0.0
        
        converged = isce3_geo2rdr(
            isce3_llh, isce3_ellipsoid, isce3_orbit, doppler,
            aztime, slant_range, 0.0565642,
            radar_grid.start_range, radar_grid.range_pixel_spacing, radar_grid.width,
            look_side
        )
        
        print(f'ISCE3 geo2rdr: converged={converged}')
        if converged:
            print(f'  aztime={aztime:.10f} s')
            print(f'  slant_range={slant_range:.6f} m')
        
        i2sar_orbit = OrbitInterpolator(
            np.array(isce3_orbit.time),
            np.array([state.position for state in isce3_orbit]),
            np.array([state.velocity for state in isce3_orbit])
        )
        
        i2sar_look_side = I2SARLookSide.RIGHT if look_side == LookSide.Right else I2SARLookSide.LEFT
        
        i2sar_aztime, i2sar_range = geo2rdr(
            lat=lat,
            lon=lon,
            height=hgt,
            radar_grid=radar_grid,
            satellite_position=i2sar_orbit,
            velocity=None,
            doppler=125.0,
            look_side=i2sar_look_side,
        )
        
        print(f'I2SAR geo2rdr:')
        if np.isnan(i2sar_aztime):
            print('  返回 NaN')
        else:
            print(f'  aztime={i2sar_aztime:.10f} s')
            print(f'  slant_range={i2sar_range:.6f} m')
            
            if converged:
                aztime_diff = abs(i2sar_aztime - aztime)
                range_diff = abs(i2sar_range - slant_range)
                print(f'  与 ISCE3 的差异: aztime={aztime_diff*1e9:.2f} ns, range={range_diff*1000:.2f} mm')


if __name__ == '__main__':
    test_with_isce3_api()
