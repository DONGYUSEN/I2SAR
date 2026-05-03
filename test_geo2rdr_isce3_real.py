"""使用 ISCE3 真实轨道数据测试 geo2rdr"""

import numpy as np
import xml.etree.ElementTree as ET

from i2sar.geometry import RadarGrid, geo2rdr, rdr2geo, LookSide, llh_to_ecef
from i2sar.orbit.interpolate import OrbitInterpolator


def parse_isce3_orbit_xml(xml_path):
    """解析 ISCE3 轨道 XML 文件"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    times = []
    positions = []
    velocities = []
    
    for state_vector in root.findall('.//orbitStateVector'):
        gps_time = float(state_vector.find('gps').text)
        x = float(state_vector.find('x').text)
        y = float(state_vector.find('y').text)
        z = float(state_vector.find('z').text)
        vx = float(state_vector.find('vx').text)
        vy = float(state_vector.find('vy').text)
        vz = float(state_vector.find('vz').text)
        
        times.append(gps_time)
        positions.append([x, y, z])
        velocities.append([vx, vy, vz])
    
    return np.array(times), np.array(positions), np.array(velocities)


def compute_look_side(orbit, radar_grid):
    """计算正确的观察侧"""
    from i2sar.geometry import llh_to_ecef, rdr2geo
    
    mid_line = radar_grid.length / 2.0
    mid_pixel = radar_grid.width / 2.0
    
    lat, lon, hgt = rdr2geo(
        line=mid_line,
        pixel=mid_pixel,
        radar_grid=radar_grid,
        satellite_position=orbit,
        velocity=None,
        doppler=125.0,
        look_side=LookSide.RIGHT,
    )
    
    target_ecef = llh_to_ecef(lat, lon, hgt)
    
    expected_aztime = radar_grid.line_to_azimuth_time(mid_line)
    orbit_state = orbit.state_at(expected_aztime)
    pos = orbit_state.position
    vel = orbit_state.velocity
    
    r = target_ecef - pos
    det = np.dot(np.cross(r, vel), pos)
    
    return LookSide.RIGHT if det > 0 else LookSide.LEFT


def test_geo2rdr_with_isce3_data():
    """使用 ISCE3 真实数据测试 geo2rdr"""
    print('=' * 80)
    print('使用 ISCE3 真实轨道数据测试 geo2rdr')
    print('=' * 80)
    
    orbit_xml_path = '/home/ysdong/Software/isce/isce3/tests/data/orbit.xml'
    print(f'\n加载轨道数据: {orbit_xml_path}')
    
    times, positions, velocities = parse_isce3_orbit_xml(orbit_xml_path)
    print(f'轨道时间范围: {times[0]} 到 {times[-1]}')
    print(f'轨道状态点数: {len(times)}')
    print(f'平均卫星高度: {np.mean(np.linalg.norm(positions, axis=1))/1000:.1f} km')
    
    orbit = OrbitInterpolator(times, positions, velocities)
    
    radar_grid = RadarGrid(
        length=1000,
        width=500,
        sensing_start_s=times[0] + (times[-1] - times[0]) * 0.2,
        prf_hz=1666.67,
        starting_range_m=700000.0,
        range_pixel_spacing_m=2.33,
    )
    
    print(f'\n雷达网格参数:')
    print(f'  时间范围: {radar_grid.start_time} 到 {radar_grid.end_time}')
    print(f'  距离范围: {radar_grid.start_range:.0f} 到 {radar_grid.end_range:.0f} m')
    
    look_side = compute_look_side(orbit, radar_grid)
    print(f'计算得到的观察侧: {look_side}')
    
    print('\n测试往返一致性 (rdr2geo -> geo2rdr):')
    
    test_points = [
        {'line': 100, 'pixel': 100},
        {'line': 500, 'pixel': 250},
        {'line': 900, 'pixel': 400},
    ]
    
    for case in test_points:
        line = case['line']
        pixel = case['pixel']
        
        expected_aztime = radar_grid.line_to_azimuth_time(line)
        expected_range = radar_grid.pixel_to_slant_range(pixel)
        
        print(f'\n测试点: line={line}, pixel={pixel}')
        print(f'期望: aztime={expected_aztime:.6f} s, range={expected_range:.2f} m')
        
        try:
            lat, lon, hgt = rdr2geo(
                line=line,
                pixel=pixel,
                radar_grid=radar_grid,
                satellite_position=orbit,
                velocity=None,
                doppler=125.0,
                look_side=look_side,
            )
            print(f'rdr2geo: lat={lat:.6f}, lon={lon:.6f}, hgt={hgt:.2f}')
            
            aztime, slant_range = geo2rdr(
                lat=lat,
                lon=lon,
                height=hgt,
                radar_grid=radar_grid,
                satellite_position=orbit,
                velocity=None,
                doppler=125.0,
                look_side=look_side,
            )
            
            if np.isnan(aztime):
                print('geo2rdr: 返回 NaN')
                
                target_ecef = llh_to_ecef(lat, lon, hgt)
                print('\n调试信息:')
                print(f'目标点 ECEF: {target_ecef}')
                
                mid_time = radar_grid.start_time + radar_grid.number_of_seconds / 2.0
                orbit_state = orbit.state_at(mid_time)
                pos = orbit_state.position
                vel = orbit_state.velocity
                r = target_ecef - pos
                
                det = np.dot(np.cross(r, vel), pos)
                print(f'在中间时刻 {mid_time}:')
                print(f'  卫星位置: {pos}')
                print(f'  卫星速度: {vel}')
                print(f'  r = target - pos: {r}')
                print(f'  det = dot(cross(r, vel), pos) = {det}')
                print(f'  观察侧判定: {"RIGHT" if det > 0 else "LEFT"}')
                
            else:
                aztime_error = aztime - expected_aztime
                range_error = slant_range - expected_range
                print(f'geo2rdr: aztime={aztime:.6f} s, range={slant_range:.2f} m')
                print(f'误差: aztime={aztime_error*1e9:.2f} ns, range={range_error*1000:.2f} mm')
                
                if abs(aztime_error) < 1e-6 and abs(range_error) < 1e-3:
                    print('测试通过!')
                else:
                    print('警告: 误差超出阈值')
                    
        except Exception as e:
            print(f'错误: {e}')


def test_direct_geo2rdr():
    """直接测试 geo2rdr 对已知点"""
    print('\n' + '=' * 80)
    print('直接测试 geo2rdr (已知地理坐标)')
    print('=' * 80)
    
    orbit_xml_path = '/home/ysdong/Software/isce/isce3/tests/data/orbit.xml'
    times, positions, velocities = parse_isce3_orbit_xml(orbit_xml_path)
    orbit = OrbitInterpolator(times, positions, velocities)
    
    radar_grid = RadarGrid(
        length=1000,
        width=500,
        sensing_start_s=times[0] + (times[-1] - times[0]) * 0.2,
        prf_hz=1666.67,
        starting_range_m=700000.0,
        range_pixel_spacing_m=2.33,
    )
    
    look_side = compute_look_side(orbit, radar_grid)
    print(f'使用观察侧: {look_side}')
    
    test_coords = [
        {'lat': -39.0, 'lon': 0.0, 'hgt': 0.0},
        {'lat': -39.5, 'lon': 0.1, 'hgt': 500.0},
        {'lat': -38.5, 'lon': -0.1, 'hgt': 1000.0},
    ]
    
    for coord in test_coords:
        lat, lon, hgt = coord['lat'], coord['lon'], coord['hgt']
        
        print(f'\n测试坐标: lat={lat:.4f}, lon={lon:.4f}, hgt={hgt:.0f}')
        
        aztime, slant_range = geo2rdr(
            lat=lat,
            lon=lon,
            height=hgt,
            radar_grid=radar_grid,
            satellite_position=orbit,
            velocity=None,
            doppler=125.0,
            look_side=look_side,
        )
        
        if np.isnan(aztime):
            print('geo2rdr: 返回 NaN')
        else:
            print(f'geo2rdr: aztime={aztime:.6f} s, range={slant_range:.2f} m')
            
            orbit_state = orbit.state_at(aztime)
            pos = orbit_state.position
            vel = orbit_state.velocity
            target_ecef = llh_to_ecef(lat, lon, hgt)
            r = target_ecef - pos
            
            det = np.dot(np.cross(r, vel), pos)
            print(f'验证观察侧: det={det:.2e}, 应该{"" if (det > 0) == (look_side == LookSide.RIGHT) else "不"}匹配')


if __name__ == '__main__':
    test_geo2rdr_with_isce3_data()
    test_direct_geo2rdr()
    
    print('\n' + '=' * 80)
    print('测试完成!')
    print('=' * 80)
