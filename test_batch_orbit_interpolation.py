"""测试 OrbitInterpolator 的批量插值接口"""

import numpy as np
from i2sar.orbit.interpolate import OrbitInterpolator, OrbitInterpMethod


def test_batch_interpolation_consistency():
    """测试批量插值与单点插值的一致性"""
    np.random.seed(42)
    
    n_states = 20
    times = np.linspace(0, 100, n_states)
    positions = np.random.randn(n_states, 3) * 1000 + np.array([7078000.0, 0.0, 0.0])
    velocities = np.random.randn(n_states, 3) + np.array([0.0, 7500.0, 0.0])
    
    methods = [OrbitInterpMethod.LINEAR, OrbitInterpMethod.HERMITE, OrbitInterpMethod.LEGENDRE]
    
    for method in methods:
        print(f'\n测试 {method.value} 插值')
        
        orbit = OrbitInterpolator(times, positions, velocities, method=method)
        
        n_queries = 100
        query_times = np.linspace(times[1], times[-2], n_queries)
        
        positions_batch, velocities_batch = orbit.state_at_many(query_times)
        
        for i, t in enumerate(query_times):
            state = orbit.state_at(t)
            
            pos_diff = np.abs(positions_batch[i] - state.position)
            vel_diff = np.abs(velocities_batch[i] - state.velocity)
            
            if np.any(pos_diff > 1e-10) or np.any(vel_diff > 1e-10):
                print(f'  时间 {t:.2f}: 位置差异 {pos_diff}, 速度差异 {vel_diff}')
                raise AssertionError(f'{method.value} 批量插值与单点插值不一致')
        
        print(f'  所有 {n_queries} 个测试点一致，通过!')


def test_batch_interpolation_performance():
    """测试批量插值的性能"""
    np.random.seed(42)
    
    n_states = 100
    times = np.linspace(0, 1000, n_states)
    positions = np.random.randn(n_states, 3) * 1000 + np.array([7078000.0, 0.0, 0.0])
    velocities = np.random.randn(n_states, 3) + np.array([0.0, 7500.0, 0.0])
    
    orbit = OrbitInterpolator(times, positions, velocities, method=OrbitInterpMethod.HERMITE)
    
    n_queries = 10000
    
    import time
    
    query_times = np.linspace(times[1], times[-2], n_queries)
    
    t0 = time.time()
    for t in query_times:
        orbit.state_at(t)
    t_single = time.time() - t0
    print(f'\n单点调用 {n_queries} 次: {t_single*1000:.2f} ms')
    
    t0 = time.time()
    orbit.state_at_many(query_times)
    t_batch = time.time() - t0
    print(f'批量调用 {n_queries} 次: {t_batch*1000:.2f} ms')
    
    speedup = t_single / t_batch
    print(f'加速比: {speedup:.2f}x')


def test_edge_cases():
    """测试边界情况"""
    np.random.seed(42)
    
    times = np.linspace(0, 100, 10)
    positions = np.random.randn(10, 3) + np.array([7078000.0, 0.0, 0.0])
    velocities = np.random.randn(10, 3) + np.array([0.0, 7500.0, 0.0])
    
    orbit = OrbitInterpolator(times, positions, velocities)
    
    try:
        orbit.state_at_many(np.array([-1.0]))
        print('\n错误：应该拒绝外推')
    except ValueError:
        print('\n正确拒绝外推')
    
    try:
        orbit.state_at_many(np.array([101.0]))
        print('错误：应该拒绝外推')
    except ValueError:
        print('正确拒绝外推')
    
    try:
        orbit.state_at_many(np.array([[1.0, 2.0]]))
        print('错误：应该拒绝二维数组')
    except ValueError:
        print('正确拒绝二维数组')


if __name__ == '__main__':
    print('=' * 80)
    print('OrbitInterpolator 批量插值接口测试')
    print('=' * 80)
    
    test_batch_interpolation_consistency()
    test_batch_interpolation_performance()
    test_edge_cases()
    
    print('\n' + '=' * 80)
    print('所有测试通过!')
    print('=' * 80)
