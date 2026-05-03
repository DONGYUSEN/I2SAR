import numpy as np
from typing import Union, Optional
from .interpolate import OrbitInterpolator


class OrbitData:
    """统一的轨道数据容器，支持预采样和多种插值算法"""
    
    def __init__(self, times: np.ndarray, positions: np.ndarray, velocities: np.ndarray):
        """
        初始化轨道数据
        
        Args:
            times: 采样时间点数组 (N,)
            positions: 对应位置数组 (N, 3)
            velocities: 对应速度数组 (N, 3)
        """
        self.times = np.asarray(times, dtype=np.float64)
        self.positions = np.asarray(positions, dtype=np.float64)
        self.velocities = np.asarray(velocities, dtype=np.float64)
        
        if len(self.times) != len(self.positions) or len(self.times) != len(self.velocities):
            raise ValueError("times, positions, and velocities must have the same length")
        
        self._sorted_indices = np.argsort(self.times)
        self.times = self.times[self._sorted_indices]
        self.positions = self.positions[self._sorted_indices]
        self.velocities = self.velocities[self._sorted_indices]
        
        self.t_min = self.times[0]
        self.t_max = self.times[-1]
    
    @classmethod
    def from_orbit_interpolator(cls, orbit: OrbitInterpolator, num_samples: int = 2000):
        """从 OrbitInterpolator 创建预采样轨道数据"""
        times = np.linspace(
            orbit.reference_epoch,
            orbit.reference_epoch + orbit.number_of_seconds,
            num_samples
        )
        positions = np.zeros((num_samples, 3), dtype=np.float64)
        velocities = np.zeros((num_samples, 3), dtype=np.float64)
        
        for i, t in enumerate(times):
            state = orbit.state_at(t)
            positions[i] = state.position
            velocities[i] = state.velocity
        
        return cls(times, positions, velocities)
    
    @classmethod
    def from_static_orbit(cls, position: np.ndarray, velocity: np.ndarray):
        """从静态轨道创建（时间不变）"""
        times = np.array([0.0, 1.0])
        positions = np.array([position, position])
        velocities = np.array([velocity, velocity])
        return cls(times, positions, velocities)
    
    def _find_indices(self, t: np.ndarray) -> tuple:
        """找到插值所需的索引"""
        indices = np.searchsorted(self.times, t)
        indices = np.clip(indices, 1, len(self.times) - 1)
        return indices - 1, indices
    
    def interpolate_linear(self, t: np.ndarray) -> tuple:
        """
        线性插值
        
        Args:
            t: 要插值的时间点 (...,)
        
        Returns:
            positions: 插值后的位置 (..., 3)
            velocities: 插值后的速度 (..., 3)
        """
        original_shape = t.shape
        t_flat = np.asarray(t, dtype=np.float64).ravel()
        
        idx0, idx1 = self._find_indices(t_flat)
        t0 = self.times[idx0]
        t1 = self.times[idx1]
        
        alpha = (t_flat - t0) / (t1 - t0 + 1e-12)
        alpha = np.clip(alpha, 0.0, 1.0)
        
        pos0 = self.positions[idx0]
        pos1 = self.positions[idx1]
        vel0 = self.velocities[idx0]
        vel1 = self.velocities[idx1]
        
        positions = (1 - alpha[:, np.newaxis]) * pos0 + alpha[:, np.newaxis] * pos1
        velocities = (1 - alpha[:, np.newaxis]) * vel0 + alpha[:, np.newaxis] * vel1
        
        return positions.reshape(original_shape + (3,)), velocities.reshape(original_shape + (3,))
    
    def interpolate_hermite(self, t: np.ndarray) -> tuple:
        """
        Hermite 插值（带导数的三次插值）
        
        Args:
            t: 要插值的时间点 (...,)
        
        Returns:
            positions: 插值后的位置 (..., 3)
            velocities: 插值后的速度 (..., 3)
        """
        original_shape = t.shape
        t_flat = np.asarray(t, dtype=np.float64).ravel()
        
        idx0, idx1 = self._find_indices(t_flat)
        t0 = self.times[idx0]
        t1 = self.times[idx1]
        dt = t1 - t0 + 1e-12
        
        s = (t_flat - t0) / dt
        s2 = s * s
        s3 = s2 * s
        
        h00 = 2 * s3 - 3 * s2 + 1
        h10 = s3 - 2 * s2 + s
        h01 = -2 * s3 + 3 * s2
        h11 = s3 - s2
        
        pos0 = self.positions[idx0]
        pos1 = self.positions[idx1]
        vel0 = self.velocities[idx0]
        vel1 = self.velocities[idx1]
        
        positions = h00[:, np.newaxis] * pos0 + \
                   h10[:, np.newaxis] * vel0 * dt[:, np.newaxis] + \
                   h01[:, np.newaxis] * pos1 + \
                   h11[:, np.newaxis] * vel1 * dt[:, np.newaxis]
        
        # 速度是位置的导数
        h00_deriv = 6 * s2 - 6 * s
        h10_deriv = 3 * s2 - 4 * s + 1
        h01_deriv = -6 * s2 + 6 * s
        h11_deriv = 3 * s2 - 2 * s
        
        velocities = (h00_deriv[:, np.newaxis] * pos0 + \
                      h10_deriv[:, np.newaxis] * vel0 * dt[:, np.newaxis] + \
                      h01_deriv[:, np.newaxis] * pos1 + \
                      h11_deriv[:, np.newaxis] * vel1 * dt[:, np.newaxis]) / dt[:, np.newaxis]
        
        return positions.reshape(original_shape + (3,)), velocities.reshape(original_shape + (3,))
    
    def interpolate_lagrange(self, t: np.ndarray, order: int = 5) -> tuple:
        """
        Lagrange 多项式插值
        
        Args:
            t: 要插值的时间点 (...,)
            order: 插值阶数，默认5阶
        
        Returns:
            positions: 插值后的位置 (..., 3)
            velocities: 插值后的速度 (..., 3)
        """
        original_shape = t.shape
        t_flat = np.asarray(t, dtype=np.float64).ravel()
        n_points = len(t_flat)
        
        positions = np.zeros((n_points, 3), dtype=np.float64)
        velocities = np.zeros((n_points, 3), dtype=np.float64)
        
        half_order = order // 2
        
        for i, t_val in enumerate(t_flat):
            idx = np.searchsorted(self.times, t_val)
            idx = max(half_order, min(idx, len(self.times) - half_order - 1))
            
            start_idx = idx - half_order
            end_idx = idx + half_order + 1
            start_idx = max(0, start_idx)
            end_idx = min(len(self.times), end_idx)
            
            local_times = self.times[start_idx:end_idx]
            local_positions = self.positions[start_idx:end_idx]
            local_velocities = self.velocities[start_idx:end_idx]
            
            n_local = len(local_times)
            
            pos = np.zeros(3)
            vel = np.zeros(3)
            
            for j in range(n_local):
                L = 1.0
                dL = 0.0
                
                for k in range(n_local):
                    if k != j:
                        denom = local_times[j] - local_times[k]
                        L *= (t_val - local_times[k]) / denom
                        
                        dL_term = 1.0 / denom
                        for m in range(n_local):
                            if m != j and m != k:
                                dL_term *= (t_val - local_times[m]) / (local_times[j] - local_times[m])
                        dL += dL_term
                
                pos += L * local_positions[j]
                vel += L * local_velocities[j] + dL * local_positions[j]
            
            positions[i] = pos
            velocities[i] = vel
        
        return positions.reshape(original_shape + (3,)), velocities.reshape(original_shape + (3,))
    
    def interpolate(self, t: np.ndarray, method: str = "hermite", **kwargs) -> tuple:
        """
        统一的插值接口
        
        Args:
            t: 要插值的时间点
            method: 插值方法 ('linear', 'hermite', 'lagrange')
            **kwargs: 额外参数（如 lagrange 的 order）
        
        Returns:
            positions: 插值后的位置
            velocities: 插值后的速度
        """
        method = method.lower()
        if method == "linear":
            return self.interpolate_linear(t)
        elif method == "hermite":
            return self.interpolate_hermite(t)
        elif method == "lagrange":
            order = kwargs.get("order", 5)
            return self.interpolate_lagrange(t, order=order)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
    
    def __len__(self):
        return len(self.times)
    
    def __repr__(self):
        return f"OrbitData(n_samples={len(self)}, t_range=[{self.t_min:.2f}, {self.t_max:.2f}])"