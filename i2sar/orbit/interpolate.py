from __future__ import annotations

from dataclasses import dataclass
from i2sar.core.enums import StrEnum
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class OrbitState:
    position: np.ndarray
    velocity: np.ndarray


class OrbitInterpMethod(StrEnum):
    LINEAR = "linear"
    HERMITE = "hermite"
    LEGENDRE = "legendre"


class OrbitInterpolator:
    def __init__(
        self,
        time,
        position,
        velocity,
        method: OrbitInterpMethod = OrbitInterpMethod.LINEAR,
    ):
        time_arr = np.asarray(time, dtype=np.float64)
        position_arr = np.asarray(position, dtype=np.float64)
        velocity_arr = np.asarray(velocity, dtype=np.float64)

        if time_arr.ndim != 1:
            raise ValueError("orbit time must be one-dimensional")
        if position_arr.shape != (time_arr.size, 3):
            raise ValueError("orbit position must have shape (n, 3)")
        if velocity_arr.shape != (time_arr.size, 3):
            raise ValueError("orbit velocity must have shape (n, 3)")
        if time_arr.size < 2:
            raise ValueError("orbit interpolation requires at least two state vectors")

        order = np.argsort(time_arr)
        self.time = time_arr[order]
        self.position = position_arr[order]
        self.velocity = velocity_arr[order]
        self.method = method
        if np.any(np.diff(self.time) <= 0.0):
            raise ValueError("orbit time values must be unique")

        if self.method == OrbitInterpMethod.HERMITE:
            self._precompute_hermite_coeffs()
        elif self.method == OrbitInterpMethod.LEGENDRE:
            self._precompute_legendre_coeffs()

    @property
    def reference_epoch(self) -> float:
        return self.time[0]

    @property
    def number_of_seconds(self) -> float:
        return self.time[-1] - self.time[0]

    @property
    def start_time(self) -> float:
        return self.time[0]

    @property
    def end_time(self) -> float:
        return self.time[-1]

    def _precompute_hermite_coeffs(self):
        n = len(self.time)
        self.hermite_c0 = self.position.copy()
        self.hermite_c1 = self.velocity.copy()
        
        self.hermite_c2 = np.zeros_like(self.position)
        self.hermite_c3 = np.zeros_like(self.position)
        
        h = np.diff(self.time)
        delta_p = np.diff(self.position, axis=0)
        delta_v = np.diff(self.velocity, axis=0)
        
        for i in range(n - 1):
            h_i = h[i]
            t0 = self.time[i]
            t1 = self.time[i + 1]
            
            p0, p1 = self.position[i], self.position[i + 1]
            v0, v1 = self.velocity[i], self.velocity[i + 1]
            
            self.hermite_c2[i] = (3 * (p1 - p0) / h_i**2 - (v1 + 2 * v0) / h_i)
            self.hermite_c3[i] = (-2 * (p1 - p0) / h_i**3 + (v1 + v0) / h_i**2)

    def _precompute_legendre_coeffs(self):
        self.legendre_order = min(5, len(self.time) - 1)

    def _hermite_interpolate(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        indices = np.searchsorted(self.time, query, side='right') - 1
        indices = np.clip(indices, 0, len(self.time) - 2)
        
        t = query
        t0 = self.time[indices]
        h = self.time[indices + 1] - t0
        u = (t - t0) / h
        
        u2 = u * u
        u3 = u2 * u
        
        idx = indices if query.ndim == 0 else indices[:, np.newaxis]
        
        pos = (
            self.hermite_c0[idx]
            + self.hermite_c1[idx] * (t - t0)
            + self.hermite_c2[idx] * (t - t0)**2
            + self.hermite_c3[idx] * (t - t0)**3
        )
        
        vel = (
            self.hermite_c1[idx]
            + 2 * self.hermite_c2[idx] * (t - t0)
            + 3 * self.hermite_c3[idx] * (t - t0)**2
        )
        
        if query.ndim == 0:
            pos = pos.reshape(3)
            vel = vel.reshape(3)
        
        return pos, vel

    def _legendre_interpolate(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.time)
        order = min(self.legendre_order, n - 1)
        
        indices = np.searchsorted(self.time, query, side='right') - 1
        indices = np.clip(indices, order // 2, n - 1 - order // 2)
        
        t = query
        result_pos = np.zeros(query.shape + (3,), dtype=np.float64)
        result_vel = np.zeros(query.shape + (3,), dtype=np.float64)
        
        if query.ndim == 0:
            idx = int(indices)
            start_idx = max(0, idx - order // 2)
            end_idx = min(n, start_idx + order + 1)
            
            local_times = self.time[start_idx:end_idx]
            local_pos = self.position[start_idx:end_idx]
            local_vel = self.velocity[start_idx:end_idx]
            
            m = len(local_times)
            for i in range(m):
                legendre_val = 1.0
                legendre_deriv = 0.0
                for j in range(m):
                    if j != i:
                        legendre_val *= (t - local_times[j]) / (local_times[i] - local_times[j])
                        term = 1.0 / (local_times[i] - local_times[j])
                        for k in range(m):
                            if k != i and k != j:
                                term *= (t - local_times[k]) / (local_times[i] - local_times[k])
                        legendre_deriv += term
                
                result_pos += legendre_val * local_pos[i]
                result_vel += legendre_deriv * local_pos[i]
        else:
            for i in range(len(query)):
                idx = int(indices[i])
                start_idx = max(0, idx - order // 2)
                end_idx = min(n, start_idx + order + 1)
                
                local_times = self.time[start_idx:end_idx]
                local_pos = self.position[start_idx:end_idx]
                
                m = len(local_times)
                pos_i = np.zeros(3)
                vel_i = np.zeros(3)
                
                for j in range(m):
                    legendre_val = 1.0
                    legendre_deriv = 0.0
                    for k in range(m):
                        if k != j:
                            legendre_val *= (query[i] - local_times[k]) / (local_times[j] - local_times[k])
                            term = 1.0 / (local_times[j] - local_times[k])
                            for l in range(m):
                                if l != j and l != k:
                                    term *= (query[i] - local_times[l]) / (local_times[j] - local_times[l])
                            legendre_deriv += term
                    
                    pos_i += legendre_val * local_pos[j]
                    vel_i += legendre_deriv * local_pos[j]
                
                result_pos[i] = pos_i
                result_vel[i] = vel_i
        
        return result_pos, result_vel

    def state_at(self, time, *, allow_extrapolation: bool = False) -> OrbitState:
        query = np.asarray(time, dtype=np.float64)
        if not allow_extrapolation:
            outside = (query < self.time[0]) | (query > self.time[-1])
            if bool(np.any(outside)):
                raise ValueError("query time is outside orbit time span")

        if self.method == OrbitInterpMethod.HERMITE:
            position, velocity = self._hermite_interpolate(query)
        elif self.method == OrbitInterpMethod.LEGENDRE:
            position, velocity = self._legendre_interpolate(query)
        else:
            position = _interp_vectors(query, self.time, self.position)
            velocity = _interp_vectors(query, self.time, self.velocity)
        
        return OrbitState(position=position, velocity=velocity)


def _interp_vectors(query: np.ndarray, time: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.empty(query.shape + (3,), dtype=np.float64)
    for idx in range(3):
        out[..., idx] = np.interp(query, time, values[:, idx])
    return out.reshape((3,)) if query.ndim == 0 else out


def read_orbit_interpolator(
    scene_h5_path: str | Path,
    group: str = "orbit",
    method: OrbitInterpMethod = OrbitInterpMethod.LINEAR,
) -> OrbitInterpolator:
    with h5py.File(scene_h5_path, "r") as h5:
        orbit = h5[group]
        return OrbitInterpolator(
            time=orbit["time"][...],
            position=orbit["position"][...],
            velocity=orbit["velocity"][...],
            method=method,
        )
