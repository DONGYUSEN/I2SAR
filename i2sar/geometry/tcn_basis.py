from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TCNBasis:
    t: np.ndarray
    c: np.ndarray
    n: np.ndarray

    @staticmethod
    def from_position_velocity(position: np.ndarray, velocity: np.ndarray) -> "TCNBasis":
        pos_norm = np.linalg.norm(position)
        n = -position / pos_norm

        c = np.cross(n, velocity)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            return TCNBasis(t=np.array([1.0, 0.0, 0.0]), c=np.array([0.0, 1.0, 0.0]), n=n)
        c = c / c_norm

        t = np.cross(c, n)
        t = t / np.linalg.norm(t)

        return TCNBasis(t=t, c=c, n=n)

    def project(self, vec: np.ndarray) -> np.ndarray:
        return np.array([
            np.dot(self.t, vec),
            np.dot(self.c, vec),
            np.dot(self.n, vec),
        ])

    def combine(self, weights: np.ndarray) -> np.ndarray:
        t_weight, c_weight, n_weight = weights
        return t_weight * self.t + c_weight * self.c + n_weight * self.n
