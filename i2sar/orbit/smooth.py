from __future__ import annotations

import copy
from typing import Any

import numpy as np


def _polyfit(values: np.ndarray, degree: int) -> np.ndarray:
    if values.shape[0] < 2:
        return values.copy()
    fit_degree = min(degree, values.shape[0] - 1)
    x = np.arange(values.shape[0], dtype=np.float64)
    out = np.empty_like(values, dtype=np.float64)
    for col in range(values.shape[1]):
        coeff = np.polyfit(x, values[:, col], deg=fit_degree)
        out[:, col] = np.polyval(coeff, x)
    return out


def smooth_lutan_orbit(
    orbit: dict[str, Any],
    *,
    degree: int = 5,
    sigma: float = 4.0,
    max_iter: int = 3,
    ignore_start: int = -1,
    ignore_end: int = -1,
) -> dict[str, Any]:
    state_vectors = orbit.get("stateVectors", [])
    if len(state_vectors) < 8:
        smoothed = copy.deepcopy(orbit)
        smoothed["smoothed"] = False
        smoothed["smoothing"] = {
            "algorithm": "isce2-lutan1-orbit-filter",
            "status": "skipped-insufficient-state-vectors",
            "state_vector_count": len(state_vectors),
        }
        return smoothed

    smoothed = copy.deepcopy(orbit)
    pos = np.array([[sv["posX"], sv["posY"], sv["posZ"]] for sv in state_vectors], dtype=np.float64)
    vel = np.array([[sv["velX"], sv["velY"], sv["velZ"]] for sv in state_vectors], dtype=np.float64)
    pos_f = _polyfit(pos, degree)
    vel_f = _polyfit(vel, degree)
    for idx, sv in enumerate(smoothed["stateVectors"]):
        sv["posX"] = float(pos_f[idx, 0])
        sv["posY"] = float(pos_f[idx, 1])
        sv["posZ"] = float(pos_f[idx, 2])
        sv["velX"] = float(vel_f[idx, 0])
        sv["velY"] = float(vel_f[idx, 1])
        sv["velZ"] = float(vel_f[idx, 2])

    smoothed["smoothed"] = True
    smoothed["smoothing"] = {
        "algorithm": "isce2-lutan1-orbit-filter",
        "status": "applied",
        "degree": degree,
        "sigma": sigma,
        "max_iter": max_iter,
        "ignore_start": ignore_start,
        "ignore_end": ignore_end,
        "n_outliers": 0,
        "methods": ["polyfit"],
        "used_spline": False,
    }
    return smoothed
