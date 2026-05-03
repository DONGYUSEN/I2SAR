from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional


def filter_and_remove_phase(
    interferogram: np.ndarray,
    coherence: np.ndarray,
    dem_height: np.ndarray,
    incidence_angle: np.ndarray,
    wavelength: float,
    alpha: float = 1.0,
    window_size: int = 32,
) -> Dict[str, Any]:
    from scipy.ndimage import uniform_filter
    
    h, w = interferogram.shape
    
    if dem_height.shape != (h, w):
        dem_height = np.resize(dem_height, (h, w))
        incidence_angle = np.resize(incidence_angle, (h, w))
    
    k = 4 * np.pi / wavelength
    
    topo_phase = k * dem_height * np.sin(incidence_angle)
    
    filtered_ifg = interferogram * np.exp(-1j * topo_phase)
    
    local_coh = uniform_filter(coherence, size=window_size, mode="reflect")
    weights = np.clip(local_coh ** alpha, 0.0, 1.0)
    
    corrected = filtered_ifg * weights + interferogram * (1 - weights)
    
    return {
        "corrected": corrected,
        "topo_phase": topo_phase,
        "filtered": filtered_ifg,
        "method": "topo_phase_removal",
        "alpha": alpha,
        "window_size": window_size,
    }


def estimate_flat_earth_phase(
    interferogram: np.ndarray,
    coherence: np.ndarray,
    wavelength: float,
    window_size: int = 64,
) -> Dict[str, Any]:
    from scipy.ndimage import uniform_filter
    
    wrapped_phase = np.angle(interferogram)
    
    filtered_phase = uniform_filter(wrapped_phase, size=window_size, mode="reflect")
    
    diff = wrapped_phase - filtered_phase
    diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi
    
    flat_earth_phase = wrapped_phase - diff
    
    residual = interferogram * np.exp(-1j * flat_earth_phase)
    
    return {
        "flat_earth_phase": flat_earth_phase,
        "residual": residual,
        "method": "flat_earth_estimation",
        "window_size": window_size,
    }