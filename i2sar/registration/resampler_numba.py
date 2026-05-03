from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def _bilinear_interp_numba(
    image: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """Numba 加速的双线性插值"""
    h, w = image.shape
    n_points = len(rows)
    result = np.zeros(n_points, dtype=image.dtype)
    
    for i in prange(n_points):
        r = rows[i]
        c = cols[i]
        
        r0 = int(np.floor(r))
        c0 = int(np.floor(c))
        r1 = r0 + 1
        c1 = c0 + 1
        
        r0 = max(0, min(r0, h - 1))
        c0 = max(0, min(c0, w - 1))
        r1 = max(0, min(r1, h - 1))
        c1 = max(0, min(c1, w - 1))
        
        dr = r - r0
        dc = c - c0
        
        tl = image[r0, c0]
        tr = image[r0, c1]
        bl = image[r1, c0]
        br = image[r1, c1]
        
        t = tl * (1 - dc) + tr * dc
        b = bl * (1 - dc) + br * dc
        
        result[i] = t * (1 - dr) + b * dr
    
    return result


@njit(fastmath=True)
def _cubic_kernel_numba(t: float) -> float:
    """三次样条核函数"""
    abs_t = abs(t)
    t2 = t * t
    t3 = t2 * t
    
    if abs_t <= 1:
        return 1 - 2 * t2 + t3
    elif abs_t <= 2:
        return 4 - 8 * abs_t + 5 * t2 - t3
    else:
        return 0.0


@njit(parallel=True, fastmath=True)
def _bicubic_interp_numba(
    image: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """Numba 加速的双三次插值"""
    h, w = image.shape
    n_points = len(rows)
    result = np.zeros(n_points, dtype=np.float64)
    
    for i in prange(n_points):
        r = rows[i]
        c = cols[i]
        
        r0 = int(np.floor(r))
        c0 = int(np.floor(c))
        
        sum_val = 0.0
        
        for di in range(-1, 3):
            for dj in range(-1, 3):
                rr = r0 + di
                cc = c0 + dj
                
                rr = max(0, min(rr, h - 1))
                cc = max(0, min(cc, w - 1))
                
                t = r - (r0 + di)
                u = c - (c0 + dj)
                
                wt_t = _cubic_kernel_numba(t)
                wt_u = _cubic_kernel_numba(u)
                weight = wt_t * wt_u
                
                sum_val += image[rr, cc] * weight
        
        result[i] = sum_val
    
    return result


@njit(fastmath=True)
def _sinc_kernel_numba(t: float, alpha: float = 2.0) -> float:
    """Sinc 插值核函数"""
    if t == 0:
        return 1.0
    return np.sinc(t / alpha) * np.sinc(t)


@njit(parallel=True, fastmath=True)
def _sinc_interp_numba(
    image: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    sinc_width: int = 16,
) -> np.ndarray:
    """Numba 加速的 Sinc 插值"""
    h, w = image.shape
    n_points = len(rows)
    result = np.zeros(n_points, dtype=np.float64)
    half_width = sinc_width // 2
    
    for i in prange(n_points):
        r = rows[i]
        c = cols[i]
        
        if r < half_width or r >= h - half_width or c < half_width or c >= w - half_width:
            continue
        
        r0 = int(np.floor(r))
        c0 = int(np.floor(c))
        
        sum_val = 0.0
        sum_weight = 0.0
        
        for di in range(-half_width, half_width):
            for dj in range(-half_width, half_width):
                rr = r0 + di
                cc = c0 + dj
                
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    continue
                
                t = r - rr
                u = c - cc
                
                weight = _sinc_kernel_numba(t) * _sinc_kernel_numba(u)
                
                sum_val += image[rr, cc] * weight
                sum_weight += weight
        
        if sum_weight > 0:
            result[i] = sum_val / sum_weight
    
    return result


@njit(parallel=True, fastmath=True)
def _nearest_interp_numba(
    image: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """Numba 加速的最近邻插值"""
    h, w = image.shape
    n_points = len(rows)
    result = np.zeros(n_points, dtype=image.dtype)
    
    for i in prange(n_points):
        r = rows[i]
        c = cols[i]
        
        if r < 0 or r >= h or c < 0 or c >= w:
            continue
        
        r0 = int(np.round(r))
        c0 = int(np.round(c))
        
        r0 = max(0, min(r0, h - 1))
        c0 = max(0, min(c0, w - 1))
        
        result[i] = image[r0, c0]
    
    return result
