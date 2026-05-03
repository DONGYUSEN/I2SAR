from __future__ import annotations

from dataclasses import dataclass
from i2sar.core.enums import StrEnum
from typing import Dict, Any, Optional, Tuple

import numpy as np


class InterpMethod(StrEnum):
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    NEAREST = "nearest"
    SINC = "sinc"


@dataclass(frozen=True)
class ResampleResult:
    resampled: np.ndarray
    valid_mask: np.ndarray
    method: str
    backend: str = "numpy"


def _bilinear_interp(
    image: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    h, w = image.shape
    
    rows_floor = np.floor(rows).astype(np.int32)
    cols_floor = np.floor(cols).astype(np.int32)
    
    rows_ceil = rows_floor + 1
    cols_ceil = cols_floor + 1
    
    rows_floor = np.clip(rows_floor, 0, h - 1)
    cols_floor = np.clip(cols_floor, 0, w - 1)
    rows_ceil = np.clip(rows_ceil, 0, h - 1)
    cols_ceil = np.clip(cols_ceil, 0, w - 1)
    
    tl = image[rows_floor, cols_floor]
    tr = image[rows_floor, cols_ceil]
    bl = image[rows_ceil, cols_floor]
    br = image[rows_ceil, cols_ceil]
    
    row_frac = rows - rows_floor
    col_frac = cols - cols_floor
    
    t = tl * (1 - col_frac) + tr * col_frac
    b = bl * (1 - col_frac) + br * col_frac
    
    return t * (1 - row_frac) + b * row_frac


def _bilinear_interp_arrayfire(
    image: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    raise NotImplementedError("ArrayFire resampler backend is disabled for ArrayFire 3.8.3 Python stability")


def _bicubic_interp(
    image: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    h, w = image.shape
    
    rows_floor = np.floor(rows).astype(np.int32)
    cols_floor = np.floor(cols).astype(np.int32)
    
    result = np.zeros_like(rows, dtype=image.dtype)
    
    for i in range(-1, 3):
        for j in range(-1, 3):
            r = np.clip(rows_floor + i, 0, h - 1)
            c = np.clip(cols_floor + j, 0, w - 1)
            
            t = rows - (rows_floor + i)
            u = cols - (cols_floor + j)
            
            wt_t = _cubic_kernel(t)
            wt_u = _cubic_kernel(u)
            weight = wt_t * wt_u
            
            result += image[r, c] * weight
    
    return result


def _cubic_kernel(t: np.ndarray) -> np.ndarray:
    abs_t = np.abs(t)
    t2 = t * t
    t3 = t2 * t
    
    mask1 = abs_t <= 1
    mask2 = (abs_t > 1) & (abs_t <= 2)
    
    result = np.zeros_like(t)
    result[mask1] = 1 - 2 * t2[mask1] + t3[mask1]
    result[mask2] = 4 - 8 * abs_t[mask2] + 5 * t2[mask2] - t3[mask2]
    
    return result


def _nearest_interp(
    image: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    h, w = image.shape
    
    rows_round = np.round(rows).astype(np.int32)
    cols_round = np.round(cols).astype(np.int32)
    
    rows_round = np.clip(rows_round, 0, h - 1)
    cols_round = np.clip(cols_round, 0, w - 1)
    
    return image[rows_round, cols_round]


def _sinc_kernel(t: np.ndarray) -> np.ndarray:
    mask = t == 0
    result = np.empty_like(t, dtype=np.float64)
    result[mask] = 1.0
    result[~mask] = np.sin(np.pi * t[~mask]) / (np.pi * t[~mask])
    return result


def _sinc_interp(
    image: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    kernel_size: int = 8,
) -> np.ndarray:
    h, w = image.shape
    
    rows_floor = np.floor(rows).astype(np.int32)
    cols_floor = np.floor(cols).astype(np.int32)
    
    half_size = kernel_size // 2
    
    result = np.zeros_like(rows, dtype=np.float64)
    weight_sum = np.zeros_like(rows, dtype=np.float64)
    
    for i in range(-half_size, half_size):
        for j in range(-half_size, half_size):
            r = np.clip(rows_floor + i, 0, h - 1)
            c = np.clip(cols_floor + j, 0, w - 1)
            
            t = rows - (rows_floor + i)
            u = cols - (cols_floor + j)
            
            wt_t = _sinc_kernel(t)
            wt_u = _sinc_kernel(u)
            weight = wt_t * wt_u
            
            result += image[r, c] * weight
            weight_sum += weight
    
    mask = weight_sum > 1e-10
    result[mask] /= weight_sum[mask]
    
    if np.iscomplexobj(image):
        return result.astype(np.complex128)
    return result


class Resampler:
    def __init__(
        self,
        method: InterpMethod = InterpMethod.BILINEAR,
        fill_value: float = 0.0,
        sinc_kernel_size: int = 8,
        use_numba: bool = True,
        backend: str = "auto",
    ):
        self.method = method
        self.fill_value = fill_value
        self.sinc_kernel_size = sinc_kernel_size
        self.use_numba = use_numba
        self.backend = backend.lower()

    def resample(
        self,
        slave: np.ndarray,
        offsets_az: np.ndarray,
        offsets_rg: np.ndarray,
    ) -> ResampleResult:
        h, w = slave.shape
        
        rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        
        src_rows = rows + offsets_az
        src_cols = cols + offsets_rg
        
        valid_mask = (src_rows >= 0) & (src_rows < h) & (src_cols >= 0) & (src_cols < w)
        
        n_points = h * w
        use_arrayfire = self.backend == "arrayfire"
        selected_backend = "numpy"
        use_numba = self.use_numba and n_points > 100 and not use_arrayfire
        
        if self.method == InterpMethod.BILINEAR:
            if use_arrayfire:
                try:
                    resampled = _bilinear_interp_arrayfire(slave, src_rows, src_cols)
                    selected_backend = "arrayfire"
                except Exception:
                    if self.backend == "arrayfire":
                        raise
                    resampled = _bilinear_interp(slave, src_rows, src_cols)
            elif use_numba:
                try:
                    from .resampler_numba import _bilinear_interp_numba
                    resampled = _bilinear_interp_numba(slave, src_rows.flatten(), src_cols.flatten())
                    resampled = resampled.reshape(h, w)
                except ImportError:
                    resampled = _bilinear_interp(slave, src_rows, src_cols)
            else:
                resampled = _bilinear_interp(slave, src_rows, src_cols)
        elif self.method == InterpMethod.BICUBIC:
            if use_numba:
                try:
                    from .resampler_numba import _bicubic_interp_numba
                    resampled = _bicubic_interp_numba(slave, src_rows.flatten(), src_cols.flatten())
                    resampled = resampled.reshape(h, w)
                except ImportError:
                    resampled = _bicubic_interp(slave, src_rows, src_cols)
            else:
                resampled = _bicubic_interp(slave, src_rows, src_cols)
        elif self.method == InterpMethod.SINC:
            if use_numba:
                try:
                    from .resampler_numba import _sinc_interp_numba
                    resampled = _sinc_interp_numba(slave, src_rows.flatten(), src_cols.flatten(), self.sinc_kernel_size)
                    resampled = resampled.reshape(h, w)
                except ImportError:
                    resampled = _sinc_interp(slave, src_rows, src_cols, self.sinc_kernel_size)
            else:
                resampled = _sinc_interp(slave, src_rows, src_cols, self.sinc_kernel_size)
        else:
            if use_numba:
                try:
                    from .resampler_numba import _nearest_interp_numba
                    resampled = _nearest_interp_numba(slave, src_rows.flatten(), src_cols.flatten())
                    resampled = resampled.reshape(h, w)
                except ImportError:
                    resampled = _nearest_interp(slave, src_rows, src_cols)
            else:
                resampled = _nearest_interp(slave, src_rows, src_cols)
        
        resampled[~valid_mask] = self.fill_value
        
        return ResampleResult(
            resampled=resampled,
            valid_mask=valid_mask,
            method=str(self.method),
            backend=selected_backend,
        )


def resample_slave(
    slave: np.ndarray,
    offsets_az: np.ndarray,
    offsets_rg: np.ndarray,
    method: str = "bilinear",
    fill_value: float = 0.0,
    sinc_kernel_size: int = 8,
) -> Dict[str, Any]:
    method_enum = InterpMethod(method.lower())
    
    resampler = Resampler(
        method=method_enum,
        fill_value=fill_value,
        sinc_kernel_size=sinc_kernel_size,
    )
    
    result = resampler.resample(slave, offsets_az, offsets_rg)
    
    return {
        "resampled": result.resampled,
        "valid_mask": result.valid_mask,
        "method": result.method,
    }
