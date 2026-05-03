from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional


def arrayfire_available() -> bool:
    """检查 ArrayFire 是否可用"""
    try:
        from i2sar.accel.arrayfire_backend import arrayfire_available as af_available
        return af_available()
    except ImportError:
        return False


def goldstein_filter_gpu(
    interferogram: np.ndarray,
    coherence: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    window_size: int = 32,
) -> Dict[str, Any]:
    """ArrayFire GPU 加速版 Goldstein 滤波"""
    try:
        import arrayfire as af
    except ImportError:
        raise RuntimeError("ArrayFire 未安装")

    h, w = interferogram.shape
    
    if coherence is None:
        coh = np.ones((h, w), dtype=np.float32)
    else:
        coh = np.asarray(coherence, dtype=np.float32)
        coh = np.clip(coh, 0.0, 1.0)
    
    ifg_af = af.Array(interferogram.astype(np.complex64))
    coh_af = af.Array(coh)
    
    uniform_kernel = af.constant(1.0 / (window_size ** 2), window_size, window_size, dtype=af.Dtype.f32)
    
    local_coh_af = af.convolve(coh_af, uniform_kernel, mode=af.CONV_MODE.NORMAL)
    local_coh = af.abs(local_coh_af).to_array()
    
    weights = np.clip(local_coh ** alpha, 0.0, 1.0).astype(np.float32)
    weights_af = af.Array(weights)
    
    padded_ifg = af.pad(ifg_af, pad_s=window_size, pad_n=window_size, pad_w=window_size, pad_e=window_size, mode=af.BORDER_MODE.REFLECT)
    
    fft2d_af = af.fft2(padded_ifg)
    fft_mag_af = af.abs(fft2d_af)
    fft_angle_af = af.arg(fft2d_af)
    
    win = af.constant(1.0 / (window_size ** 2), window_size, window_size, dtype=af.Dtype.f32)
    fft_mag_conv_af = af.convolve(fft_mag_af, win, mode=af.CONV_MODE.NORMAL)
    
    half_pad = window_size
    fft_mag_conv = af.abs(fft_mag_conv_af).to_array()
    fft_mag_conv = fft_mag_conv[half_pad:-half_pad, half_pad:-half_pad] if half_pad > 0 else fft_mag_conv
    
    fft_mag = af.abs(fft2d_af).to_array()
    
    threshold = float(np.percentile(fft_mag_conv[fft_mag_conv > 0], 50))
    if not np.isfinite(threshold):
        threshold = float(np.mean(fft_mag_conv))
    
    threshold_af = af.Array(np.array([[threshold]], dtype=np.float32))
    
    weights_pad_af = af.pad(weights_af, pad_s=window_size, pad_n=window_size, pad_w=window_size, pad_e=window_size, mode=af.BORDER_MODE.REFLECT)
    
    soft_mag_af = af.select(fft_mag < threshold_af, fft_mag * weights_pad_af, fft_mag)
    
    filtered_fft_af = soft_mag_af * af.exp(1j * fft_angle_af)
    filtered_padded_af = af.ifft2(filtered_fft_af)
    filtered_padded = af.abs(filtered_padded_af).to_array()
    
    result = filtered_padded[half_pad:-half_pad, half_pad:-half_pad] * np.exp(1j * np.angle(interferogram))
    
    result = np.where(local_coh > 0.01, result, interferogram)
    
    return {
        "filtered": result,
        "coherence": coh,
        "method": "goldstein_arrayfire",
        "alpha": alpha,
        "window_size": window_size,
    }


def phase_unwrap_gpu(wrapped_phase: np.ndarray) -> np.ndarray:
    """ArrayFire GPU 加速版相位解缠（简单逐行算法）"""
    try:
        import arrayfire as af
    except ImportError:
        raise RuntimeError("ArrayFire 未安装")

    h, w = wrapped_phase.shape
    wrapped_af = af.Array(wrapped_phase.astype(np.float32))
    unwrapped_af = af.copy(wrapped_af)
    
    for i in range(1, h):
        diff = unwrapped_af[i, :] - unwrapped_af[i - 1, :]
        diff = diff - 2 * np.pi * np.round(diff / (2 * np.pi))
        unwrapped_af[i, :] = unwrapped_af[i - 1, :] + diff
    
    for j in range(1, w):
        diff = unwrapped_af[:, j] - unwrapped_af[:, j - 1]
        diff = diff - 2 * np.pi * np.round(diff / (2 * np.pi))
        unwrapped_af[:, j] = unwrapped_af[:, j - 1] + diff
    
    af.eval(unwrapped_af)
    af.sync()
    
    return unwrapped_af.to_array()


def compute_coherence_gpu(master: np.ndarray, slave: np.ndarray, window_size: int = 5) -> np.ndarray:
    """ArrayFire GPU 加速版相干性计算"""
    try:
        import arrayfire as af
    except ImportError:
        raise RuntimeError("ArrayFire 未安装")

    master_af = af.Array(master.astype(np.complex64))
    slave_af = af.Array(slave.astype(np.complex64))
    
    master_mag_sq = af.abs(master_af) ** 2
    slave_mag_sq = af.abs(slave_af) ** 2
    
    kernel = af.constant(1.0 / (window_size ** 2), window_size, window_size, dtype=af.Dtype.f32)
    
    master_sum = af.convolve(master_mag_sq, kernel)
    slave_sum = af.convolve(slave_mag_sq, kernel)
    
    cross = master_af * af.conjg(slave_af)
    cross_sum = af.convolve(af.abs(cross), kernel)
    
    coherence_af = af.abs(cross_sum) / (af.sqrt(master_sum * slave_sum) + 1e-10)
    coherence = coherence_af.to_array()
    
    return np.clip(coherence, 0, 1)


def interpolate_slc_gpu(slave_slc: np.ndarray, az_offset: np.ndarray, rg_offset: np.ndarray, 
                         num_rows: int, num_cols: int) -> np.ndarray:
    """ArrayFire GPU 加速版 SLC 重采样（双线性插值）"""
    try:
        import arrayfire as af
    except ImportError:
        raise RuntimeError("ArrayFire 未安装")

    full_rows = np.arange(num_rows, dtype=np.float32)
    full_cols = np.arange(num_cols, dtype=np.float32)
    grid_rows, grid_cols = np.meshgrid(full_rows, full_cols, indexing='ij')
    
    sample_rows = grid_rows + az_offset
    sample_cols = grid_cols + rg_offset
    
    valid_mask = (sample_rows >= 0) & (sample_rows < slave_slc.shape[0] - 1) & \
                 (sample_cols >= 0) & (sample_cols < slave_slc.shape[1] - 1) & \
                 ~np.isnan(sample_rows) & ~np.isnan(sample_cols)
    
    sample_rows = np.clip(sample_rows, 0, slave_slc.shape[0] - 2)
    sample_cols = np.clip(sample_cols, 0, slave_slc.shape[1] - 2)
    
    sample_rows = np.nan_to_num(sample_rows, nan=0.0)
    sample_cols = np.nan_to_num(sample_cols, nan=0.0)
    
    row_floor = np.floor(sample_rows).astype(np.int64)
    col_floor = np.floor(sample_cols).astype(np.int64)
    
    row_floor = np.clip(row_floor, 0, slave_slc.shape[0] - 2)
    col_floor = np.clip(col_floor, 0, slave_slc.shape[1] - 2)
    
    row_frac = sample_rows - row_floor
    col_frac = sample_cols - col_floor
    
    slave_af = af.Array(slave_slc.astype(np.complex64))
    
    resampled = (1 - row_frac) * (1 - col_frac) * af.lookup(slave_af, row_floor, 0)[:, :, 0].to_array()[row_floor, col_floor] + \
                (1 - row_frac) * col_frac * af.lookup(slave_af, row_floor, 0)[:, :, 0].to_array()[row_floor, col_floor + 1] + \
                row_frac * (1 - col_frac) * af.lookup(slave_af, row_floor + 1, 0)[:, :, 0].to_array()[row_floor, col_floor] + \
                row_frac * col_frac * af.lookup(slave_af, row_floor + 1, 0)[:, :, 0].to_array()[row_floor + 1, col_floor + 1]
    
    return resampled