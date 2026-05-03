from __future__ import annotations

from dataclasses import dataclass
from i2sar.core.enums import StrEnum
from typing import Dict, Any, Optional

import numpy as np
from scipy.fft import fft2, ifft2
import scipy.fftpack as fftpack
from scipy.ndimage import uniform_filter
from scipy.signal import fftconvolve
from concurrent.futures import ThreadPoolExecutor, as_completed


class FilterType(StrEnum):
    GOLDSTEIN = "goldstein"
    BOXCAR = "boxcar"


@dataclass(frozen=True)
class FilterResult:
    filtered: np.ndarray
    coherence: np.ndarray
    method: str


def _goldstein_filter_single(
    row: int,
    interferogram: np.ndarray,
    coherence: np.ndarray,
    alpha: float,
    window_size: int,
) -> np.ndarray:
    h, w = interferogram.shape
    half_win = window_size // 2
    
    filtered_row = np.zeros(w, dtype=np.complex128)
    
    for col in range(w):
        row_start = max(0, row - half_win)
        row_end = min(h, row + half_win + 1)
        col_start = max(0, col - half_win)
        col_end = min(w, col + half_win + 1)
        
        local_ifg = interferogram[row_start:row_end, col_start:col_end]
        local_coh = coherence[row_start:row_end, col_start:col_end]
        
        avg_coh = np.mean(local_coh)
        
        if avg_coh < 0.01:
            filtered_row[col] = interferogram[row, col]
            continue
        
        weight = avg_coh ** alpha
        
        local_fft = fftpack.fft2(local_ifg)
        local_amp = np.abs(local_fft)
        
        threshold = np.percentile(local_amp, 50)
        local_amp[local_amp < threshold] *= weight
        local_amp[local_amp >= threshold] = local_amp[local_amp >= threshold]
        
        filtered_fft = local_amp * np.exp(1j * np.angle(local_fft))
        filtered_local = fftpack.ifft2(filtered_fft)
        
        center_row = (row_end - row_start) // 2
        center_col = (col_end - col_start) // 2
        
        filtered_row[col] = filtered_local[center_row, center_col]
    
    return filtered_row


class GoldsteinFilter:
    def __init__(
        self,
        alpha: float = 1.0,
        window_size: int = 32,
        num_workers: int = 8,
    ):
        self.alpha = alpha
        self.window_size = window_size
        self.num_workers = num_workers

    def filter(
        self,
        interferogram: np.ndarray,
        coherence: Optional[np.ndarray] = None,
    ) -> FilterResult:
        h, w = interferogram.shape
        
        if coherence is None:
            coherence = np.ones((h, w), dtype=np.float64)
        else:
            coherence = np.asarray(coherence, dtype=np.float64)
            coherence = np.clip(coherence, 0.0, 1.0)
        
        filtered = np.zeros((h, w), dtype=np.complex128)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            for row in range(h):
                future = executor.submit(
                    _goldstein_filter_single,
                    row,
                    interferogram,
                    coherence,
                    self.alpha,
                    self.window_size,
                )
                futures[future] = row
            
            for future in as_completed(futures):
                row = futures[future]
                try:
                    filtered[row] = future.result()
                except Exception:
                    filtered[row] = interferogram[row]
        
        filtered_coherence = coherence.copy()
        
        return FilterResult(
            filtered=filtered,
            coherence=filtered_coherence,
            method="goldstein",
        )


class BoxcarFilter:
    def __init__(
        self,
        window_size: int = 5,
        num_workers: int = 8,
    ):
        self.window_size = window_size
        self.num_workers = num_workers

    def filter(
        self,
        interferogram: np.ndarray,
        coherence: Optional[np.ndarray] = None,
    ) -> FilterResult:
        h, w = interferogram.shape
        
        window = np.ones((self.window_size, self.window_size))
        window = window / window.sum()
        
        real_part = fftpack.fftconvolve(np.real(interferogram), window, mode="same")
        imag_part = fftpack.fftconvolve(np.imag(interferogram), window, mode="same")
        filtered = real_part + 1j * imag_part
        
        if coherence is not None:
            filtered_coherence = fftpack.fftconvolve(coherence, window, mode="same")
            filtered_coherence = np.clip(filtered_coherence, 0.0, 1.0)
        else:
            filtered_coherence = np.ones((h, w), dtype=np.float64)
        
        return FilterResult(
            filtered=filtered,
            coherence=filtered_coherence,
            method="boxcar",
        )


def goldstein_filter_vectorized(
    interferogram: np.ndarray,
    coherence: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    window_size: int = 32,
) -> Dict[str, Any]:
    if interferogram.ndim != 2:
        raise ValueError(
            f"interferogram 必须为二维数组，当前 shape={interferogram.shape}"
        )
    
    h, w = interferogram.shape
    
    if coherence is None:
        coh = np.ones((h, w), dtype=np.float64)
    else:
        coh = np.asarray(coherence, dtype=np.float64)
        coh = np.clip(coh, 0.0, 1.0)
    
    local_coh = uniform_filter(coh, size=window_size, mode="reflect")
    weights = np.clip(local_coh ** alpha, 0.0, 1.0)
    
    padded = np.pad(
        interferogram,
        window_size,
        mode="reflect",
    )
    
    fft2d = fft2(padded)
    fft_mag = np.abs(fft2d)
    fft_angle = np.angle(fft2d)
    
    win = np.ones((window_size, window_size), dtype=np.float64)
    win = win / win.sum()
    
    fft_mag_conv = fftconvolve(fft_mag, win, mode="same")
    fft_mag_conv = fft_mag_conv[window_size:-window_size, window_size:-window_size]
    
    threshold = float(np.percentile(fft_mag_conv[fft_mag_conv > 0], 50))
    threshold = threshold if np.isfinite(threshold) else fft_mag_conv.mean()
    
    weights_pad = np.pad(weights, window_size, mode="reflect")
    
    soft_mag = np.where(
        fft_mag < threshold,
        fft_mag * weights_pad,
        fft_mag,
    )
    
    filtered_fft = soft_mag * np.exp(1j * fft_angle)
    filtered_padded = np.real(ifft2(filtered_fft))
    
    half_pad = window_size
    result = filtered_padded[half_pad:-half_pad, half_pad:-half_pad]
    result = result * np.exp(1j * np.angle(interferogram))
    
    result = np.where(
        local_coh > 0.01,
        result,
        interferogram,
    )
    
    return {
        "filtered": result,
        "coherence": coh,
        "method": "goldstein_vectorized",
        "alpha": alpha,
        "window_size": window_size,
    }


def goldstein_filter(
    interferogram: np.ndarray,
    coherence: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    window_size: int = 32,
    num_workers: int = 8,
) -> Dict[str, Any]:
    filter_obj = GoldsteinFilter(
        alpha=alpha,
        window_size=window_size,
        num_workers=num_workers,
    )
    result = filter_obj.filter(interferogram, coherence)
    
    return {
        "filtered": result.filtered,
        "coherence": result.coherence,
        "method": "goldstein",
        "alpha": alpha,
        "window_size": window_size,
    }


def boxcar_filter(
    interferogram: np.ndarray,
    coherence: Optional[np.ndarray] = None,
    window_size: int = 5,
) -> Dict[str, Any]:
    filter_obj = BoxcarFilter(window_size=window_size)
    result = filter_obj.filter(interferogram, coherence)
    
    return {
        "filtered": result.filtered,
        "coherence": result.coherence,
        "method": "boxcar",
        "window_size": window_size,
    }
