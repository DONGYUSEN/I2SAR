from __future__ import annotations

from dataclasses import dataclass
from i2sar.core.enums import StrEnum
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
from scipy import fftpack


class WindowStrategy(StrEnum):
    CENTER = "center"
    GRID = "grid"
    RANDOM = "random"


@dataclass(frozen=True)
class CoarseResult:
    azimuth_offset: float
    range_offset: float
    correlation: float
    window_size: int
    search_range: int


def _cross_correlation_fft(
    ref: np.ndarray,
    sec: np.ndarray,
    search_range: int,
) -> Tuple[float, float, float]:
    ref_amp = np.abs(ref) if np.iscomplexobj(ref) else ref
    sec_amp = np.abs(sec) if np.iscomplexobj(sec) else sec

    ref_amp = ref_amp.astype(np.float64)
    sec_amp = sec_amp.astype(np.float64)

    ref_amp -= ref_amp.mean()
    sec_amp -= sec_amp.mean()

    h, w = ref_amp.shape
    if search_range < 1:
        search_range = min(h, w) // 4

    sec_amp[:search_range, :] = 0
    sec_amp[h - search_range:, :] = 0
    sec_amp[:, :search_range] = 0
    sec_amp[:, w - search_range:] = 0

    ref_fft = fftpack.fft2(ref_amp)
    sec_fft = fftpack.fft2(sec_amp)

    cross_power = ref_fft * np.conj(sec_fft)

    cols = w // 2 + 1
    sign = (1.0 - 2.0 * (np.arange(h * cols) % 2)).reshape(h, cols)
    cross_power[:, :cols] *= sign

    correlation = np.abs(np.real(fftpack.ifft2(cross_power))) / (h * w)

    pad_size = search_range
    corr_padded = np.pad(correlation, pad_size, mode="constant")

    corr_search = corr_padded[
        pad_size : pad_size + h,
        pad_size : pad_size + w,
    ]

    max_idx = np.unravel_index(np.argmax(corr_search), corr_search.shape)
    cmax = float(corr_search[max_idx])

    offset_az = max_idx[0] - h // 2
    offset_rg = max_idx[1] - w // 2

    def time_corr(c1, c2, xoff, yoff):
        ny_corr = h
        nx_corr = w
        y1_start = max(0, yoff)
        y1_end = min(h, h + yoff)
        x1_start = max(0, xoff)
        x1_end = min(w, w + xoff)
        if y1_end <= y1_start or x1_end <= x1_start:
            return 0.0

        y2_start = max(0, -yoff)
        y2_end = y2_start + (y1_end - y1_start)
        x2_start = max(0, -xoff)
        x2_end = x2_start + (x1_end - x1_start)
        a = c1[y1_start:y1_end, x1_start:x1_end]
        b = c2[y2_start:y2_end, x2_start:x2_end]
        if a.size == 0 or b.size == 0:
            return 0.0
        num = float(np.sum(a * b))
        denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        return 100.0 * np.abs(num / denom) if denom > 0 else 0.0

    max_corr = time_corr(ref_amp, sec_amp, int(offset_rg), int(offset_az))

    return offset_az, offset_rg, max_corr


class CoarseRegistration:
    def __init__(
        self,
        window_size: int = 1024,
        search_range: int = 256,
        strategy: WindowStrategy = WindowStrategy.CENTER,
        num_grid_points: int = 9,
        correlation_threshold: float = 20.0,
    ):
        self.window_size = window_size
        self.search_range = search_range
        self.strategy = strategy
        self.num_grid_points = num_grid_points
        self.correlation_threshold = correlation_threshold

    def _get_window_positions(self, h: int, w: int) -> List[Tuple[int, int]]:
        positions = []
        
        if self.strategy == WindowStrategy.CENTER:
            positions.append((h // 2, w // 2))
            
        elif self.strategy == WindowStrategy.GRID:
            step_row = max(1, (h - self.window_size) // int(np.sqrt(self.num_grid_points)))
            step_col = max(1, (w - self.window_size) // int(np.sqrt(self.num_grid_points)))
            for i in range(0, h - self.window_size + 1, step_row):
                for j in range(0, w - self.window_size + 1, step_col):
                    positions.append((i + self.window_size // 2, j + self.window_size // 2))
                    
        elif self.strategy == WindowStrategy.RANDOM:
            np.random.seed(42)
            for _ in range(self.num_grid_points):
                row = np.random.randint(self.window_size // 2, h - self.window_size // 2)
                col = np.random.randint(self.window_size // 2, w - self.window_size // 2)
                positions.append((row, col))
                
        return positions

    def register(
        self,
        master: np.ndarray,
        slave: np.ndarray,
    ) -> CoarseResult:
        h, w = master.shape
        half_win = self.window_size // 2
        half_search = self.search_range // 2

        positions = self._get_window_positions(h, w)
        
        offsets_az = []
        offsets_rg = []
        correlations = []

        for row, col in positions:
            r_start = max(0, row - half_win - half_search)
            r_end = min(h, row + half_win + half_search)
            c_start = max(0, col - half_win - half_search)
            c_end = min(w, col + half_win + half_search)

            master_win = master[r_start:r_end, c_start:c_end]
            slave_win = slave[r_start:r_end, c_start:c_end]

            if master_win.shape[0] < self.window_size or master_win.shape[1] < self.window_size:
                continue

            offset_az, offset_rg, corr = _cross_correlation_fft(
                master_win, slave_win, half_search
            )

            if corr >= self.correlation_threshold:
                offsets_az.append(offset_az)
                offsets_rg.append(offset_rg)
                correlations.append(corr)

        if not offsets_az:
            return CoarseResult(
                azimuth_offset=0.0,
                range_offset=0.0,
                correlation=0.0,
                window_size=self.window_size,
                search_range=self.search_range,
            )

        mean_az = float(np.mean(offsets_az))
        mean_rg = float(np.mean(offsets_rg))
        mean_corr = float(np.mean(correlations))

        return CoarseResult(
            azimuth_offset=mean_az,
            range_offset=mean_rg,
            correlation=mean_corr,
            window_size=self.window_size,
            search_range=self.search_range,
        )


def coarse_register(
    master: np.ndarray,
    slave: np.ndarray,
    window_size: int = 1024,
    search_range: int = 256,
    strategy: str = "center",
    num_grid_points: int = 9,
    correlation_threshold: float = 20.0,
) -> Dict[str, Any]:
    strategy_enum = WindowStrategy(strategy.lower())
    
    registrar = CoarseRegistration(
        window_size=window_size,
        search_range=search_range,
        strategy=strategy_enum,
        num_grid_points=num_grid_points,
        correlation_threshold=correlation_threshold,
    )
    
    result = registrar.register(master, slave)
    
    return {
        "azimuth_offset": result.azimuth_offset,
        "range_offset": result.range_offset,
        "correlation": result.correlation,
        "window_size": result.window_size,
        "search_range": result.search_range,
        "method": "coarse_registration",
    }
