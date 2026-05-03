from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
from scipy import fftpack
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass(frozen=True)
class FinePointResult:
    row: float
    col: float
    azimuth: float
    range: float
    azimuth_local: float
    range_local: float
    correlation: float
    window_size: int


def _adjust_window(window: np.ndarray, target_size: int) -> np.ndarray:
    h, w = window.shape
    if h == target_size and w == target_size:
        return window
    result = np.zeros((target_size, target_size), dtype=window.dtype)
    copy_h = min(h, target_size)
    copy_w = min(w, target_size)
    dst_h = (target_size - copy_h) // 2
    dst_w = (target_size - copy_w) // 2
    result[dst_h:dst_h+copy_h, dst_w:dst_w+copy_w] = window[:copy_h, :copy_w]
    return result


def _cross_correlation(
    master: np.ndarray,
    slave: np.ndarray,
) -> Tuple[float, float, float]:
    ref = np.abs(master) if np.iscomplexobj(master) else master
    sec = np.abs(slave) if np.iscomplexobj(slave) else slave

    ref_amp = ref.astype(np.float64)
    sec_amp = sec.astype(np.float64)

    ref_amp -= ref_amp.mean()
    sec_amp -= sec_amp.mean()

    h, w = ref_amp.shape
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

    corr_search = correlation[search_range:search_range+2*search_range, search_range:search_range+2*search_range]

    if corr_search.size == 0:
        search_range = min(search_range, h // 4)
        corr_search = correlation[search_range:search_range+2*search_range, search_range:search_range+2*search_range]

    max_idx = np.unravel_index(np.argmax(corr_search), corr_search.shape)
    cmax = float(corr_search[max_idx])

    offset_az = max_idx[0] - search_range
    offset_rg = max_idx[1] - search_range

    def time_corr(c1, c2, xoff, yoff):
        ny_corr = 2 * search_range
        nx_corr = 2 * search_range
        y1_start = search_range + max(0, yoff)
        y1_end = search_range + ny_corr + min(0, yoff)
        x1_start = search_range + max(0, xoff)
        x1_end = search_range + nx_corr + min(0, xoff)
        if y1_end <= y1_start or x1_end <= x1_start:
            return 0.0

        y2_start = search_range + max(0, -yoff)
        y2_end = y2_start + (y1_end - y1_start)
        x2_start = search_range + max(0, -xoff)
        x2_end = x2_start + (x1_end - x1_start)
        a = c1[y1_start:y1_end, x1_start:x1_end]
        b = c2[y2_start:y2_end, x2_start:x2_end]
        if a.size == 0 or b.size == 0:
            return 0.0
        num = float(np.sum(a * b))
        denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        return 100.0 * np.abs(num / denom) if denom > 0 else 0.0

    max_corr = time_corr(ref_amp, sec_amp, int(offset_rg), int(offset_az))

    interp_factor = 16
    n2x = 8
    n2y = 8

    if interp_factor > 1:
        if cmax > 0:
            corr_search = corr_search * (max_corr / cmax)

        if offset_az + search_range < n2y/2:
            offset_az = n2y/2 - search_range
        elif offset_az + search_range >= 2*search_range - n2y/2:
            offset_az = 2*search_range - n2y/2 - search_range - 1

        if offset_rg + search_range < n2x/2:
            offset_rg = n2x/2 - search_range
        elif offset_rg + search_range >= 2*search_range - n2x/2:
            offset_rg = 2*search_range - n2x/2 - search_range - 1

        y_start = int(offset_az + search_range - n2y/2)
        y_end = y_start + n2y
        x_start = int(offset_rg + search_range - n2x/2)
        x_end = x_start + n2x

        if 0 <= y_start < corr_search.shape[0] and 0 <= x_start < corr_search.shape[1]:
            corr2 = corr_search[y_start:y_end, x_start:x_end]

            corr2 = np.power(corr2, 0.25)

            if corr2.shape == (8, 8):
                ny_hi = n2y * interp_factor
                nx_hi = n2x * interp_factor

                corr_fft = fftpack.fft2(corr2)
                fft_padded = np.zeros((ny_hi, nx_hi), dtype=np.complex128)
                fft_padded[:n2y//2, :n2x//2] = corr_fft[:n2y//2, :n2x//2]
                fft_padded[:n2y//2, -n2x//2:] = corr_fft[:n2y//2, -n2x//2:]
                fft_padded[-n2y//2:, :n2x//2] = corr_fft[-n2y//2:, :n2x//2]
                fft_padded[-n2y//2:, -n2x//2:] = corr_fft[-n2y//2:, -n2x//2:]

                hi_corr = np.abs(fftpack.ifft2(fft_padded))

                max_idx = np.unravel_index(np.argmax(hi_corr), hi_corr.shape)
                ypeak2 = max_idx[0] - ny_hi // 2
                xpeak2 = max_idx[1] - nx_hi // 2

                offset_az += ypeak2 / interp_factor
                offset_rg += xpeak2 / interp_factor

    return (offset_az, offset_rg, max_corr)


def _register_single_point(
    args: Tuple[int, int, np.ndarray, np.ndarray, int, Tuple[float, float]]
) -> Optional[FinePointResult]:
    row, col, master, slave, window_size, initial_offset = args

    half = window_size // 2
    h, w = master.shape

    r0 = int(row + initial_offset[0])
    c0 = int(col + initial_offset[1])

    r_start = max(0, r0 - half)
    r_end = min(h, r0 + half)
    c_start = max(0, c0 - half)
    c_end = min(w, c0 + half)

    master_window = master[r_start:r_end, c_start:c_end]
    slave_window = slave[r_start:r_end, c_start:c_end]

    if master_window.shape[0] < 10 or master_window.shape[1] < 10:
        return None

    master_win = _adjust_window(master_window, window_size)
    slave_win = _adjust_window(slave_window, window_size)

    offset_az, offset_rg, corr = _cross_correlation(master_win, slave_win)

    offset_az = -offset_az
    offset_rg = -offset_rg

    real_az = offset_az + initial_offset[0]
    real_rg = offset_rg + initial_offset[1]

    return FinePointResult(
        row=float(row),
        col=float(col),
        azimuth=real_az,
        range=real_rg,
        azimuth_local=offset_az,
        range_local=offset_rg,
        correlation=corr,
        window_size=window_size,
    )


class FineRegistration:
    def __init__(
        self,
        window_sizes: Optional[List[int]] = None,
        grid_spacing: Optional[int] = None,
        initial_offset: Optional[Tuple[float, float]] = None,
        num_workers: Optional[int] = None,
        correlation_threshold: float = 30.0,
    ):
        if window_sizes is None:
            window_sizes = [64, 128, 256, 512]
        self.window_sizes = sorted(window_sizes)

        self.grid_spacing = int(grid_spacing) if grid_spacing is not None else 256
        
        if initial_offset is None:
            self.initial_offset = (0.0, 0.0)
        else:
            self.initial_offset = (float(initial_offset[0]), float(initial_offset[1]))
        self.initial_offset = (round(self.initial_offset[0]), round(self.initial_offset[1]))

        if num_workers is None:
            num_workers = 8
        self.num_workers = num_workers
        self.correlation_threshold = float(correlation_threshold)

    def _create_grid(self, h: int, w: int) -> List[Tuple[int, int]]:
        grid_spacing = self.grid_spacing

        points = []
        row = grid_spacing // 2
        while row < h - grid_spacing:
            col = grid_spacing // 2
            while col < w - grid_spacing:
                points.append((row, col))
                col += grid_spacing
            row += grid_spacing

        if len(points) < 900:
            overlap_factor = 0.5
            new_spacing = int(grid_spacing * (1 - overlap_factor))

            points = []
            row = grid_spacing // 2
            while row < h - grid_spacing:
                col = grid_spacing // 2
                while col < w - grid_spacing:
                    points.append((row, col))
                    col += new_spacing
                row += new_spacing

        return points

    def _register_parallel(
        self,
        master: np.ndarray,
        slave: np.ndarray,
        grid_points: List[Tuple[int, int]],
        window_size: int,
    ) -> List[FinePointResult]:
        args_list = [
            (row, col, master, slave, window_size, self.initial_offset)
            for row, col in grid_points
        ]
        offsets = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(_register_single_point, args): args for args in args_list}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        offsets.append(result)
                except Exception:
                    pass
        return offsets

    def register(
        self,
        master: np.ndarray,
        slave: np.ndarray,
    ) -> Dict[str, Any]:
        h, w = master.shape
        grid_points = self._create_grid(h, w)

        best_window_size = min(self.window_sizes, key=lambda v: abs(v - 256))
        offsets = self._register_parallel(master, slave, grid_points, best_window_size)

        filtered_offsets = [
            offset for offset in offsets 
            if offset.correlation >= self.correlation_threshold
        ]

        return {
            "method": "fine_registration",
            "window_size": best_window_size,
            "grid_spacing": self.grid_spacing,
            "initial_offset": self.initial_offset,
            "offsets": filtered_offsets,
            "num_points": len(filtered_offsets),
            "num_workers": self.num_workers,
        }


def fine_register(
    master: np.ndarray,
    slave: np.ndarray,
    window_sizes: Optional[List[int]] = None,
    grid_spacing: Optional[int] = None,
    initial_offset: Tuple[float, float] = (0.0, 0.0),
    num_workers: Optional[int] = None,
    correlation_threshold: float = 30.0,
) -> Dict[str, Any]:
    registrar = FineRegistration(
        window_sizes=window_sizes,
        grid_spacing=grid_spacing,
        initial_offset=initial_offset,
        num_workers=num_workers,
        correlation_threshold=correlation_threshold,
    )
    return registrar.register(master, slave)
