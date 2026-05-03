from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class InterferogramResult:
    interferogram: np.ndarray
    coherence: np.ndarray
    amplitude: np.ndarray


class InterferogramGenerator:
    def __init__(self, use_fft: bool = True):
        self.use_fft = use_fft

    def _to_complex(self, arr: np.ndarray) -> np.ndarray:
        """将结构化数组转换为复数数组"""
        if arr.dtype.names is not None:
            return arr['real'] + 1j * arr['imag']
        return arr
    
    def _compute_interferogram(
        self,
        master: np.ndarray,
        slave: np.ndarray,
    ) -> np.ndarray:
        master = self._to_complex(master)
        slave = self._to_complex(slave)
        
        if self.use_fft:
            conj_slave = np.conj(slave)
            ifg = master * conj_slave
        else:
            ifg = master * np.conj(slave)
        return ifg

    def _compute_coherence(
        self,
        master: np.ndarray,
        slave: np.ndarray,
        window_size: int = 5,
    ) -> np.ndarray:
        master = self._to_complex(master)
        slave = self._to_complex(slave)
        
        h, w = master.shape
        half_win = window_size // 2
        
        master_amp = np.abs(master)
        slave_amp = np.abs(slave)
        
        ones = np.ones((window_size, window_size))
        
        numerator = np.abs(
            fftconvolve(master * np.conj(slave), ones, mode="same")
        )
        
        master_power = fftconvolve(master_amp ** 2, ones, mode="same")
        slave_power = fftconvolve(slave_amp ** 2, ones, mode="same")
        
        denominator = np.sqrt(master_power * slave_power)
        denominator[denominator < 1e-10] = 1e-10
        
        coherence = numerator / denominator
        coherence = np.clip(coherence, 0.0, 1.0)
        
        return coherence

    def generate(
        self,
        master: np.ndarray,
        slave: np.ndarray,
        coherence_window: int = 5,
    ) -> InterferogramResult:
        master_complex = self._to_complex(master)
        slave_complex = self._to_complex(slave)
        
        interferogram = self._compute_interferogram(master_complex, slave_complex)
        coherence = self._compute_coherence(master_complex, slave_complex, coherence_window)
        amplitude = (np.abs(master_complex) + np.abs(slave_complex)) / 2.0
        
        return InterferogramResult(
            interferogram=interferogram,
            coherence=coherence,
            amplitude=amplitude,
        )


def generate_interferogram(
    master: np.ndarray,
    slave: np.ndarray,
    coherence_window: int = 5,
    use_fft: bool = True,
) -> Dict[str, Any]:
    generator = InterferogramGenerator(use_fft=use_fft)
    result = generator.generate(master, slave, coherence_window)
    
    return {
        "interferogram": result.interferogram,
        "coherence": result.coherence,
        "amplitude": result.amplitude,
        "method": "complex_multiplication",
        "coherence_window": coherence_window,
    }
