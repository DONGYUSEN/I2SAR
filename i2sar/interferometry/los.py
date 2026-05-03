from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np


@dataclass(frozen=True)
class LOSResult:
    los_deformation_m: np.ndarray
    los_deformation_mm: np.ndarray
    incidence_angle: np.ndarray
    heading_angle: np.ndarray


def phase_to_los_deformation(
    unwrapped_phase: np.ndarray,
    wavelength: float,
) -> np.ndarray:
    """
    将解缠相位转换为 LOS 方向形变（米）。
    
    disp_los = phase * wavelength / (4 * pi)
    """
    return np.asarray(unwrapped_phase, dtype=np.float64) * float(wavelength) / (4.0 * np.pi)


def los_conversion(
    unwrapped_phase: np.ndarray,
    wavelength: float,
    incidence_angle: Optional[np.ndarray] = None,
    heading_angle: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    执行完整的 LOS 转换流程。
    
    Args:
        unwrapped_phase: 解缠相位（弧度）
        wavelength: 雷达波长（米）
        incidence_angle: 入射角（弧度），可选
        heading_angle: 航向角（弧度），可选
    
    Returns:
        Dict 包含 LOS 形变（米和毫米）以及角度信息
    """
    los_m = phase_to_los_deformation(unwrapped_phase, wavelength)
    los_mm = los_m * 1000.0
    
    h, w = los_m.shape
    
    if incidence_angle is None:
        incidence_angle = np.ones((h, w), dtype=np.float64) * np.pi / 4
    else:
        incidence_angle = np.asarray(incidence_angle, dtype=np.float64)
    
    if heading_angle is None:
        heading_angle = np.zeros((h, w), dtype=np.float64)
    else:
        heading_angle = np.asarray(heading_angle, dtype=np.float64)
    
    return {
        "los_deformation_m": los_m,
        "los_deformation_mm": los_mm,
        "incidence_angle": incidence_angle,
        "heading_angle": heading_angle,
        "wavelength": wavelength,
    }


def los_to_enu(
    los_deformation: np.ndarray,
    incidence_angle: np.ndarray,
    heading_angle: np.ndarray,
    look_side: str = "right",
) -> Dict[str, np.ndarray]:
    """
    将 LOS 形变分解为东-北-天（ENU）坐标系下的分量。
    
    Args:
        los_deformation: LOS 方向形变（米）
        incidence_angle: 入射角（弧度）
        heading_angle: 航向角（弧度）
        look_side: 观测侧，"right" 或 "left"
    
    Returns:
        Dict 包含东、北、天三个方向的形变分量
    """
    los_deformation = np.asarray(los_deformation, dtype=np.float64)
    incidence_angle = np.asarray(incidence_angle, dtype=np.float64)
    heading_angle = np.asarray(heading_angle, dtype=np.float64)
    
    look_factor = 1.0 if look_side.lower() == "right" else -1.0
    
    east = look_factor * los_deformation * np.sin(incidence_angle) * np.sin(heading_angle)
    north = -los_deformation * np.sin(incidence_angle) * np.cos(heading_angle)
    up = -los_deformation * np.cos(incidence_angle)
    
    return {
        "east": east.astype(np.float32),
        "north": north.astype(np.float32),
        "up": up.astype(np.float32),
    }
