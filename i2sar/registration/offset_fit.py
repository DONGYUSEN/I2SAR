from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np


@dataclass(frozen=True)
class OffsetFitResult:
    polynomial_order: int
    parameters: Dict[str, float]
    normalization: Dict[str, float]
    azimuth_rms: float
    range_rms: float
    final_points: int
    outliers_removed: int


def _build_design_matrix(
    rows_norm: np.ndarray,
    cols_norm: np.ndarray,
    polynomial_order: int,
) -> np.ndarray:
    if polynomial_order <= 1:
        return np.column_stack([
            np.ones_like(rows_norm),
            rows_norm,
            cols_norm,
        ])
    return np.column_stack([
        np.ones_like(rows_norm),
        rows_norm,
        cols_norm,
        rows_norm * cols_norm,
        rows_norm * rows_norm,
        cols_norm * cols_norm,
    ])


def _fit_once(
    rows: np.ndarray,
    cols: np.ndarray,
    az: np.ndarray,
    rg: np.ndarray,
    polynomial_order: int,
) -> Dict[str, Any]:
    rows_mean = float(np.mean(rows))
    cols_mean = float(np.mean(cols))
    rows_std = float(np.std(rows))
    cols_std = float(np.std(cols))
    rows_std = rows_std if rows_std > 1e-8 else 1.0
    cols_std = cols_std if cols_std > 1e-8 else 1.0

    rows_norm = (rows - rows_mean) / rows_std
    cols_norm = (cols - cols_mean) / cols_std
    A = _build_design_matrix(rows_norm, cols_norm, polynomial_order)

    coeff_az, _, _, _ = np.linalg.lstsq(A, az, rcond=None)
    coeff_rg, _, _, _ = np.linalg.lstsq(A, rg, rcond=None)

    pred_az = A @ coeff_az
    pred_rg = A @ coeff_rg
    res_az = az - pred_az
    res_rg = rg - pred_rg

    return {
        "coeff_az": coeff_az,
        "coeff_rg": coeff_rg,
        "rows_mean": rows_mean,
        "rows_std": rows_std,
        "cols_mean": cols_mean,
        "cols_std": cols_std,
        "pred_az": pred_az,
        "pred_rg": pred_rg,
        "res_az": res_az,
        "res_rg": res_rg,
    }


def _coeff_to_params(
    coeff_az: np.ndarray,
    coeff_rg: np.ndarray,
    polynomial_order: int,
) -> Dict[str, float]:
    coeff_az = np.asarray(coeff_az, dtype=np.float64)
    coeff_rg = np.asarray(coeff_rg, dtype=np.float64)
    if polynomial_order <= 1:
        coeff_az = np.pad(coeff_az, (0, max(0, 6 - coeff_az.size)))
        coeff_rg = np.pad(coeff_rg, (0, max(0, 6 - coeff_rg.size)))
    params = {
        "a0": float(coeff_az[0]),
        "a1": float(coeff_az[1]),
        "a2": float(coeff_az[2]),
        "a3": float(coeff_az[3]),
        "a4": float(coeff_az[4]),
        "a5": float(coeff_az[5]),
        "b0": float(coeff_rg[0]),
        "b1": float(coeff_rg[1]),
        "b2": float(coeff_rg[2]),
        "b3": float(coeff_rg[3]),
        "b4": float(coeff_rg[4]),
        "b5": float(coeff_rg[5]),
    }
    return params


class OffsetFitter:
    def __init__(
        self,
        polynomial_order: int = 2,
        max_iterations: int = 10,
        initial_outlier_threshold: float = 2.0,
        min_correlation: float = 0.0,
    ):
        self.polynomial_order = polynomial_order
        self.max_iterations = max_iterations
        self.initial_outlier_threshold = initial_outlier_threshold
        self.min_correlation = min_correlation

    def fit(
        self,
        offsets: List[Dict[str, float]],
    ) -> Optional[OffsetFitResult]:
        if not offsets:
            return None

        arr_cols = np.array([p["col"] for p in offsets], dtype=np.float64)
        arr_rows = np.array([p["row"] for p in offsets], dtype=np.float64)
        arr_rg = np.array([p["range"] for p in offsets], dtype=np.float64)
        arr_az = np.array([p["azimuth"] for p in offsets], dtype=np.float64)
        arr_corr = np.array([p.get("correlation", 0.0) for p in offsets], dtype=np.float64)

        valid_mask = np.isfinite(arr_cols) & np.isfinite(arr_rows) & np.isfinite(arr_rg) & np.isfinite(arr_az)
        valid_mask &= arr_corr >= float(self.min_correlation)
        if np.count_nonzero(valid_mask) < 3:
            return None

        arr_cols = arr_cols[valid_mask]
        arr_rows = arr_rows[valid_mask]
        arr_rg = arr_rg[valid_mask]
        arr_az = arr_az[valid_mask]

        polynomial_order = 2 if self.polynomial_order >= 2 else 1
        min_points = 6 if polynomial_order == 2 else 3
        if arr_rows.size < min_points:
            polynomial_order = 1
            min_points = 3
        if arr_rows.size < min_points:
            return None

        inlier_mask = np.ones(arr_rows.size, dtype=bool)
        fit = None
        for _ in range(max(1, int(self.max_iterations))):
            if np.count_nonzero(inlier_mask) < min_points:
                break
            fit = _fit_once(
                arr_rows[inlier_mask],
                arr_cols[inlier_mask],
                arr_az[inlier_mask],
                arr_rg[inlier_mask],
                polynomial_order,
            )

            rows_mean = fit["rows_mean"]
            rows_std = fit["rows_std"]
            cols_mean = fit["cols_mean"]
            cols_std = fit["cols_std"]
            rows_norm_all = (arr_rows - rows_mean) / rows_std
            cols_norm_all = (arr_cols - cols_mean) / cols_std
            A_all = _build_design_matrix(rows_norm_all, cols_norm_all, polynomial_order)
            pred_az_all = A_all @ fit["coeff_az"]
            pred_rg_all = A_all @ fit["coeff_rg"]
            res_az_all = arr_az - pred_az_all
            res_rg_all = arr_rg - pred_rg_all

            az_sigma = float(np.std(res_az_all[inlier_mask])) if np.count_nonzero(inlier_mask) > 1 else 0.0
            rg_sigma = float(np.std(res_rg_all[inlier_mask])) if np.count_nonzero(inlier_mask) > 1 else 0.0
            az_sigma = max(az_sigma, 1e-6)
            rg_sigma = max(rg_sigma, 1e-6)
            az_thr = float(self.initial_outlier_threshold) * az_sigma
            rg_thr = float(self.initial_outlier_threshold) * rg_sigma

            new_mask = (np.abs(res_az_all) <= az_thr) & (np.abs(res_rg_all) <= rg_thr)
            if np.count_nonzero(new_mask) < min_points:
                break
            if np.array_equal(new_mask, inlier_mask):
                inlier_mask = new_mask
                break
            inlier_mask = new_mask

        if fit is None:
            fit = _fit_once(arr_rows, arr_cols, arr_az, arr_rg, polynomial_order)
            inlier_mask = np.ones(arr_rows.size, dtype=bool)

        fit = _fit_once(
            arr_rows[inlier_mask],
            arr_cols[inlier_mask],
            arr_az[inlier_mask],
            arr_rg[inlier_mask],
            polynomial_order,
        )

        params = _coeff_to_params(fit["coeff_az"], fit["coeff_rg"], polynomial_order)
        az_rms = float(np.sqrt(np.mean(fit["res_az"] ** 2))) if fit["res_az"].size else 0.0
        rg_rms = float(np.sqrt(np.mean(fit["res_rg"] ** 2))) if fit["res_rg"].size else 0.0

        return OffsetFitResult(
            polynomial_order=int(polynomial_order),
            parameters=params,
            normalization={
                "rows_mean": float(fit["rows_mean"]),
                "rows_std": float(fit["rows_std"]),
                "cols_mean": float(fit["cols_mean"]),
                "cols_std": float(fit["cols_std"]),
            },
            azimuth_rms=az_rms,
            range_rms=rg_rms,
            final_points=int(np.count_nonzero(inlier_mask)),
            outliers_removed=int(arr_rows.size - np.count_nonzero(inlier_mask)),
        )

    def predict(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        result: OffsetFitResult,
    ) -> np.ndarray:
        rows_norm = (rows - result.normalization["rows_mean"]) / result.normalization["rows_std"]
        cols_norm = (cols - result.normalization["cols_mean"]) / result.normalization["cols_std"]
        
        order = result.polynomial_order
        if order <= 1:
            A = np.column_stack([
                np.ones_like(rows_norm),
                rows_norm,
                cols_norm,
            ])
        else:
            A = np.column_stack([
                np.ones_like(rows_norm),
                rows_norm,
                cols_norm,
                rows_norm * cols_norm,
                rows_norm * rows_norm,
                cols_norm * cols_norm,
            ])
        
        coeff_az = np.array([
            result.parameters["a0"],
            result.parameters["a1"],
            result.parameters["a2"],
            result.parameters["a3"],
            result.parameters["a4"],
            result.parameters["a5"],
        ])[:A.shape[1]]
        
        coeff_rg = np.array([
            result.parameters["b0"],
            result.parameters["b1"],
            result.parameters["b2"],
            result.parameters["b3"],
            result.parameters["b4"],
            result.parameters["b5"],
        ])[:A.shape[1]]
        
        az_pred = A @ coeff_az
        rg_pred = A @ coeff_rg
        
        return np.stack([az_pred, rg_pred], axis=-1)


def fit_offsets(
    offsets: List[Dict[str, float]],
    polynomial_order: int = 2,
    max_iterations: int = 10,
    initial_outlier_threshold: float = 2.0,
    min_correlation: float = 0.0,
) -> Optional[Dict[str, Any]]:
    fitter = OffsetFitter(
        polynomial_order=polynomial_order,
        max_iterations=max_iterations,
        initial_outlier_threshold=initial_outlier_threshold,
        min_correlation=min_correlation,
    )
    
    result = fitter.fit(offsets)
    if result is None:
        return None
    
    return {
        "polynomial_order": result.polynomial_order,
        "parameters": result.parameters,
        "normalization": result.normalization,
        "azimuth_rms": result.azimuth_rms,
        "range_rms": result.range_rms,
        "final_points": result.final_points,
        "outliers_removed": result.outliers_removed,
    }
