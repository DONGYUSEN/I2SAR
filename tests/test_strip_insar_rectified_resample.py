from __future__ import annotations

import numpy as np

from i2sar.geometry.radar_grid import RadarGrid
from i2sar.interferometry.strip_insar import PairContext, StageResult, StripInSARProcessor


def test_refined_resample_prefers_rectified_range_offsets(tmp_path, monkeypatch):
    ctx = PairContext(tmp_path / "m.h5", tmp_path / "s.h5", tmp_path / "out", "pair", 0.056)
    proc = StripInSARProcessor(ctx, use_gpu=False)

    grid = RadarGrid(4, 4, 0.0, 1.0, 0.0, 1.0)
    prep = StageResult(success=True, outputs={"slave_slc": np.ones((4, 4), dtype=np.complex64), "master_grid": grid}, metadata={})
    coarse = StageResult(
        success=True,
        outputs={
            "resampled_slave": np.ones((4, 4), dtype=np.complex64),
            "base_az_offset": np.zeros((4, 4), dtype=np.float64),
            "base_rg_offset": np.zeros((4, 4), dtype=np.float64),
            "rectified_range_offset": np.full((4, 4), 2.0, dtype=np.float64),
        },
        metadata={},
    )
    refine = StageResult(
        success=True,
        outputs={"az_offset": np.zeros((4, 4), dtype=np.float64), "rg_offset": np.zeros((4, 4), dtype=np.float64)},
        metadata={},
    )

    captured = {}

    def fake_resample(slave_slc, az_offset, rg_offset, num_rows, num_cols):
        captured["rg"] = rg_offset.copy()
        return np.zeros((num_rows, num_cols), dtype=np.complex64)

    monkeypatch.setattr(proc, "_refined_resample_cpu", fake_resample)

    _ = proc.stage_refined_resample(prep, coarse, refine)

    assert np.allclose(captured["rg"], 2.0)
