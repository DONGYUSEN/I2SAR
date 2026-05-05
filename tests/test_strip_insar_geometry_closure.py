from __future__ import annotations

import numpy as np

from i2sar.geometry.radar_grid import RadarGrid
from i2sar.interferometry.strip_insar import PairContext, StageResult, StripInSARProcessor


def test_stage_geo2rdr_exports_geometry_closure_artifacts(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir(parents=True)

    ctx = PairContext(
        master_scene_path=tmp_path / "master.h5",
        slave_scene_path=tmp_path / "slave.h5",
        output_root=out,
        pair_name="pair",
        wavelength=0.056,
    )
    proc = StripInSARProcessor(ctx, use_gpu=False)

    grid = RadarGrid(
        length=2,
        width=2,
        sensing_start_s=0.0,
        prf_hz=1.0,
        starting_range_m=10.0,
        range_pixel_spacing_m=1.0,
    )

    prep = StageResult(
        success=True,
        outputs={
            "master_grid": grid,
            "slave_grid": grid,
            "master_orbit": object(),
            "slave_orbit": object(),
        },
        metadata={},
    )
    topo = StageResult(
        success=True,
        outputs={
            "lat": np.zeros((2, 2), dtype=np.float64),
            "lon": np.zeros((2, 2), dtype=np.float64),
            "height": np.zeros((2, 2), dtype=np.float64),
            "incidence_angle": np.zeros((2, 2), dtype=np.float64),
        },
        metadata={},
    )

    def fake_slave(*args, **kwargs):
        az = np.full((2, 2), 3.0)
        rg = np.full((2, 2), 4.0)
        return az, rg, {"lines": np.full((2, 2), 1.0), "pixels": np.full((2, 2), 1.0), "valid_mask": np.ones((2, 2), dtype=bool)}

    def fake_rdrdem(*args, **kwargs):
        az = np.ones((2, 2))
        rg = np.zeros((2, 2))
        return az, rg, {"valid_mask": np.ones((2, 2), dtype=bool), "affine": [1, 0, 0, 1, 0, 0]}

    monkeypatch.setattr(proc, "_compute_slave_geo2rdr_offset", fake_slave)
    monkeypatch.setattr(proc, "_compute_master_rdrdem_offset", fake_rdrdem)

    result = proc.stage_geo2rdr(prep, topo, dem_path=str(tmp_path / "dem.h5"))

    assert "rdrdem_rg_offset" in result.outputs
    assert "rectified_range_offset" in result.outputs
    assert "merged_rg_offset" in result.outputs
    assert result.outputs["valid_mask"].shape == (2, 2)
