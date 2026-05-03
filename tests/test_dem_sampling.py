"""Tests for DEM sampling."""

from __future__ import annotations

import numpy as np
import pytest


class TestSampleDemAtLatlons:
    def test_sample_dem_returns_elevation_at_point(self):
        from i2sar.dem.sampling import sample_dem_at_latlons

        elevation = np.array([
            [100.0, 200.0, 300.0],
            [150.0, 250.0, 350.0],
            [200.0, 300.0, 400.0],
        ])

        geotransform = [0.0, 1.0 / (3 - 1), 0.0, 1.0, 0.0, -1.0 / (3 - 1)]

        lats = np.array([0.5])
        lons = np.array([0.5])

        result = sample_dem_at_latlons(lats, lons, elevation, geotransform)

        assert np.isfinite(result[0])

    def test_sample_dem_bilinear_interpolation(self):
        from i2sar.dem.sampling import sample_dem_at_latlons

        elevation = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0],
        ])

        geotransform = [0.0, 1.0 / 2.0, 0.0, 1.0, 0.0, -1.0 / 2.0]

        lats = np.array([0.5])
        lons = np.array([0.5])

        result = sample_dem_at_latlons(lats, lons, elevation, geotransform)

        np.testing.assert_allclose(result[0], 100.0, atol=1.0)

    def test_sample_dem_returns_nan_for_out_of_bounds(self):
        from i2sar.dem.sampling import sample_dem_at_latlons

        elevation = np.array([
            [100.0, 200.0],
            [150.0, 250.0],
        ])

        geotransform = [0.0, 1.0, 0.0, 1.0, 0.0, -1.0]

        lats = np.array([5.0])
        lons = np.array([5.0])

        result = sample_dem_at_latlons(lats, lons, elevation, geotransform)

        assert np.isnan(result[0])

    def test_sample_dem_vectorized(self):
        from i2sar.dem.sampling import sample_dem_at_latlons

        elevation = np.array([
            [100.0, 200.0, 300.0],
            [150.0, 250.0, 350.0],
            [200.0, 300.0, 400.0],
        ])

        geotransform = [0.0, 1.0 / 2.0, 0.0, 1.0, 0.0, -1.0 / 2.0]

        lats = np.array([0.25, 0.5, 0.75])
        lons = np.array([0.25, 0.5, 0.75])

        result = sample_dem_at_latlons(lats, lons, elevation, geotransform)

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))


class TestSampleDemProduct:
    def test_sample_dem_product(self):
        import tempfile
        import h5py

        from i2sar.dem.sampling import sample_dem_product

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = f"{tmpdir}/dem.h5"

            elevation = np.array([
                [100.0, 200.0],
                [150.0, 250.0],
            ], dtype=np.float32)
            mask = np.isfinite(elevation)
            geotransform = [0.0, 1.0, 0.0, 1.0, 0.0, -1.0]

            with h5py.File(h5_path, "w") as h5:
                h5.require_group("dem")
                h5["dem"].create_dataset("elevation", data=elevation)
                h5["dem"].create_dataset("mask", data=mask)
                h5["dem"].attrs["geotransform"] = np.array(geotransform, dtype=np.float64)

            lats = np.array([0.5])
            lons = np.array([0.5])

            result = sample_dem_product(h5_path, lats, lons)

            assert np.isfinite(result[0])
