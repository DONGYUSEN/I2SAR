"""Tests for DEMInterpolator - reference implementation from ISCE3."""

from __future__ import annotations

import numpy as np
import pytest

from i2sar.dem import DEMInterpolator, InterpMethod


class TestDEMInterpolatorConstruction:
    def test_create_dem_interpolator(self):
        elevation = np.array([[100.0, 200.0], [150.0, 250.0]], dtype=np.float32)
        geotransform = [0.0, 1.0, 0.0, 1.0, 0.0, -1.0]
        dem = DEMInterpolator(elevation=elevation, geotransform=geotransform)
        assert dem.width() == 2
        assert dem.length() == 2
        assert dem.x_start() == 0.0
        assert dem.y_start() == 1.0
        assert dem.delta_x() == 1.0
        assert dem.delta_y() == -1.0

    def test_from_hdf5(self):
        import tempfile
        import h5py

        elevation = np.array([[100.0, 200.0], [150.0, 250.0]], dtype=np.float32)
        geotransform = [0.0, 1.0, 0.0, 1.0, 0.0, -1.0]

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            with h5py.File(f.name, "w") as h5:
                h5.require_group("dem")
                h5["dem"].create_dataset("elevation", data=elevation)
                h5["dem"].attrs["geotransform"] = np.array(geotransform, dtype=np.float64)
                h5["dem"].attrs["epsg"] = 4326

            dem_read = DEMInterpolator.from_hdf5(f.name)
            assert dem_read.width() == 2
            assert dem_read.length() == 2


class TestDEMInterpolatorBilinear:
    def test_bilinear_at_center(self):
        elevation = np.array([[0.0, 100.0], [200.0, 300.0]], dtype=np.float32)
        geotransform = [0.0, 1.0, 0.0, 1.0, 0.0, -1.0]
        dem = DEMInterpolator(
            elevation=elevation,
            geotransform=geotransform,
            interp_method=InterpMethod.BILINEAR,
        )

        result = dem.interpolate_at_xy(0.5, 0.5)
        expected = 150.0
        assert np.isclose(result, expected)

    def test_bilinear_at_grid_point(self):
        elevation = np.array([[0.0, 100.0], [200.0, 300.0]], dtype=np.float32)
        geotransform = [0.0, 1.0, 0.0, 1.0, 0.0, -1.0]
        dem = DEMInterpolator(
            elevation=elevation,
            geotransform=geotransform,
            interp_method=InterpMethod.BILINEAR,
        )

        result = dem.interpolate_at_xy(0.5, 0.5)
        assert np.isclose(result, 150.0)

        result = dem.interpolate_at_xy(0.99, 0.5)
        assert np.isclose(result, 199.0)

        result = dem.interpolate_at_xy(0.5, 0.99)
        assert np.isclose(result, 52.0)

        result = dem.interpolate_at_xy(0.99, 0.99)
        assert np.isclose(result, 101.0)

    def test_bilinear_out_of_bounds_returns_ref_height(self):
        elevation = np.array([[0.0, 100.0], [200.0, 300.0]], dtype=np.float32)
        geotransform = [0.0, 1.0, 0.0, 1.0, 0.0, -1.0]
        dem = DEMInterpolator(
            elevation=elevation,
            geotransform=geotransform,
            ref_height=1000.0,
            interp_method=InterpMethod.BILINEAR,
        )

        result = dem.interpolate_at_xy(5.0, 5.0)
        assert result == 1000.0


class TestDEMInterpolatorNearest:
    def test_nearest_at_center(self):
        elevation = np.array([[0.0, 100.0], [200.0, 300.0]], dtype=np.float32)
        geotransform = [0.0, 1.0, 0.0, 1.0, 0.0, -1.0]
        dem = DEMInterpolator(
            elevation=elevation,
            geotransform=geotransform,
            interp_method=InterpMethod.NEAREST,
        )

        result = dem.interpolate_at_xy(0.4, 0.4)
        assert np.isclose(result, 200.0)

        result = dem.interpolate_at_xy(0.6, 0.6)
        assert np.isclose(result, 100.0)

    def test_nearest_out_of_bounds_returns_ref_height(self):
        elevation = np.array([[0.0, 100.0], [200.0, 300.0]], dtype=np.float32)
        geotransform = [0.0, 1.0, 0.0, 1.0, 0.0, -1.0]
        dem = DEMInterpolator(
            elevation=elevation,
            geotransform=geotransform,
            ref_height=1000.0,
            interp_method=InterpMethod.NEAREST,
        )

        result = dem.interpolate_at_xy(5.0, 5.0)
        assert result == 1000.0


class TestDEMInterpolatorBicubic:
    def test_bicubic_smooth_gradient(self):
        x = np.arange(4.0)
        y = np.arange(4.0)
        xx, yy = np.meshgrid(x, y)
        elevation = (xx + yy * 10).astype(np.float32)
        geotransform = [0.0, 1.0, 0.0, 3.0, 0.0, -1.0]
        dem = DEMInterpolator(
            elevation=elevation,
            geotransform=geotransform,
            interp_method=InterpMethod.BICUBIC,
        )

        result = dem.interpolate_at_xy(1.5, 1.5)
        assert np.isfinite(result)
        assert 15 < result < 18

    def test_bicubic_vs_bilinear(self):
        elevation = np.array([
            [0.0, 100.0, 200.0],
            [300.0, 400.0, 500.0],
            [600.0, 700.0, 800.0],
        ], dtype=np.float32)
        geotransform = [0.0, 1.0, 0.0, 2.0, 0.0, -1.0]

        dem_bilinear = DEMInterpolator(
            elevation=elevation,
            geotransform=geotransform,
            interp_method=InterpMethod.BILINEAR,
        )
        dem_bicubic = DEMInterpolator(
            elevation=elevation,
            geotransform=geotransform,
            interp_method=InterpMethod.BICUBIC,
        )

        for row in [0.5, 1.0, 1.5]:
            for col in [0.5, 1.0, 1.5]:
                b_lin = dem_bilinear.interpolate_at_xy(col, row)
                b_cub = dem_bicubic.interpolate_at_xy(col, row)
                assert np.isclose(b_lin, b_cub, atol=1e-5)


class TestDEMInterpolatorStats:
    def test_min_max_mean(self):
        elevation = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        geotransform = [0.0, 1.0, 0.0, 1.0, 0.0, -1.0]
        dem = DEMInterpolator(elevation=elevation, geotransform=geotransform)

        assert dem.min_height() == 10.0
        assert dem.max_height() == 40.0
        assert dem.mean_height() == 25.0


class TestDEMInterpolatorLonLat:
    def test_interpolate_at_lonlat(self):
        elevation = np.array([[0.0, 100.0], [200.0, 300.0]], dtype=np.float32)
        geotransform = [-1.0, 2.0, 0.0, 1.0, 0.0, -2.0]
        dem = DEMInterpolator(
            elevation=elevation,
            geotransform=geotransform,
            epsg=4326,
        )

        result = dem.interpolate_at_lonlat(0.0, 0.0)
        assert np.isfinite(result)


class TestDEMInterpolatorDateline:
    def test_wrap_longitude_positive(self):
        elevation = np.array([[100.0, 200.0]], dtype=np.float32)
        geotransform = [170.0, 1.0, 0.0, 0.0, 0.0, -1.0]
        dem = DEMInterpolator(
            elevation=elevation,
            geotransform=geotransform,
            epsg=4326,
        )

        result = dem.interpolate_at_xy(185.0, 0.0)
        assert np.isfinite(result)

    def test_wrap_longitude_negative(self):
        elevation = np.array([[100.0, 200.0]], dtype=np.float32)
        geotransform = [-170.0, 1.0, 0.0, 0.0, 0.0, -1.0]
        dem = DEMInterpolator(
            elevation=elevation,
            geotransform=geotransform,
            epsg=4326,
        )

        result = dem.interpolate_at_xy(-185.0, 0.0)
        assert np.isfinite(result)
