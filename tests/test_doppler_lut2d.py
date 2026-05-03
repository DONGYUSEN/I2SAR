"""Tests for DopplerLUT2d."""

from __future__ import annotations

import numpy as np
import pytest

from i2sar.geometry import DopplerLUT2d


class TestDopplerLUT2dConstruction:
    def test_from_centroid_and_slopes(self):
        range_axis = np.linspace(400_000.0, 500_000.0, 11)
        azimuth_axis = np.linspace(0.0, 10.0, 21)

        lut = DopplerLUT2d.from_centroid_and_slopes(
            centroid_hz=500.0,
            range_slope=0.1,
            azimuth_slope=5.0,
            range_axis=range_axis,
            azimuth_axis=azimuth_axis,
        )

        assert lut.range_axis.shape == (11,)
        assert lut.azimuth_axis.shape == (21,)
        assert lut.data.shape == (21, 11)
        assert lut.validate()

    def test_eval_at_grid_point(self):
        range_axis = np.array([400_000.0, 450_000.0, 500_000.0])
        azimuth_axis = np.array([0.0, 5.0, 10.0])

        lut = DopplerLUT2d.from_centroid_and_slopes(
            centroid_hz=500.0,
            range_slope=0.0,
            azimuth_slope=0.0,
            range_axis=range_axis,
            azimuth_axis=azimuth_axis,
        )

        result = lut.eval(5.0, 450_000.0)
        assert np.isclose(result, 500.0)

    def test_eval_interpolates_bilinearly(self):
        range_axis = np.array([400_000.0, 500_000.0])
        azimuth_axis = np.array([0.0, 10.0])

        lut = DopplerLUT2d(
            range_axis=range_axis,
            azimuth_axis=azimuth_axis,
            data=np.array([[0.0, 100.0], [200.0, 300.0]]),
        )

        result = lut.eval(5.0, 450_000.0)
        expected = 150.0
        assert np.isclose(result, expected)

    def test_eval_extrapolates(self):
        range_axis = np.array([400_000.0, 500_000.0])
        azimuth_axis = np.array([0.0, 10.0])

        lut = DopplerLUT2d(
            range_axis=range_axis,
            azimuth_axis=azimuth_axis,
            data=np.array([[0.0, 100.0], [200.0, 300.0]]),
        )

        result = lut.eval(5.0, 350_000.0)
        assert np.isfinite(result)

    def test_rejects_invalid_shape(self):
        range_axis = np.array([400_000.0, 500_000.0])
        azimuth_axis = np.array([0.0, 10.0])

        with pytest.raises(ValueError, match="must match"):
            DopplerLUT2d(
                range_axis=range_axis,
                azimuth_axis=azimuth_axis,
                data=np.array([[0.0, 100.0, 200.0]]),
            )

    def test_validate_rejects_non_monotonic(self):
        range_axis = np.array([500_000.0, 400_000.0])
        azimuth_axis = np.array([0.0, 10.0])

        lut = DopplerLUT2d(
            range_axis=range_axis,
            azimuth_axis=azimuth_axis,
            data=np.array([[0.0, 100.0], [200.0, 300.0]]),
        )

        assert not lut.validate()

    def test_eval_vectorized(self):
        range_axis = np.linspace(400_000.0, 500_000.0, 11)
        azimuth_axis = np.linspace(0.0, 10.0, 11)

        lut = DopplerLUT2d.from_centroid_and_slopes(
            centroid_hz=500.0,
            range_slope=0.0,
            azimuth_slope=0.0,
            range_axis=range_axis,
            azimuth_axis=azimuth_axis,
        )

        ranges = np.array([420_000.0, 450_000.0, 480_000.0])
        times = np.array([2.0, 5.0, 8.0])

        for r, t in zip(ranges, times):
            result = lut.eval(t, r)
            assert np.isclose(result, 500.0, atol=1e-6)


class TestDopplerLUT2dHdf5:
    def test_write_and_read_doppler_lut2d(self):
        import tempfile
        import h5py

        range_axis = np.linspace(400_000.0, 500_000.0, 11)
        azimuth_axis = np.linspace(0.0, 10.0, 21)

        lut = DopplerLUT2d.from_centroid_and_slopes(
            centroid_hz=750.0,
            range_slope=0.05,
            azimuth_slope=2.0,
            range_axis=range_axis,
            azimuth_axis=azimuth_axis,
        )

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            with h5py.File(f.name, "w") as h5:
                h5.require_group("doppler")
                h5["doppler"].create_dataset("range_axis", data=lut.range_axis)
                h5["doppler"].create_dataset("azimuth_axis", data=lut.azimuth_axis)
                h5["doppler"].create_dataset("data", data=lut.data)

            with h5py.File(f.name, "r") as h5:
                lut_read = DopplerLUT2d(
                    range_axis=h5["doppler"]["range_axis"][...],
                    azimuth_axis=h5["doppler"]["azimuth_axis"][...],
                    data=h5["doppler"]["data"][...],
                )

                assert lut_read.data.shape == lut.data.shape
                np.testing.assert_array_almost_equal(lut_read.range_axis, lut.range_axis)
                np.testing.assert_array_almost_equal(lut_read.azimuth_axis, lut.azimuth_axis)
