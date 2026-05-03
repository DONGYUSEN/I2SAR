"""Tests for Doppler model."""

from __future__ import annotations

import numpy as np

from i2sar.geometry import Doppler


class TestDopplerConstant:
    def test_doppler_constant(self):
        d = Doppler.constant(centroid_hz=500.0)
        assert d.centroid_hz == 500.0
        assert d.toa_poly is None

    def test_doppler_evaluate_returns_centroid(self):
        d = Doppler.constant(centroid_hz=500.0)
        result = d.evaluate(slant_range_m=500_000.0)
        assert result == 500.0

    def test_doppler_evaluate_ignores_slant_range(self):
        d = Doppler.constant(centroid_hz=500.0)
        result1 = d.evaluate(slant_range_m=400_000.0)
        result2 = d.evaluate(slant_range_m=600_000.0)
        assert result1 == result2 == 500.0


class TestDopplerPolynomial:
    def test_doppler_polynomial(self):
        coeffs = np.array([1.0, 2.0, 3.0])
        d = Doppler.polynomial(centroid_hz=500.0, toa_poly=coeffs)
        assert d.centroid_hz == 500.0
        np.testing.assert_array_equal(d.toa_poly, coeffs)

    def test_doppler_polynomial_evaluate(self):
        coeffs = np.array([0.0, 100.0])
        d = Doppler.polynomial(centroid_hz=500.0, toa_poly=coeffs)
        result = d.evaluate(slant_range_m=500_000.0)
        expected = 600.0
        assert np.isclose(result, expected)


class TestDopplerHdf5:
    def test_write_and_read_doppler(self):
        import tempfile
        import h5py

        d = Doppler.constant(centroid_hz=750.0)
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            with h5py.File(f.name, "w") as h5:
                h5.require_group("doppler")
                from i2sar.geometry import write_doppler_to_hdf5
                write_doppler_to_hdf5(h5, d)

            with h5py.File(f.name, "r") as h5:
                from i2sar.geometry import read_doppler_from_hdf5
                d_read = read_doppler_from_hdf5(h5)
                assert d_read.centroid_hz == 750.0
                assert d_read.toa_poly is None

    def test_write_and_read_polynomial_doppler(self):
        import tempfile
        import h5py

        coeffs = np.array([1.0, 2.0, 3.0])
        d = Doppler.polynomial(centroid_hz=750.0, toa_poly=coeffs)
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            with h5py.File(f.name, "w") as h5:
                h5.require_group("doppler")
                from i2sar.geometry import write_doppler_to_hdf5
                write_doppler_to_hdf5(h5, d)

            with h5py.File(f.name, "r") as h5:
                from i2sar.geometry import read_doppler_from_hdf5
                d_read = read_doppler_from_hdf5(h5)
                assert d_read.centroid_hz == 750.0
                np.testing.assert_array_equal(d_read.toa_poly, coeffs)
