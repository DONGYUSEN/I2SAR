import numpy as np

from i2sar.geometry import WGS84, ecef_to_llh, llh_to_ecef


def test_llh_to_ecef_at_equator_prime_meridian():
    xyz = llh_to_ecef(0.0, 0.0, 0.0)

    assert np.allclose(xyz, [WGS84.a, 0.0, 0.0], atol=1e-6)


def test_llh_ecef_round_trip_scalar():
    lat, lon, height = 30.5, 101.25, 1234.5

    xyz = llh_to_ecef(lat, lon, height)
    out_lat, out_lon, out_height = ecef_to_llh(*xyz)

    assert np.isclose(out_lat, lat, atol=1e-10)
    assert np.isclose(out_lon, lon, atol=1e-10)
    assert np.isclose(out_height, height, atol=1e-5)


def test_llh_ecef_round_trip_vectorized():
    lat = np.array([0.0, 30.0, -45.0, 89.0])
    lon = np.array([0.0, 100.0, -120.0, 10.0])
    height = np.array([0.0, 100.0, 500.0, 10.0])

    x, y, z = llh_to_ecef(lat, lon, height)
    out_lat, out_lon, out_height = ecef_to_llh(x, y, z)

    assert np.allclose(out_lat, lat, atol=1e-10)
    assert np.allclose(out_lon, lon, atol=1e-10)
    assert np.allclose(out_height, height, atol=1e-5)
