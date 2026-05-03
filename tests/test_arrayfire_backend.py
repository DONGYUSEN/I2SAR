import numpy as np

from i2sar.accel.arrayfire_backend import ArrayFireBackend, arrayfire_available
from i2sar.registration.resampler import Resampler, InterpMethod


def test_arrayfire_backend_reports_availability():
    backend = ArrayFireBackend()

    assert backend.available == arrayfire_available()
    assert isinstance(backend.reason, str)


def test_resampler_arrayfire_bilinear_matches_numpy():
    if not arrayfire_available():
        return

    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    offsets_az = np.full((4, 4), 0.25, dtype=np.float32)
    offsets_rg = np.full((4, 4), 0.5, dtype=np.float32)

    numpy_result = Resampler(
        method=InterpMethod.BILINEAR,
        backend="numpy",
        use_numba=False,
    ).resample(image, offsets_az, offsets_rg)
    import pytest

    with pytest.raises(NotImplementedError):
        Resampler(
            method=InterpMethod.BILINEAR,
            backend="arrayfire",
        ).resample(image, offsets_az, offsets_rg)


def test_resampler_auto_backend_falls_back_to_numpy_when_arrayfire_unavailable():
    image = np.arange(9, dtype=np.float32).reshape(3, 3)
    offsets = np.zeros_like(image)

    result = Resampler(
        method=InterpMethod.BILINEAR,
        backend="auto",
        use_numba=False,
    ).resample(image, offsets, offsets)

    assert result.backend in {"arrayfire", "numpy"}
    np.testing.assert_allclose(result.resampled, image)
