import h5py
import numpy as np
import pytest

from i2sar.orbit import OrbitInterpolator, read_orbit_interpolator


def test_orbit_interpolator_sorts_time_and_interpolates_constant_velocity():
    orbit = OrbitInterpolator(
        time=np.array([20.0, 0.0, 10.0]),
        position=np.array([[20.0, 40.0, 60.0], [0.0, 0.0, 0.0], [10.0, 20.0, 30.0]]),
        velocity=np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
    )

    state = orbit.state_at(5.0)

    assert np.allclose(state.position, [5.0, 10.0, 15.0])
    assert np.allclose(state.velocity, [1.0, 2.0, 3.0])


def test_orbit_interpolator_vectorizes_state_queries():
    orbit = OrbitInterpolator(
        time=np.array([0.0, 10.0, 20.0]),
        position=np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0], [20.0, 40.0, 60.0]]),
        velocity=np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
    )

    state = orbit.state_at(np.array([0.0, 5.0, 20.0]))

    assert state.position.shape == (3, 3)
    assert np.allclose(state.position[1], [5.0, 10.0, 15.0])
    assert np.allclose(state.velocity[2], [1.0, 2.0, 3.0])


def test_orbit_interpolator_rejects_out_of_bounds_time_by_default():
    orbit = OrbitInterpolator(
        time=np.array([0.0, 10.0]),
        position=np.zeros((2, 3)),
        velocity=np.zeros((2, 3)),
    )

    with pytest.raises(ValueError, match="outside orbit time span"):
        orbit.state_at(11.0)


def test_read_orbit_interpolator_from_scene_hdf5(tmp_path):
    scene_path = tmp_path / "scene.h5"
    with h5py.File(scene_path, "w") as h5:
        orbit = h5.require_group("orbit")
        orbit.create_dataset("time", data=np.array([0.0, 10.0]))
        orbit.create_dataset("position", data=np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]]))
        orbit.create_dataset("velocity", data=np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))

    orbit = read_orbit_interpolator(scene_path)

    assert np.allclose(orbit.state_at(5.0).position, [5.0, 10.0, 15.0])
