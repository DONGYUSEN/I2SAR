from __future__ import annotations

import numpy as np
import pytest

from i2sar.geometry.tcn_basis import TCNBasis


class TestTCNBasis:
    def test_from_position_velocity(self):
        position = np.array([0.0, 0.0, 6371000.0])
        velocity = np.array([3000.0, 0.0, 0.0])

        basis = TCNBasis.from_position_velocity(position, velocity)

        assert np.allclose(np.linalg.norm(basis.t), 1.0, atol=1e-6)
        assert np.allclose(np.linalg.norm(basis.c), 1.0, atol=1e-6)
        assert np.allclose(np.linalg.norm(basis.n), 1.0, atol=1e-6)

        assert np.allclose(np.dot(basis.t, basis.c), 0.0, atol=1e-6)
        assert np.allclose(np.dot(basis.c, basis.n), 0.0, atol=1e-6)
        assert np.allclose(np.dot(basis.n, basis.t), 0.0, atol=1e-6)

    def test_project(self):
        position = np.array([0.0, 0.0, 6371000.0])
        velocity = np.array([3000.0, 0.0, 0.0])
        basis = TCNBasis.from_position_velocity(position, velocity)

        vec = np.array([1.0, 0.0, 0.0])
        projected = basis.project(vec)

        assert np.allclose(projected[0], np.dot(basis.t, vec), atol=1e-6)
        assert np.allclose(projected[1], np.dot(basis.c, vec), atol=1e-6)
        assert np.allclose(projected[2], np.dot(basis.n, vec), atol=1e-6)

    def test_combine(self):
        position = np.array([0.0, 0.0, 6371000.0])
        velocity = np.array([3000.0, 0.0, 0.0])
        basis = TCNBasis.from_position_velocity(position, velocity)

        weights = np.array([1.0, 0.0, 0.0])
        result = basis.combine(weights)
        assert np.allclose(result, basis.t, atol=1e-6)

        weights = np.array([0.0, 1.0, 0.0])
        result = basis.combine(weights)
        assert np.allclose(result, basis.c, atol=1e-6)

        weights = np.array([0.0, 0.0, 1.0])
        result = basis.combine(weights)
        assert np.allclose(result, basis.n, atol=1e-6)
