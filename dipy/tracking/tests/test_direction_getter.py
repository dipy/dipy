"""
Test DirectionGetter interface for pure Python implementations.
This is essential for:
-ML-based tractography methods using PyTorch/TensorFlow
Note on Performance:
    Pure Python DirectionGetter implementations are ~10-100x slower than
    Cython implementations but enable full compatibility with Python ML
    frameworks and easier debugging.
"""

import numpy as np
import numpy.testing as npt
import pytest

from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.direction_getter import DirectionGetter
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamline import Streamlines


class PythonDirectionGetter(DirectionGetter):

    def __init__(self, direction):
        direction = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            raise ValueError("Direction must have non-zero length")
        self.direction = direction / norm

    def initial_direction(self, point):

        return np.array([self.direction], dtype=np.float64)

    def get_direction(self, point, direction):

        direction[:] = self.direction
        return 0


def test_pure_python_direction_getter_basic():

    data = np.ones((5, 5, 5))
    stopping = BinaryStoppingCriterion(data)
    seeds = np.array([[2., 2., 2.]])

    dg = PythonDirectionGetter([1, 0, 0])

    tracking = LocalTracking(
        dg, stopping, seeds, np.eye(4),
        step_size=0.5, maxlen=10
    )

    streamlines = Streamlines(tracking)
    assert len(streamlines) > 0

    streamline = streamlines[0]
    assert len(streamline) > 1
    x_coords = streamline[:, 0]
    assert np.all(np.diff(x_coords) > 0), "Should track in +x direction"


def test_initial_direction_shape():

    dg = PythonDirectionGetter([1, 0, 0])
    point = np.array([0., 0., 0.])

    directions = dg.initial_direction(point)

    assert directions.ndim == 2
    assert directions.shape[1] == 3
    assert directions.shape[0] == 1


def test_direction_normalization():

    dg = PythonDirectionGetter([3, 4, 0])

    point = np.array([0., 0., 0.])
    directions = dg.initial_direction(point)

    npt.assert_almost_equal(np.linalg.norm(directions[0]), 1.0)
    expected = np.array([3., 4., 0.]) / 5.0
    npt.assert_array_almost_equal(directions[0], expected)


def test_inplace_direction_update():

    dg = PythonDirectionGetter([0, 1, 0])

    point = np.array([1., 1., 1.])
    direction = np.array([1., 0., 0.])

    status = dg.get_direction(point, direction)

    assert status == 0
    expected = np.array([0., 1., 0.])
    npt.assert_array_almost_equal(direction, expected)


def test_multiple_tracking_directions():

    data = np.ones((10, 10, 10))
    stopping = BinaryStoppingCriterion(data)
    seeds = np.array([[5., 5., 5.]])

    test_directions = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
    ]

    for direction in test_directions:
        dg = PythonDirectionGetter(direction)
        tracking = LocalTracking(
            dg, stopping, seeds, np.eye(4),
            step_size=0.5, maxlen=5
        )
        streamlines = list(tracking)

        assert len(streamlines) > 0

        streamline = streamlines[0]
        displacement = streamline[-1] - streamline[0]

        expected_dir = np.array(direction, dtype=float)
        expected_dir /= np.linalg.norm(expected_dir)

        displacement_norm = displacement / np.linalg.norm(displacement)
        dot_product = np.dot(displacement_norm, expected_dir)
        assert dot_product > 0.99


def test_invalid_direction_raises():

    with pytest.raises(ValueError, match="non-zero length"):
        PythonDirectionGetter([0, 0, 0])


def test_with_nonidentity_affine():

    data = np.ones((5, 5, 5))
    stopping = BinaryStoppingCriterion(data)
    seeds = np.array([[2., 2., 2.]])

    affine = np.diag([2., 2., 2., 1.])

    dg = PythonDirectionGetter([1, 0, 0])
    tracking = LocalTracking(
        dg, stopping, seeds, affine,
        step_size=1.0, maxlen=10
    )

    streamlines = Streamlines(tracking)
    assert len(streamlines) > 0
