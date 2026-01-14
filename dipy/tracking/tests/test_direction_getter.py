"""
Test DirectionGetter interface for pure Python implementations.

This module validates that DirectionGetter subclasses implemented in pure
Python (overriding get_direction() instead of get_direction_c()) work
correctly with LocalTracking.

This is essential for:
- ML-based tractography methods using PyTorch/TensorFlow
- Rapid prototyping of novel tracking algorithms
- Educational purposes and documentation

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
    """
    Pure Python DirectionGetter returning a constant direction.

    Demonstrates that DirectionGetter can be subclassed in pure Python
    by overriding get_direction() instead of implementing get_direction_c().
    This pattern is important for ML-based methods requiring PyTorch/TF.

    Parameters
    ----------
    direction : array-like, shape (3,)
        Constant direction vector (will be normalized).

    Examples
    --------
    >>> dg = PythonDirectionGetter([1, 0, 0])
    >>> point = np.array([0., 0., 0.])
    >>> dirs = dg.initial_direction(point)
    >>> print(dirs.shape)
    (1, 3)
    """

    def __init__(self, direction):
        direction = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            raise ValueError("Direction must have non-zero length")
        self.direction = direction / norm

    def initial_direction(self, point):
        """
        Return initial direction(s) at seed point.

        Returns
        -------
        directions : ndarray, shape (n_directions, 3)
            Array of initial directions. For deterministic: (1, 3).
        """
        return np.array([self.direction], dtype=np.float64)

    def get_direction(self, point, direction):
        """
        Pure Python direction getter.

        Overrides get_direction() wrapper instead of implementing
        get_direction_c(). Slower but maintains Python compatibility.

        Parameters
        ----------
        point : memoryview, shape (3,)
            Current position.
        direction : memoryview, shape (3,)
            Direction array (modified in-place).

        Returns
        -------
        status : int
            0 for success, 1 for termination.
        """
        direction[:] = self.direction
        return 0


def test_pure_python_direction_getter_basic():
    """Test that pure Python DirectionGetter works with LocalTracking."""
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
    """Validate initial_direction returns 2D array."""
    dg = PythonDirectionGetter([1, 0, 0])
    point = np.array([0., 0., 0.])

    directions = dg.initial_direction(point)

    assert directions.ndim == 2
    assert directions.shape[1] == 3
    assert directions.shape[0] == 1


def test_direction_normalization():
    """Test that directions are properly normalized."""
    dg = PythonDirectionGetter([3, 4, 0])

    point = np.array([0., 0., 0.])
    directions = dg.initial_direction(point)

    npt.assert_almost_equal(np.linalg.norm(directions[0]), 1.0)
    expected = np.array([3., 4., 0.]) / 5.0
    npt.assert_array_almost_equal(directions[0], expected)


def test_inplace_direction_update():
    """Verify get_direction updates direction in-place."""
    dg = PythonDirectionGetter([0, 1, 0])

    point = np.array([1., 1., 1.])
    direction = np.array([1., 0., 0.])

    status = dg.get_direction(point, direction)

    assert status == 0
    expected = np.array([0., 1., 0.])
    npt.assert_array_almost_equal(direction, expected)


def test_multiple_tracking_directions():
    """Test tracking in various directions."""
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
    """Zero-length direction should raise ValueError."""
    with pytest.raises(ValueError, match="non-zero length"):
        PythonDirectionGetter([0, 0, 0])


def test_with_nonidentity_affine():
    """Validate compatibility with scaled affine transforms."""
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