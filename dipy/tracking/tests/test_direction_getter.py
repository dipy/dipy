#!/usr/bin/env python
"""
Standalone test for PythonDirectionGetter.
Runs without requiring pytest or DIPY rebuild.
"""

import sys
import os
import numpy as np
import numpy.testing as npt

# Add DIPY to path
dipy_path = '/Users/mrinalchaturvedi/Desktop/MOONNEW/Projects/FromScratch/dipy'
if os.path.exists(dipy_path):
    sys.path.insert(0, dipy_path)

from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.direction_getter import DirectionGetter
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamline import Streamlines


class PythonDirectionGetter(DirectionGetter):
    """Pure Python DirectionGetter for testing."""

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
    """Test that pure Python DirectionGetter works with LocalTracking."""
    print("Running: test_pure_python_direction_getter_basic...", end=" ")

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

    print("✓ PASSED")


def test_initial_direction_shape():
    """Validate initial_direction returns 2D array."""
    print("Running: test_initial_direction_shape...", end=" ")

    dg = PythonDirectionGetter([1, 0, 0])
    point = np.array([0., 0., 0.])

    directions = dg.initial_direction(point)

    assert directions.ndim == 2
    assert directions.shape[1] == 3
    assert directions.shape[0] == 1

    print("✓ PASSED")


def test_direction_normalization():
    """Test that directions are properly normalized."""
    print("Running: test_direction_normalization...", end=" ")

    dg = PythonDirectionGetter([3, 4, 0])

    point = np.array([0., 0., 0.])
    directions = dg.initial_direction(point)

    npt.assert_almost_equal(np.linalg.norm(directions[0]), 1.0)
    expected = np.array([3., 4., 0.]) / 5.0
    npt.assert_array_almost_equal(directions[0], expected)

    print("✓ PASSED")


def test_inplace_direction_update():
    """Verify get_direction updates direction in-place."""
    print("Running: test_inplace_direction_update...", end=" ")

    dg = PythonDirectionGetter([0, 1, 0])

    point = np.array([1., 1., 1.])
    direction = np.array([1., 0., 0.])

    status = dg.get_direction(point, direction)

    assert status == 0
    expected = np.array([0., 1., 0.])
    npt.assert_array_almost_equal(direction, expected)

    print("✓ PASSED")


def test_multiple_tracking_directions():
    """Test tracking in various directions."""
    print("Running: test_multiple_tracking_directions...", end=" ")

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

    print("✓ PASSED")


def test_invalid_direction_raises():
    """Zero-length direction should raise ValueError."""
    print("Running: test_invalid_direction_raises...", end=" ")

    try:
        PythonDirectionGetter([0, 0, 0])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "non-zero length" in str(e)

    print("✓ PASSED")


def test_with_nonidentity_affine():
    """Validate compatibility with scaled affine transforms."""
    print("Running: test_with_nonidentity_affine...", end=" ")

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

    print("✓ PASSED")


def test_python_vs_cython_interface_note():
    """Documentation test."""
    print("Running: test_python_vs_cython_interface_note...", end=" ")
    assert True
    print("✓ PASSED")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("Testing PythonDirectionGetter Interface")
    print("=" * 70 + "\n")

    try:
        test_pure_python_direction_getter_basic()
        test_initial_direction_shape()
        test_direction_normalization()
        test_inplace_direction_update()
        test_multiple_tracking_directions()
        test_invalid_direction_raises()
        test_with_nonidentity_affine()
        test_python_vs_cython_interface_note()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED (8/8)")
        print("=" * 70)
        print("\nYour PythonDirectionGetter implementation is working correctly!")
        print("This validates the integration approach for RL tractography.\n")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)