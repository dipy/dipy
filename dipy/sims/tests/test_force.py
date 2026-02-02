"""Tests for FORCE simulation module."""

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_less

from dipy.sims.force import (
    bingham_to_sf,
    bingham_dictionary,
    validate_diffusivity_config,
    get_default_diffusivity_config,
)


def test_bingham_to_sf_shape():
    """Test that bingham_to_sf returns correct shape."""
    vertices = np.random.randn(100, 3)
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    major_axis = np.array([1, 0, 0])
    minor_axis = np.array([0, 1, 0])

    sf = bingham_to_sf(1.0, 10.0, 10.0, major_axis, minor_axis, vertices)

    assert sf.shape == (100,), f"Expected shape (100,), got {sf.shape}"


def test_bingham_to_sf_values():
    """Test that bingham_to_sf produces valid values."""
    vertices = np.random.randn(100, 3)
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    major_axis = np.array([1, 0, 0])
    minor_axis = np.array([0, 1, 0])

    sf = bingham_to_sf(1.0, 10.0, 10.0, major_axis, minor_axis, vertices)

    # All values should be positive and <= f0
    assert np.all(sf >= 0), "SF values should be non-negative"
    assert np.all(sf <= 1.0), "SF values should be <= f0"


def test_bingham_dictionary_structure():
    """Test bingham_dictionary returns correct structure."""
    sphere = np.random.randn(10, 3)
    sphere = sphere / np.linalg.norm(sphere, axis=1, keepdims=True)
    odi_list = np.array([0.1, 0.2, 0.3])

    result = bingham_dictionary(sphere, odi_list)

    assert isinstance(result, dict)
    assert len(result) == 10
    for i in range(10):
        assert i in result
        assert isinstance(result[i], dict)
        for odi in odi_list:
            assert odi in result[i]
