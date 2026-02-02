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


def test_validate_diffusivity_config_valid():
    """Test validation of valid diffusivity config."""
    config = get_default_diffusivity_config()
    assert validate_diffusivity_config(config) is True


def test_validate_diffusivity_config_missing_key():
    """Test validation fails for missing keys."""
    config = {"wm_d_par_range": (2e-3, 3e-3)}

    try:
        validate_diffusivity_config(config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Missing required key" in str(e)


def test_validate_diffusivity_config_invalid_range():
    """Test validation fails for invalid range."""
    config = get_default_diffusivity_config()
    config["wm_d_par_range"] = (3e-3, 2e-3)  # min > max

    try:
        validate_diffusivity_config(config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "min must be <= max" in str(e)


def test_get_default_diffusivity_config():
    """Test default config has all required keys."""
    config = get_default_diffusivity_config()

    assert "wm_d_par_range" in config
    assert "wm_d_perp_range" in config
    assert "gm_d_iso_range" in config
    assert "csf_d" in config

    # Check reasonable values
    assert config["csf_d"] > 0
    assert config["wm_d_par_range"][0] > 0
    assert config["wm_d_par_range"][1] >= config["wm_d_par_range"][0]
