"""Tests for FORCE simulation module."""

import numpy as np

from dipy.sims.force import (
    dispersion_lut,
    get_default_diffusivity_config,
    validate_diffusivity_config,
)


def test_dispersion_lut_structure():
    """Test dispersion_lut returns correct structure."""
    sphere = np.random.randn(10, 3)
    sphere = sphere / np.linalg.norm(sphere, axis=1, keepdims=True)
    odi_list = np.array([0.1, 0.2, 0.3])

    result = dispersion_lut(sphere, odi_list)

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
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Missing required key" in str(e)


def test_validate_diffusivity_config_invalid_range():
    """Test validation fails for invalid range."""
    config = get_default_diffusivity_config()
    config["wm_d_par_range"] = (3e-3, 2e-3)  # min > max

    try:
        validate_diffusivity_config(config)
        raise AssertionError("Should have raised ValueError")
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


def test_smallest_shell_bval():
    """Test smallest shell b-value finding."""
    from dipy.sims.force import smallest_shell_bval

    bvals = np.array([0, 0, 1000, 1000, 2000, 2000, 3000])

    min_shell, mask = smallest_shell_bval(bvals)

    assert min_shell == 1000
    assert mask.sum() == 2
    assert mask[2] and mask[3]


def test_smallest_shell_bval_no_nonzero():
    """Test smallest_shell_bval raises for all-zero bvals."""
    from dipy.sims.force import smallest_shell_bval

    bvals = np.array([0, 0, 10, 20, 30])  # all below threshold

    try:
        smallest_shell_bval(bvals)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "No non-b0 volumes" in str(e)


def test_save_load_force_simulations(tmp_path):
    """Test saving and loading FORCE simulations."""
    from dipy.sims.force import load_force_simulations, save_force_simulations

    # Create test simulations
    test_sims = {
        "signals": np.random.randn(10, 100).astype(np.float32),
        "labels": np.random.randint(0, 2, (10, 50)).astype(np.uint8),
        "fa": np.random.rand(10).astype(np.float32),
    }

    output_path = tmp_path / "test_sims.npz"
    save_force_simulations(test_sims, str(output_path))

    # Load and verify
    loaded = load_force_simulations(str(output_path))

    assert set(loaded.keys()) == set(test_sims.keys())
    for key in test_sims:
        np.testing.assert_array_equal(loaded[key], test_sims[key])
