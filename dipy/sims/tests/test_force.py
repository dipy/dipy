"""Tests for FORCE simulation module."""

import numpy as np
import pytest

from dipy.sims.force import (
    dispersion_lut,
    get_default_diffusivity_config,
    smallest_shell_bval,
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
    with pytest.raises(ValueError, match="Missing required key"):
        validate_diffusivity_config(config)


def test_validate_diffusivity_config_invalid_range():
    """Test validation fails for invalid range."""
    config = get_default_diffusivity_config()
    config["wm_d_par_range"] = (3e-3, 2e-3)  # min > max
    with pytest.raises(ValueError, match="min must be <= max"):
        validate_diffusivity_config(config)


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
    """Test smallest shell b-value finding for n=1 (default)."""
    bvals = np.array([0, 0, 1000, 1000, 2000, 2000, 3000])

    min_shell, mask = smallest_shell_bval(bvals)

    assert len(min_shell) == 1
    assert min_shell[0] == 1000
    assert mask.sum() == 2
    assert mask[2] and mask[3]
    # Other shells must NOT be in the mask
    assert not mask[4] and not mask[5] and not mask[6]


def test_smallest_shell_bval_n2():
    """Test smallest_shell_bval returns two smallest shells with n=2."""
    bvals = np.array([0, 0, 1000, 1000, 2000, 2000, 3000, 3000])

    min_shells, mask = smallest_shell_bval(bvals, n=2)

    assert len(min_shells) == 2
    assert min_shells[0] == 1000
    assert min_shells[1] == 2000
    # b0s excluded from shell mask
    assert not mask[0] and not mask[1]
    # Both shells selected
    assert mask[2] and mask[3]  # b=1000
    assert mask[4] and mask[5]  # b=2000
    # Largest shell excluded
    assert not mask[6] and not mask[7]


def test_smallest_shell_bval_n_too_large():
    """Test smallest_shell_bval raises when n exceeds available shells."""
    bvals = np.array([0, 1000, 2000])
    with pytest.raises(ValueError, match="unique shells found"):
        smallest_shell_bval(bvals, n=3)


def test_smallest_shell_bval_no_nonzero():
    """Test smallest_shell_bval raises for all-zero bvals."""
    bvals = np.array([0, 0, 10, 20, 30])  # all below threshold
    with pytest.raises(ValueError, match="No non-b0 volumes"):
        smallest_shell_bval(bvals)


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


def _make_gtab(shells):
    """Build a minimal GradientTable with 6 isotropic directions per shell.

    Parameters
    ----------
    shells : list of int
        Non-zero b-values (one shell per entry). Two b0 volumes are prepended.
    """
    from dipy.core.gradients import gradient_table

    # 6 directions on a hemisphere (roughly isotropic)
    dirs = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        dtype=float,
    )
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    bvals = [0, 0]
    bvecs = [[0, 0, 0], [0, 0, 0]]
    for b in shells:
        bvals.extend([b] * 6)
        bvecs.extend(dirs.tolist())

    return gradient_table(np.array(bvals, dtype=float), bvecs=np.array(bvecs))


def test_generate_force_simulations_compute_dti():
    """generate_force_simulations with compute_dti=True returns FA/MD/RD."""
    from dipy.sims.force import generate_force_simulations

    gtab = _make_gtab([1000])
    sims = generate_force_simulations(
        gtab,
        num_simulations=20,
        batch_size=20,
        num_cpus=1,
        compute_dti=True,
        compute_dki=False,
        verbose=False,
    )

    for key in ("fa", "md", "rd"):
        assert key in sims, f"Key '{key}' missing from simulations"
        arr = sims[key]
        assert arr.shape == (20,), f"Expected shape (20,) for '{key}', got {arr.shape}"
        assert np.any(arr != 0), f"'{key}' is all-zeros – DTI fitting appears skipped"

    # DKI keys must NOT be present
    for key in ("ak", "rk", "mk", "kfa"):
        assert key not in sims, f"DKI key '{key}' should be absent"


def test_generate_force_simulations_compute_dki():
    """generate_force_simulations with compute_dki=True returns AK/RK/MK/KFA."""
    from dipy.sims.force import generate_force_simulations

    # DKI requires at least 2 non-zero shells
    gtab = _make_gtab([1000, 2000])
    sims = generate_force_simulations(
        gtab,
        num_simulations=20,
        batch_size=20,
        num_cpus=1,
        compute_dti=True,
        compute_dki=True,
        verbose=False,
    )

    dki_keys = ("ak", "rk", "mk", "kfa")
    for key in dki_keys:
        assert key in sims, f"Key '{key}' missing from simulations"
        arr = sims[key]
        assert arr.shape == (20,), f"Expected shape (20,) for '{key}', got {arr.shape}"
        assert np.any(arr != 0), f"'{key}' is all-zeros – DKI fitting appears skipped"

    # DTI keys also present when compute_dti=True
    for key in ("fa", "md", "rd"):
        assert key in sims, f"DTI key '{key}' missing"


def test_generate_force_simulations_no_dti_no_dki():
    """generate_force_simulations with both flags False omits metric keys."""
    from dipy.sims.force import generate_force_simulations

    gtab = _make_gtab([1000])
    sims = generate_force_simulations(
        gtab,
        num_simulations=10,
        batch_size=10,
        num_cpus=1,
        compute_dti=False,
        compute_dki=False,
        verbose=False,
    )

    for key in ("fa", "md", "rd", "ak", "rk", "mk", "kfa"):
        if key in ("fa", "md", "rd"):
            # These keys exist but should be all zeros (initialised to zeros,
            # never filled when compute_dti=False)
            assert key in sims
            assert np.all(sims[key] == 0), f"'{key}' should be zero when DTI disabled"
        else:
            assert key not in sims, f"DKI key '{key}' should be absent"
