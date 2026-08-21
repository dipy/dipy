"""Tests for FORCE simulation module."""

import numpy as np
import pytest

from dipy.sims.force import (
    DEFAULT_NUM_ODI_VALUES,
    DEFAULT_ODI_RANGE,
    dispersion_lut,
    get_default_diffusivity_config,
    resolve_num_odi_values,
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


def test_resolve_num_odi_values_autoscale():
    """None autoscales the ODI grid to keep sampling density constant."""
    # Default range resolves to the historical fixed grid (backward compatible).
    assert (
        resolve_num_odi_values(DEFAULT_ODI_RANGE, num_odi_values=None)
        == DEFAULT_NUM_ODI_VALUES
    )

    # Doubling the span roughly doubles the number of grid points.
    assert resolve_num_odi_values((0.01, 0.6), num_odi_values=None) == 19

    # Narrower span -> fewer points; wider -> more.
    n_narrow = resolve_num_odi_values((0.05, 0.15), num_odi_values=None)
    n_wide = resolve_num_odi_values((0.01, 0.9), num_odi_values=None)
    assert n_narrow < DEFAULT_NUM_ODI_VALUES < n_wide

    # A degenerate (zero-width) range still yields a valid grid (>= 2).
    assert resolve_num_odi_values((0.2, 0.2), num_odi_values=None) == 2


def test_resolve_num_odi_values_explicit_and_invalid():
    """An explicit count is passed through; counts < 2 are rejected."""
    # Explicit value is honored regardless of the range.
    assert resolve_num_odi_values((0.01, 0.9), num_odi_values=7) == 7
    assert resolve_num_odi_values(DEFAULT_ODI_RANGE, num_odi_values=3) == 3

    for bad in (1, 0, -5):
        with pytest.raises(ValueError, match="must be >= 2"):
            resolve_num_odi_values(DEFAULT_ODI_RANGE, num_odi_values=bad)


def test_generate_force_simulations_honors_odi_grid():
    """A resolve error for num_odi_values < 2 propagates through generation."""
    from dipy.sims.force import generate_force_simulations

    gtab = _make_gtab([1000])
    with pytest.raises(ValueError, match="must be >= 2"):
        generate_force_simulations(
            gtab,
            num_simulations=10,
            batch_size=10,
            num_cpus=1,
            num_odi_values=1,
            verbose=False,
        )


def _library_crossing_angles(sims, num_fibers):
    """Antipodal-symmetric angles between the labelled fibers of *num_fibers* voxels."""
    from dipy.data import default_sphere

    vertices = default_sphere.vertices
    angles = []
    for labels in sims["labels"][sims["num_fibers"] == num_fibers]:
        dirs = vertices[np.flatnonzero(labels == 1)]
        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                cos = np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0)
                angles.append(np.rad2deg(np.arccos(abs(cos))))
    return np.asarray(angles)


def test_generate_force_simulations_default_min_crossing_angles(monkeypatch):
    """By default the library holds no crossing below 30 (two) or 60 (three) degrees."""
    from dipy.sims.force import generate_force_simulations

    monkeypatch.setattr(
        "dipy.sims.force.init_worker", lambda *a, **k: np.random.seed(0)
    )
    gtab = _make_gtab([1000, 2000])
    sims = generate_force_simulations(
        gtab, num_simulations=300, batch_size=100, num_cpus=1, verbose=False
    )

    two = _library_crossing_angles(sims, 2)
    three = _library_crossing_angles(sims, 3)
    assert two.size and three.size
    assert two.min() >= 30.0
    assert three.min() >= 60.0


def test_generate_force_simulations_relaxed_min_crossing_angles(monkeypatch):
    """Lowering the limits lets shallow crossings into the library."""
    from dipy.sims.force import generate_force_simulations

    monkeypatch.setattr(
        "dipy.sims.force.init_worker", lambda *a, **k: np.random.seed(0)
    )
    gtab = _make_gtab([1000, 2000])
    sims = generate_force_simulations(
        gtab,
        num_simulations=300,
        batch_size=100,
        num_cpus=1,
        two_fiber_min_angle=0.0,
        three_fiber_min_angle=0.0,
        verbose=False,
    )

    two = _library_crossing_angles(sims, 2)
    three = _library_crossing_angles(sims, 3)
    assert two.size and three.size
    assert two.min() < 30.0
    assert three.min() < 60.0


@pytest.mark.parametrize(
    "kwargs",
    [{"two_fiber_min_angle": 90.0}, {"three_fiber_min_angle": -1.0}],
)
def test_generate_force_simulations_invalid_min_crossing_angle(kwargs):
    """Angles outside [0, 90) degrees are rejected."""
    from dipy.sims.force import generate_force_simulations

    gtab = _make_gtab([1000])
    with pytest.raises(ValueError, match="must be in .0, 90. degrees"):
        generate_force_simulations(
            gtab, num_simulations=10, batch_size=10, num_cpus=1, verbose=False, **kwargs
        )
