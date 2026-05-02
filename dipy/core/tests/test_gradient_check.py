"""Tests for :mod:`dipy.core.gradient_check`.

Verifies that the Aganj-2018 fiber-continuity criterion correctly identifies
the (permutation, flip) configuration needed to bring a corrupted gradient
table back into agreement with the image coordinate frame.

The data-driven tests load the Stanford HARDI dataset (or skip if it is not
available in the user's ``~/.dipy`` cache so the unit test does not perform
network IO under CI). Multiple corruption configurations are applied and the
algorithm is expected to recover the inverse transform in every case.
"""

import os

import numpy as np
import numpy.testing as npt
import pytest

from dipy.core.gradient_check import (
    FLIPS,
    PERMUTATIONS,
    apply_transform,
    check_gradient_table,
    correct_bvecs,
    correct_gradient_table,
    fiber_continuity_error,
    transform_label,
)
from dipy.core.gradients import gradient_table


def _assert_bvecs_equivalent(actual, expected, atol=1e-12):
    """Two b-vector arrays are equivalent under antipodal symmetry of the ODF.

    The 24-element canonical set used by the verifier excludes the
    "flip-all-three-axes" transforms (they are equivalent to no flip thanks
    to ``psi(-n) = psi(n)``). Consequently, when the input gradient table is
    transformed by a corruption ``C``, the recovered transform ``T_rec`` may
    pick either ``A o C^-1`` or ``-A o C^-1`` -- whichever lies in the
    canonical set -- where ``A`` is the algorithm's verdict on the
    uncorrupted data. Both are equally valid corrections; they differ only
    by a global sign flip of every b-vector. This helper accepts either.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    err_pos = np.max(np.abs(actual - expected))
    err_neg = np.max(np.abs(actual + expected))
    if min(err_pos, err_neg) > atol:
        raise AssertionError(
            f"b-vectors differ from expected by more than {atol}: "
            f"max|a-e|={err_pos:.3e}, max|a+e|={err_neg:.3e}"
        )


# ---------------------------------------------------------------------------
# Pure-algebra tests (no data, fast).
# ---------------------------------------------------------------------------


def test_apply_transform_identity():
    rng = np.random.default_rng(0)
    v = rng.standard_normal((10, 3))
    npt.assert_allclose(apply_transform(v, (0, 1, 2), (1, 1, 1)), v)


def test_apply_transform_swap_xy_flip_y():
    v = np.array([[1.0, 2.0, 3.0]])
    out = apply_transform(v, (1, 0, 2), (1, -1, 1))
    npt.assert_allclose(out, [[2.0, -1.0, 3.0]])


def test_correct_bvecs_swap_xz():
    bvecs = np.array(
        [
            [0.1, 0.2, 0.3],
            [-0.4, 0.5, -0.6],
            [0.0, 0.0, 1.0],
        ]
    )
    out = correct_bvecs(bvecs, (2, 1, 0), (1, 1, 1))
    npt.assert_allclose(out[:, 0], bvecs[:, 2])
    npt.assert_allclose(out[:, 1], bvecs[:, 1])
    npt.assert_allclose(out[:, 2], bvecs[:, 0])


def test_correct_bvecs_bad_shape():
    npt.assert_raises(ValueError, correct_bvecs, np.zeros(3), (0, 1, 2), (1, 1, 1))
    npt.assert_raises(ValueError, correct_bvecs, np.zeros((4, 2)), (0, 1, 2), (1, 1, 1))


def test_canonical_set_has_24_distinct_transforms():
    # Verify that the 6 perms x 4 flips really yield 24 distinct linear maps.
    seen = set()
    e = np.eye(3)
    for perm in PERMUTATIONS:
        for flip in FLIPS:
            mat = apply_transform(e, perm, flip)
            seen.add(tuple(mat.flatten().tolist()))
    npt.assert_equal(len(seen), 24)


def test_transform_label_strings():
    s = transform_label((0, 1, 2), (1, 1, 1))
    assert "X Y Z" in s and "no flip" in s
    s = transform_label((1, 0, 2), (1, -1, 1))
    assert "Y X Z" in s and "flip y" in s


# ---------------------------------------------------------------------------
# Data-driven tests on Stanford HARDI.
# ---------------------------------------------------------------------------


def _stanford_hardi_loaded():
    """Return True if Stanford HARDI is already cached locally."""
    home = os.path.join(os.path.expanduser("~"), ".dipy", "stanford_hardi")
    needed = ["HARDI150.bval", "HARDI150.bvec", "HARDI150.nii.gz"]
    return all(os.path.exists(os.path.join(home, f)) for f in needed)


def _wm_mask_for(
    data, gtab, brain_mask, sh_order_max=4, adc_threshold=0.0015, gfa_threshold=0.4
):
    """Mean-ADC + GFA white-matter mask, restricted to ``brain_mask``."""
    from dipy.reconst.shm import CsaOdfModel

    csa = CsaOdfModel(gtab, sh_order_max=sh_order_max)
    fit = csa.fit(data, mask=brain_mask)
    gfa = np.nan_to_num(fit.gfa, nan=0.0, posinf=0.0, neginf=0.0)

    b0_mask = gtab.b0s_mask
    s0 = data[..., b0_mask].mean(axis=-1)
    sd = data[..., ~b0_mask]
    bvals_dwi = gtab.bvals[~b0_mask]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(s0[..., None] > 0, sd / s0[..., None], 0.0)
        ratio = np.clip(ratio, 1e-8, None)
        adc_per_dir = -np.log(ratio) / bvals_dwi[None, None, None, :]
    mean_adc = np.nan_to_num(adc_per_dir.mean(axis=-1), nan=np.inf)

    return (
        brain_mask & (mean_adc < adc_threshold) & (mean_adc > 0) & (gfa > gfa_threshold)
    )


@pytest.fixture(scope="module")
def stanford_hardi_cropped():
    """Load Stanford HARDI cropped tightly around the brain to keep tests fast."""
    if not _stanford_hardi_loaded():
        pytest.skip(
            "Stanford HARDI not present in ~/.dipy. "
            "Run `dipy.data.fetch_stanford_hardi()` first."
        )

    from dipy.io.gradients import read_bvals_bvecs
    from dipy.io.image import load_nifti
    from dipy.segment.mask import median_otsu

    home = os.path.join(os.path.expanduser("~"), ".dipy", "stanford_hardi")
    data, _ = load_nifti(os.path.join(home, "HARDI150.nii.gz"))
    data = data.astype(np.float32)
    bvals, bvecs = read_bvals_bvecs(
        os.path.join(home, "HARDI150.bval"),
        os.path.join(home, "HARDI150.bvec"),
    )
    gtab = gradient_table(bvals, bvecs=bvecs)

    # The Stanford HARDI fetch does not ship a brain mask; compute one with
    # median_otsu over a representative DWI volume range.
    _, brain = median_otsu(data, vol_idx=range(10, 50), median_radius=4, numpass=4)

    # Crop tightly to the brain so the spatial-gradient computation is cheap.
    coords = np.where(brain)
    sx = slice(coords[0].min(), coords[0].max() + 1)
    sy = slice(coords[1].min(), coords[1].max() + 1)
    sz = slice(coords[2].min(), coords[2].max() + 1)
    data = data[sx, sy, sz]
    brain = brain[sx, sy, sz]

    wm_mask = _wm_mask_for(data, gtab, brain)
    return data, gtab, wm_mask


@pytest.mark.parametrize(
    "corrupt_perm, corrupt_flip",
    [
        ((0, 1, 2), (-1, 1, 1)),  # flip x
        ((0, 1, 2), (1, -1, 1)),  # flip y
        ((1, 0, 2), (1, -1, 1)),  # swap x,y then flip y (Aarhus-style)
        ((2, 1, 0), (1, 1, 1)),  # swap x,z (no flip)
        ((1, 2, 0), (1, 1, -1)),  # cyclic perm + flip z
    ],
)
def test_recover_corruption_stanford_hardi(
    stanford_hardi_cropped, corrupt_perm, corrupt_flip
):
    """Algorithm must recover the inverse of every corruption applied to the bvecs."""
    data, gtab, wm_mask = stanford_hardi_cropped

    # Establish the algorithm's verdict on the *original* gradient table; this
    # is the canonical reference frame the algorithm has converged on for this
    # dataset (with Stanford HARDI it is the identity transform).
    perm0, flip0 = check_gradient_table(data, gtab, mask=wm_mask, sh_order_max=4)
    canonical_bvecs = correct_bvecs(gtab.bvecs, perm0, flip0)

    # Corrupt the gradient table.
    bvecs_corr = apply_transform(gtab.bvecs, corrupt_perm, corrupt_flip)
    gtab_corr = gradient_table(
        gtab.bvals, bvecs=bvecs_corr, b0_threshold=gtab.b0_threshold
    )

    # Run the verifier on the corrupted gradient table.
    perm_rec, flip_rec = check_gradient_table(
        data, gtab_corr, mask=wm_mask, sh_order_max=4
    )

    # Composing the recovered transform with the corrupted bvecs must
    # reproduce the canonical bvecs (i.e., undo the corruption modulo any
    # systematic transform the algorithm picks for the dataset).
    recovered_bvecs = correct_bvecs(bvecs_corr, perm_rec, flip_rec)
    _assert_bvecs_equivalent(recovered_bvecs, canonical_bvecs)


def test_correct_gradient_table_returns_aligned_table(stanford_hardi_cropped):
    """``correct_gradient_table`` should hand back a usable, corrected GradientTable."""
    data, gtab, wm_mask = stanford_hardi_cropped

    # Corrupt with [Z Y X], flip y.
    corrupt_perm = (2, 1, 0)
    corrupt_flip = (1, -1, 1)
    bvecs_corr = apply_transform(gtab.bvecs, corrupt_perm, corrupt_flip)
    gtab_corr = gradient_table(
        gtab.bvals, bvecs=bvecs_corr, b0_threshold=gtab.b0_threshold
    )

    gtab_fixed, perm, flip = correct_gradient_table(
        data, gtab_corr, mask=wm_mask, sh_order_max=4
    )

    # b-values are untouched.
    npt.assert_allclose(gtab_fixed.bvals, gtab.bvals)

    # The corrected table should map back to the canonical bvecs.
    perm0, flip0 = check_gradient_table(data, gtab, mask=wm_mask, sh_order_max=4)
    canonical = correct_bvecs(gtab.bvecs, perm0, flip0)
    _assert_bvecs_equivalent(gtab_fixed.bvecs, canonical)


def test_fiber_continuity_error_returns_24_entries(stanford_hardi_cropped):
    data, gtab, wm_mask = stanford_hardi_cropped
    errors = fiber_continuity_error(data, gtab, mask=wm_mask, sh_order_max=4)
    npt.assert_equal(len(errors), 24)
    for key, val in errors.items():
        assert isinstance(key, tuple) and len(key) == 2
        assert val >= 0.0


def test_check_gradient_table_validates_inputs():
    bvals = np.array([0, 1000, 1000, 1000])
    bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    gtab = gradient_table(bvals, bvecs=bvecs)
    npt.assert_raises(ValueError, check_gradient_table, np.zeros((3, 4, 5)), gtab)


# ---------------------------------------------------------------------------
# Optional Sherbrooke 3-shell replication: per-shell single-shell verification.
# ---------------------------------------------------------------------------


def _sherbrooke_loaded():
    home = os.path.join(os.path.expanduser("~"), ".dipy", "sherbrooke_3shell")
    needed = ["HARDI193.bval", "HARDI193.bvec", "HARDI193.nii.gz"]
    return all(os.path.exists(os.path.join(home, f)) for f in needed)


@pytest.mark.parametrize("shell_b", [1000, 2000, 3500])
def test_sherbrooke_single_shell_recovers_swap_xy_flip_y(shell_b):
    """Replicate the Aarhus-style corruption on each Sherbrooke shell."""
    if not _sherbrooke_loaded():
        pytest.skip("Sherbrooke 3-shell dataset not present in ~/.dipy.")

    from dipy.data import read_sherbrooke_3shell

    img, gtab_full = read_sherbrooke_3shell()
    data_full = img.get_fdata().astype(np.float32)

    # Extract a single shell (b=0 + b=shell_b).
    keep = (gtab_full.bvals < 50) | (np.abs(gtab_full.bvals - shell_b) < 50)
    data = data_full[..., keep]
    bvals = gtab_full.bvals[keep]
    bvecs = gtab_full.bvecs[keep]
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)

    # Crop to a brain-containing region using a coarse signal-based mask.
    s0 = data[..., gtab.b0s_mask].mean(axis=-1)
    coarse = s0 > (0.1 * np.percentile(s0[s0 > 0], 95))
    coords = np.where(coarse)
    if coords[0].size == 0:
        pytest.skip("No signal in Sherbrooke volume; cannot run replication.")
    sx = slice(coords[0].min(), coords[0].max() + 1)
    sy = slice(coords[1].min(), coords[1].max() + 1)
    sz = slice(coords[2].min(), coords[2].max() + 1)
    data = data[sx, sy, sz]
    coarse = coarse[sx, sy, sz]

    wm_mask = _wm_mask_for(
        data, gtab, coarse, sh_order_max=4, adc_threshold=0.0020, gfa_threshold=0.3
    )
    if wm_mask.sum() < 200:
        pytest.skip(f"WM mask too small for shell b={shell_b}.")

    # Establish algorithm verdict on original.
    perm0, flip0 = check_gradient_table(data, gtab, mask=wm_mask, sh_order_max=4)
    canonical = correct_bvecs(gtab.bvecs, perm0, flip0)

    # Apply the Aarhus-style corruption: swap x/y, then flip y.
    corrupt_perm = (1, 0, 2)
    corrupt_flip = (1, -1, 1)
    bvecs_corr = apply_transform(gtab.bvecs, corrupt_perm, corrupt_flip)
    gtab_corr = gradient_table(
        gtab.bvals, bvecs=bvecs_corr, b0_threshold=gtab.b0_threshold
    )

    perm_rec, flip_rec = check_gradient_table(
        data, gtab_corr, mask=wm_mask, sh_order_max=4
    )
    recovered = correct_bvecs(bvecs_corr, perm_rec, flip_rec)
    _assert_bvecs_equivalent(recovered, canonical)
