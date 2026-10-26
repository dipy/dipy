import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates

from dipy.align.metrics import CCMetric
from dipy.align.sdc import PhaseEncodingSyN, syn_sdc

# The phantoms are a third of the size of a typical DWI, so the SyN updates
# are smoothed proportionally less than by default.
SIGMA = 3.0


def _phantom(shape=(44, 52, 40), seed=0):
    """Ellipsoidal brain with T2-like contrast and asymmetric features."""
    rng = np.random.default_rng(seed)
    grid = np.mgrid[[slice(0, n) for n in shape]].astype(float)
    center = np.array(shape)[:, None, None, None] / 2
    radius = np.array(shape)[:, None, None, None] * 0.38
    r = np.sqrt((((grid - center) / radius) ** 2).sum(0))
    brain = r < 1
    texture = gaussian_filter(rng.normal(size=shape), 2.5)
    t2 = 50 + 12 * texture / texture.std()
    t2[r > 0.85] = 100
    ventricles = (grid - center) ** 2 / np.array([30, 20, 30])[:, None, None, None]
    t2[np.sqrt(ventricles.sum(0)) < 1] = 100
    blob = np.array([14, 18, 26])[:, None, None, None]
    t2[np.sqrt(((grid - blob) ** 2).sum(0)) < 4] = 100
    t2 = gaussian_filter(np.where(brain, t2, 0), 0.7)
    t2[~brain] = 0
    # Skull-stripped T1w with the inverse contrast of the T2w.
    t1 = np.where(brain, 110 - t2, 0)
    return grid, brain, t1, t2


def _distort(image, grid, pe_axis, displacement):
    """Image whose correction is ``corrected(x) = out(x + u(x))``."""
    coords = grid.copy()
    coords[pe_axis] -= displacement
    return map_coordinates(image, coords, order=1)


def _bump(grid, brain, amplitude=2.5):
    u = amplitude * np.exp(
        -(
            ((grid[0] - 22) / 9) ** 2
            + ((grid[1] - 38) / 8) ** 2
            + ((grid[2] - 12) / 8) ** 2
        )
    )
    return gaussian_filter(u * brain, 2)


def _oblique_affine():
    theta = np.deg2rad(12)
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    affine = np.diag([2.0, 2.0, 2.5, 1.0])
    affine[:3, :3] = rot @ affine[:3, :3]
    return affine


def test_phase_encoding_syn_is_restricted():
    grid, brain, _, t2 = _phantom()
    affine = _oblique_affine()
    pe_axis = 1
    moving = _distort(t2, grid, pe_axis, _bump(grid, brain))
    sdr = PhaseEncodingSyN(
        CCMetric(3, sigma_diff=3.0, radius=4),
        affine[:3, pe_axis],
        level_iters=[20, 20, 10],
    )
    mapping = sdr.optimize(
        t2, moving, static_grid2world=affine, moving_grid2world=affine
    )
    direction = affine[:3, pe_axis] / np.linalg.norm(affine[:3, pe_axis])
    for field in (mapping.get_forward_field(), mapping.get_backward_field()):
        field = np.asarray(field)
        along = field @ direction
        across = field - along[..., None] * direction
        assert np.abs(along).max() > 0.5
        assert np.abs(across).max() < 1e-4


def test_phase_encoding_syn_invalid_direction():
    with pytest.raises(ValueError, match="nonzero 3-vector"):
        PhaseEncodingSyN(CCMetric(3), [0, 0, 0])
    with pytest.raises(ValueError, match="nonzero 3-vector"):
        PhaseEncodingSyN(CCMetric(3), [1, 0])


@pytest.mark.parametrize("contrast", ["t2", "t1"])
@pytest.mark.parametrize("oblique", [False, True])
def test_syn_sdc_recovers_displacement(contrast, oblique):
    grid, brain, t1, t2 = _phantom()
    affine = _oblique_affine() if oblique else np.diag([2.0, 2.0, 2.5, 1.0])
    pe_axis = 1
    u = _bump(grid, brain)
    b0 = _distort(t2, grid, pe_axis, u)
    anat = t2 if contrast == "t2" else t1
    corrected, field = syn_sdc(
        b0,
        affine,
        anat,
        affine,
        pe_axis=pe_axis,
        anat_contrast=contrast,
        n_outer=1,
        sigma_diff=SIGMA,
        return_field=True,
    )
    assert_equal(corrected.shape, b0.shape)
    assert_equal(field.shape, b0.shape)
    err = field - u
    err -= np.median(err[brain])
    err_before = np.abs(u - np.median(u[brain]))[brain].mean()
    err_after = np.abs(err[brain]).mean()
    assert err_after < 0.5 * err_before
    assert np.corrcoef(field[brain], u[brain])[0, 1] > 0.9


def test_syn_sdc_4d_and_mapping():
    grid, brain, _, t2 = _phantom()
    affine = np.diag([2.0, 2.0, 2.5, 1.0])
    u = _bump(grid, brain)
    b0 = _distort(t2, grid, 1, u)
    dwi = np.stack([0.5 * b0, b0, 0.3 * b0], axis=-1).astype(np.float32)
    corrected, field, mapping = syn_sdc(
        dwi,
        affine,
        t2,
        affine,
        pe_axis="j",
        anat_contrast="T2",
        b0_index=1,
        n_outer=1,
        sigma_diff=SIGMA,
        return_field=True,
        return_mapping=True,
    )
    assert_equal(corrected.shape, dwi.shape)
    assert_equal(corrected.dtype, dwi.dtype)
    # every volume gets the same correction
    assert_array_almost_equal(corrected[..., 0], 0.5 * corrected[..., 1], decimal=2)
    assert_array_almost_equal(
        mapping.transform(dwi[..., 2]), corrected[..., 2], decimal=3
    )
    # the backward field undoes the forward one
    back = mapping.transform_inverse(corrected[..., 1])
    inner = brain & (np.abs(u) > 0.5)
    rel = np.abs(back - b0)[inner].mean() / b0[inner].mean()
    assert rel < 0.05


def test_syn_sdc_invalid_inputs():
    vol = np.zeros((10, 10, 10))
    vol[3:7, 3:7, 3:7] = 1
    aff = np.eye(4)
    with pytest.raises(ValueError, match="pe_axis"):
        syn_sdc(vol, aff, vol, aff, pe_axis=3)
    with pytest.raises(ValueError, match="anat_contrast"):
        syn_sdc(vol, aff, vol, aff, pe_axis=0, anat_contrast="flair")
    with pytest.raises(ValueError, match="dwi must be 3D or 4D"):
        syn_sdc(vol[0], aff, vol, aff, pe_axis=0)
    with pytest.raises(ValueError, match="anat must be 3D"):
        syn_sdc(vol, aff, vol[..., None], aff, pe_axis=0)
    with pytest.raises(ValueError, match="b0_index"):
        syn_sdc(vol[..., None], aff, vol, aff, pe_axis=0, b0_index=1)
    with pytest.raises(ValueError, match="mask is empty"):
        syn_sdc(vol, aff, np.zeros_like(vol), aff, pe_axis=0)
    with pytest.raises(ValueError, match="n_outer"):
        syn_sdc(vol, aff, vol, aff, pe_axis=0, n_outer=0)


def test_syn_sdc_truncated_fov():
    # Slices below the EPI field of view are zero in the b0 but contain brain
    # in the anatomical image; they must not drive the registration.
    grid, brain, t1, t2 = _phantom()
    affine = np.diag([2.0, 2.0, 2.5, 1.0])
    u = _bump(grid, brain)
    b0 = _distort(t2, grid, 1, u)
    b0[..., :12] = 0
    _, field = syn_sdc(
        b0,
        affine,
        t1,
        affine,
        pe_axis=1,
        n_outer=1,
        sigma_diff=SIGMA,
        return_field=True,
    )
    acquired = brain & (grid[2] >= 14)
    err = field - u
    err -= np.median(err[acquired])
    err_before = np.abs(u - np.median(u[acquired]))[acquired].mean()
    assert np.abs(err[acquired]).mean() < 0.75 * err_before
    assert np.abs(field[brain]).max() < 2 * u.max()
