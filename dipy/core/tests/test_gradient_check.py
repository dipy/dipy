"""Tests for :mod:`dipy.core.gradient_check`."""

import numpy as np
import numpy.testing as npt
import pytest
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from dipy.core.gradient_check import (
    apply_transform,
    check_gradient_table,
    correct_bvecs,
    correct_gradient_table,
    fiber_continuity_error,
    transform_label,
)
from dipy.core.gradients import gradient_table
from dipy.core.sphere import HemiSphere, disperse_charges
from dipy.data import default_sphere
from dipy.reconst.shm import CsaOdfModel

# CsaOdfModel uses the legacy descoteaux07 SH basis; silence the per-call
# PendingDeprecationWarning under DIPY_WERRORS=1.
pytestmark = pytest.mark.filterwarnings(
    "ignore:The legacy descoteaux07 SH basis:PendingDeprecationWarning"
)

CORRUPTIONS = {
    "flip_x": ((0, 1, 2), (-1, 1, 1)),
    "flip_y": ((0, 1, 2), (1, -1, 1)),
    "flip_z": ((0, 1, 2), (1, 1, -1)),
    "swap_xy": ((1, 0, 2), (1, 1, 1)),
    "swap_xy_flip_y": ((1, 0, 2), (1, -1, 1)),
    "cyclic_flip_z": ((1, 2, 0), (1, 1, -1)),
}


def _same_up_to_sign(actual, expected):
    """b-vectors are defined up to a global sign (antipodal ODF symmetry)."""
    err = min(np.abs(actual - expected).max(), np.abs(actual + expected).max())
    return err < 1e-10


def _make_phantom(voxel_size=(2.0, 2.0, 2.0), nb_tubes=12, seed=0):
    """Noisy DWI of curved tubes whose fiber direction follows their tangent.

    Returns the data, the gradient table, the brain mask and the voxel size.
    """
    rng = np.random.default_rng(seed)
    extent = np.array([60.0, 60.0, 30.0])
    shape = tuple(np.round(extent / voxel_size).astype(int))
    axes = [(np.arange(n) + 0.5) * v for n, v in zip(shape, voxel_size)]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    points, tangents = [], []
    for _ in range(nb_tubes):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        u = np.cross(direction, rng.normal(size=3))
        u /= np.linalg.norm(u)
        w = np.cross(direction, u)
        s = np.arange(0.0, 70.0, 0.5)
        amp = rng.uniform(4, 10, 2)
        freq = rng.uniform(0.05, 0.12, 2)
        phase = rng.uniform(0, 2 * np.pi, 2)
        start = extent / 2 + rng.normal(size=3) * extent * 0.15 - direction * 35
        path = (
            start
            + s[:, None] * direction
            + (amp[0] * np.sin(freq[0] * s + phase[0]))[:, None] * u
            + (amp[1] * np.sin(freq[1] * s + phase[1]))[:, None] * w
        )
        tangent = np.gradient(path, axis=0)
        points.append(path)
        tangents.append(tangent / np.linalg.norm(tangent, axis=1, keepdims=True))
    points, tangents = np.concatenate(points), np.concatenate(tangents)

    distance, nearest = cKDTree(points).query(centers.reshape(-1, 3))
    in_tube = (distance < 3.6).reshape(shape)
    fibers = np.zeros(shape + (3,))
    fibers[in_tube] = tangents[nearest][in_tube.reshape(-1)]
    brain = np.linalg.norm((centers - extent / 2) / (extent / 2 * 0.98), axis=-1) < 1

    hemisphere = HemiSphere(
        theta=np.arccos(rng.uniform(0, 1, 30)), phi=rng.uniform(0, 2 * np.pi, 30)
    )
    hemisphere, _ = disperse_charges(hemisphere, 2000)
    bvals = np.concatenate([np.zeros(4), np.full(30, 1000.0)])
    bvecs = np.concatenate([np.zeros((4, 3)), hemisphere.vertices])

    cos = np.einsum("...c,nc->...n", fibers, bvecs)
    adc = np.where(in_tube[..., None], 0.3e-3 + 1.4e-3 * cos**2, 0.8e-3)
    clean = 1000.0 * np.exp(-bvals * adc) * brain[..., None]
    noise = rng.normal(0, 1000.0 / 25, (2,) + clean.shape)
    data = np.hypot(clean + noise[0], noise[1]).astype(np.float32)
    return data, gradient_table(bvals, bvecs=bvecs), brain


@pytest.fixture(scope="module")
def phantom():
    return _make_phantom()


def test_apply_transform_identity():
    rng = np.random.default_rng(0)
    v = rng.standard_normal((10, 3))
    npt.assert_allclose(apply_transform(v, (0, 1, 2), (1, 1, 1)), v)


def test_apply_transform_swap_xy_flip_y():
    v = np.array([[1.0, 2.0, 3.0]])
    out = apply_transform(v, (1, 0, 2), (1, -1, 1))
    npt.assert_allclose(out, [[2.0, -1.0, 3.0]])


def test_correct_bvecs():
    bvecs = np.array([[0.1, 0.2, 0.3], [-0.4, 0.5, -0.6], [0.0, 0.0, 1.0]])
    out = correct_bvecs(bvecs, (2, 1, 0), (1, 1, 1))
    npt.assert_allclose(out, bvecs[:, ::-1])
    npt.assert_raises(ValueError, correct_bvecs, np.zeros(3), (0, 1, 2), (1, 1, 1))
    npt.assert_raises(ValueError, correct_bvecs, np.zeros((4, 2)), (0, 1, 2), (1, 1, 1))


def test_transform_label():
    label = transform_label((0, 1, 2), (1, 1, 1))
    assert "X Y Z" in label and "no flip" in label
    label = transform_label((1, 0, 2), (1, -1, 1))
    assert "Y X Z" in label and "flip y" in label


def test_fiber_continuity_error_keys(phantom):
    data, gtab, brain = phantom
    errors = fiber_continuity_error(data, gtab, brain_mask=brain)
    npt.assert_equal(len(errors), 24)
    npt.assert_equal(
        len({tuple(map(tuple, apply_transform(np.eye(3), *k))) for k in errors}), 24
    )
    assert all(value >= 0 for value in errors.values())


def test_fiber_continuity_error_matches_sphere_integration(phantom):
    # The accumulated-tensor score must equal the direct integration of
    # ``(T n . grad psi)^2`` over the sphere vertices of the ODF volume.
    data, gtab, brain = phantom
    mask = np.zeros(brain.shape, dtype=bool)
    mask[10:20, 8:22, 5:10] = True
    mask &= brain

    odf = CsaOdfModel(gtab, sh_order_max=4).fit(data, mask=brain).odf(default_sphere)
    gradient = np.stack(np.gradient(odf, axis=(0, 1, 2)), axis=-1)[mask]
    errors = fiber_continuity_error(data, gtab, mask=mask, brain_mask=brain)
    for (perm, flip), value in errors.items():
        rotated = apply_transform(default_sphere.vertices, perm, flip)
        expected = np.sum(np.einsum("vmd,md->vm", gradient, rotated) ** 2)
        npt.assert_allclose(value, expected, rtol=1e-3)


@pytest.mark.parametrize("name", CORRUPTIONS)
def test_check_gradient_table_recovers_corruption(phantom, name):
    data, gtab, brain = phantom
    npt.assert_equal(
        check_gradient_table(data, gtab, brain_mask=brain), ((0, 1, 2), (1, 1, 1))
    )

    bvecs = apply_transform(gtab.bvecs, *CORRUPTIONS[name])
    corrupted = gradient_table(gtab.bvals, bvecs=bvecs)
    perm, flip = check_gradient_table(data, corrupted, brain_mask=brain)
    assert _same_up_to_sign(correct_bvecs(bvecs, perm, flip), gtab.bvecs)


def test_check_gradient_table_default_masks(phantom):
    data, gtab, _ = phantom
    bvecs = apply_transform(gtab.bvecs, *CORRUPTIONS["swap_xy_flip_y"])
    corrupted = gradient_table(gtab.bvals, bvecs=bvecs)
    perm, flip = check_gradient_table(data, corrupted)
    assert _same_up_to_sign(correct_bvecs(bvecs, perm, flip), gtab.bvecs)


def test_fiber_continuity_error_ignores_brain_boundary(phantom):
    # Voxels next to the fitted region would need unfitted neighbors for the
    # spatial gradient, so they must not contribute to the score.
    data, gtab, brain = phantom
    whole = fiber_continuity_error(data, gtab, mask=brain, brain_mask=brain)
    interior = fiber_continuity_error(
        data, gtab, mask=ndi.binary_erosion(brain), brain_mask=brain
    )
    for key, value in whole.items():
        npt.assert_allclose(value, interior[key])


def test_check_gradient_table_anisotropic_voxels():
    voxel_size = (1.5, 1.5, 3.0)
    data, gtab, brain = _make_phantom(voxel_size=voxel_size, seed=1)
    bvecs = apply_transform(gtab.bvecs, *CORRUPTIONS["cyclic_flip_z"])
    corrupted = gradient_table(gtab.bvals, bvecs=bvecs)
    perm, flip = check_gradient_table(
        data, corrupted, brain_mask=brain, voxel_size=voxel_size
    )
    assert _same_up_to_sign(correct_bvecs(bvecs, perm, flip), gtab.bvecs)


def test_correct_gradient_table(phantom):
    data, gtab, brain = phantom
    same, perm, flip = correct_gradient_table(data, gtab, brain_mask=brain)
    assert same is gtab
    npt.assert_equal((perm, flip), ((0, 1, 2), (1, 1, 1)))

    bvecs = apply_transform(gtab.bvecs, *CORRUPTIONS["swap_xy_flip_y"])
    corrupted = gradient_table(gtab.bvals, bvecs=bvecs)
    fixed, _, _ = correct_gradient_table(data, corrupted, brain_mask=brain)
    npt.assert_allclose(fixed.bvals, gtab.bvals)
    assert _same_up_to_sign(fixed.bvecs, gtab.bvecs)


def test_fiber_continuity_error_invalid_input(phantom):
    data, gtab, brain = phantom
    npt.assert_raises(ValueError, fiber_continuity_error, data[..., 0], gtab)
    empty = np.zeros(brain.shape, dtype=bool)
    npt.assert_raises(
        ValueError, fiber_continuity_error, data, gtab, mask=empty, brain_mask=brain
    )
    npt.assert_raises(ValueError, fiber_continuity_error, data, gtab, gfa_threshold=1.1)
