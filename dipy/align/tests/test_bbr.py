import numpy as np
import numpy.testing as npt
import pytest
from scipy.ndimage import gaussian_filter

from dipy.align.bbr import bbr_registration
from dipy.align.imaffine import AffineMap
from dipy.core.geometry import euler_matrix

STATIC_AFFINE = np.array(
    [[1.0, 0, 0, -32.0], [0, 1.0, 0, -36.0], [0, 0, 1.0, -28.0], [0, 0, 0, 1.0]]
)
MOVING_AFFINE = np.array(
    [[2.0, 0, 0, -40.0], [0, 2.0, 0, -44.0], [0, 0, 2.0, -36.0], [0, 0, 0, 1.0]]
)
# Maps the T1 world to the EPI world: 3, -2, 2.5 degrees and 2, -3, 1.5 mm.
TRUTH = euler_matrix(*np.deg2rad([3.0, -2.0, 2.5]))
TRUTH[:3, 3] = [2.0, -3.0, 1.5]


def _phantom(shape, affine, *, gm_brighter):
    """Nested ellipsoids: white matter inside gray matter inside CSF."""
    xyz = np.indices(shape).reshape(3, -1).T @ affine[:3, :3].T + affine[:3, 3]
    r = np.linalg.norm(xyz / [22.0, 28.0, 20.0], axis=1).reshape(shape)
    wm, gm, csf = r < 0.7, (r >= 0.7) & (r < 0.9), (r >= 0.9) & (r < 1.0)
    values = (60.0, 100.0, 140.0) if gm_brighter else (100.0, 60.0, 20.0)
    return values[0] * wm + values[1] * gm + values[2] * csf, wm.astype(float)


def _setup(*, gm_brighter=True):
    t1, wm = _phantom((64, 72, 56), STATIC_AFFINE, gm_brighter=False)
    epi_in_t1, _ = _phantom((64, 72, 56), STATIC_AFFINE, gm_brighter=gm_brighter)
    epi = AffineMap(
        np.linalg.inv(TRUTH),
        domain_grid_shape=(40, 44, 36),
        domain_grid2world=MOVING_AFFINE,
        codomain_grid_shape=t1.shape,
        codomain_grid2world=STATIC_AFFINE,
    ).transform(gaussian_filter(epi_in_t1, 0.7))
    epi += np.random.default_rng(0).normal(0, 2.0, epi.shape) * (epi > 0)
    return epi, t1, wm


def _error_mm(affine, truth, *, wm, static_affine):
    """Mean distance, over the white matter, between two transforms."""
    xyz = np.argwhere(wm > 0) @ static_affine[:3, :3].T + static_affine[:3, 3]
    return np.linalg.norm(
        xyz @ (affine - truth)[:3, :3].T + (affine - truth)[:3, 3], axis=1
    ).mean()


@pytest.mark.parametrize("case", ["wm_mask", "hmrf", "dropout", "wm_brighter"])
def test_bbr_recovers_rigid(case):
    epi, t1, wm = _setup(gm_brighter=case != "wm_brighter")
    kwargs = {"wm_mask": wm}
    if case == "hmrf":
        kwargs = {"static_mask": t1 > 0}
    elif case == "dropout":
        epi[:, :, :8] = 0
    elif case == "wm_brighter":
        kwargs["boundary_contrast"] = "wm_brighter"
    resampled, affine = bbr_registration(
        epi, t1, MOVING_AFFINE, STATIC_AFFINE, starting_affine=np.eye(4), **kwargs
    )
    assert _error_mm(np.eye(4), TRUTH, wm=wm, static_affine=STATIC_AFFINE) > 2.0
    assert _error_mm(affine, TRUTH, wm=wm, static_affine=STATIC_AFFINE) < 0.75
    assert resampled.shape == t1.shape
    npt.assert_allclose(np.linalg.det(affine[:3, :3]), 1.0, atol=1e-6)


def test_bbr_intensity_scale_and_contrast_sign():
    epi, t1, wm = _setup()
    kwargs = {"wm_mask": wm, "starting_affine": np.eye(4)}
    _, affine = bbr_registration(epi, t1, MOVING_AFFINE, STATIC_AFFINE, **kwargs)
    # The cost is a ratio, so the intensity scale does not matter.
    _, scaled = bbr_registration(epi * 1000, t1, MOVING_AFFINE, STATIC_AFFINE, **kwargs)
    assert _error_mm(scaled, affine, wm=wm, static_affine=STATIC_AFFINE) < 1e-3
    # With the wrong contrast the registration drifts away, or runs out of
    # boundary points with a usable signal.
    try:
        _, wrong = bbr_registration(
            epi,
            t1,
            MOVING_AFFINE,
            STATIC_AFFINE,
            boundary_contrast="wm_brighter",
            **kwargs,
        )
    except ValueError:
        return
    assert _error_mm(wrong, TRUTH, wm=wm, static_affine=STATIC_AFFINE) > 1.0


def test_bbr_scaling():
    epi, t1, wm = _setup()
    # A T1 grid with a wrong voxel size: the anatomy is 12% larger along x.
    static_affine = STATIC_AFFINE.copy()
    static_affine[0] *= 1.12
    target = TRUTH @ np.diag([1 / 1.12, 1.0, 1.0, 1.0])
    kwargs = {"wm_mask": wm, "starting_affine": np.eye(4)}
    _, rigid = bbr_registration(epi, t1, MOVING_AFFINE, static_affine, **kwargs)
    _, zoomed = bbr_registration(
        epi, t1, MOVING_AFFINE, static_affine, scaling=True, **kwargs
    )
    zoomed_err = _error_mm(zoomed, target, wm=wm, static_affine=static_affine)
    assert zoomed_err < 0.75
    assert (
        _error_mm(rigid, target, wm=wm, static_affine=static_affine) > 1.5 * zoomed_err
    )
    smallest = np.linalg.svd(zoomed[:3, :3], compute_uv=False).min()
    npt.assert_allclose(smallest, 1 / 1.12, atol=0.03)


@pytest.mark.parametrize(
    "change",
    [
        {"moving": "4d"},
        {"boundary_contrast": "other"},
        {"wm_mask": "empty"},
        {"wm_mask": None},
        {"starting_affine": "far"},
    ],
)
def test_bbr_errors(change):
    epi, t1, wm = _setup()
    far = np.eye(4)
    far[:3, 3] = 500.0
    replace = {"4d": np.stack([epi, epi], -1), "empty": np.zeros_like(wm), "far": far}
    kwargs = {"wm_mask": wm, "starting_affine": np.eye(4)}
    kwargs.update({k: replace.get(v, v) for k, v in change.items()})
    moving = kwargs.pop("moving", epi)
    with pytest.raises(ValueError):
        bbr_registration(moving, t1, MOVING_AFFINE, STATIC_AFFINE, **kwargs)
