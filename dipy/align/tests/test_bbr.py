import numpy as np
import numpy.testing as npt
from scipy.ndimage import gaussian_filter

from dipy.align.bbr import _rigid_matrix, bbr_registration
from dipy.align.imaffine import AffineMap


def _phantom(shape, affine, *, gm_brighter):
    """Nested ellipsoids: white matter inside gray matter inside CSF."""
    ijk = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1)
    xyz = ijk @ affine[:3, :3].T + affine[:3, 3]
    r = np.sqrt(np.sum((xyz / np.array([22.0, 28.0, 20.0])) ** 2, axis=-1))
    wm, gm, csf = r < 0.7, (r >= 0.7) & (r < 0.9), (r >= 0.9) & (r < 1.0)
    values = (60.0, 100.0, 140.0) if gm_brighter else (100.0, 60.0, 20.0)
    image = values[0] * wm + values[1] * gm + values[2] * csf
    return image, wm.astype(np.float64)


def _setup(*, gm_brighter=True):
    static_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    static_affine[:3, 3] = [-32.0, -36.0, -28.0]
    static_shape = (64, 72, 56)
    t1, wm = _phantom(static_shape, static_affine, gm_brighter=False)

    moving_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    moving_affine[:3, 3] = [-40.0, -44.0, -36.0]
    moving_shape = (40, 44, 36)
    # The EPI phantom in the world of the T1, then moved by a known rigid.
    hi_res, _ = _phantom(static_shape, static_affine, gm_brighter=gm_brighter)
    truth = _rigid_matrix(np.array([3.0, -2.0, 2.5, 2.0, -3.0, 1.5]), np.zeros(3))
    # truth maps T1 world to EPI world, so resample with its inverse.
    epi = AffineMap(
        np.linalg.inv(truth),
        domain_grid_shape=moving_shape,
        domain_grid2world=moving_affine,
        codomain_grid_shape=static_shape,
        codomain_grid2world=static_affine,
    ).transform(gaussian_filter(hi_res, 0.7))
    rng = np.random.default_rng(0)
    epi = epi + rng.normal(0, 2.0, epi.shape) * (epi > 0)
    return epi, t1, moving_affine, static_affine, wm, truth


def _error_mm(affine, truth, wm, static_affine):
    xyz = np.argwhere(wm > 0) @ static_affine[:3, :3].T + static_affine[:3, 3]
    a = xyz @ affine[:3, :3].T + affine[:3, 3]
    b = xyz @ truth[:3, :3].T + truth[:3, 3]
    return np.linalg.norm(a - b, axis=1).mean()


def test_bbr_recovers_rigid():
    epi, t1, m_aff, s_aff, wm, truth = _setup()
    resampled, affine = bbr_registration(
        epi, t1, m_aff, s_aff, wm_mask=wm, starting_affine=np.eye(4)
    )
    npt.assert_(_error_mm(np.eye(4), truth, wm, s_aff) > 2.0)
    npt.assert_(_error_mm(affine, truth, wm, s_aff) < 0.5)
    npt.assert_(resampled.shape == t1.shape)

    # The cost is a ratio, so the intensity scale does not matter.
    _, scaled = bbr_registration(
        epi * 1000, t1, m_aff, s_aff, wm_mask=wm, starting_affine=np.eye(4)
    )
    npt.assert_(_error_mm(scaled, affine, wm, s_aff) < 1e-3)


def test_bbr_scaling():
    epi, t1, m_aff, s_aff, wm, truth = _setup()
    # A T1 grid with a wrong voxel size: the anatomy is 12% larger along x.
    scaled_aff = s_aff.copy()
    scaled_aff[0] *= 1.12
    kwargs = {"wm_mask": wm, "starting_affine": np.eye(4)}
    _, rigid = bbr_registration(epi, t1, m_aff, scaled_aff, **kwargs)
    _, zoomed = bbr_registration(epi, t1, m_aff, scaled_aff, scaling=True, **kwargs)
    # truth maps the unscaled T1 world to the EPI world.
    unscale = np.diag([1 / 1.12, 1.0, 1.0, 1.0])
    target = truth @ unscale
    rigid_err = _error_mm(rigid, target, wm, scaled_aff)
    zoomed_err = _error_mm(zoomed, target, wm, scaled_aff)
    npt.assert_(zoomed_err < 0.75)
    npt.assert_(rigid_err > 1.5 * zoomed_err)
    # The smallest singular value is the scaling along x.
    smallest = np.linalg.svd(zoomed[:3, :3], compute_uv=False).min()
    npt.assert_(abs(smallest - 1 / 1.12) < 0.03)
    npt.assert_(abs(np.linalg.det(rigid[:3, :3]) - 1.0) < 1e-6)


def test_bbr_contrast_sign():
    epi, t1, m_aff, s_aff, wm, truth = _setup(gm_brighter=False)
    kwargs = {"wm_mask": wm, "starting_affine": np.eye(4)}
    _, good = bbr_registration(
        epi, t1, m_aff, s_aff, boundary_contrast="wm_brighter", **kwargs
    )
    good_err = _error_mm(good, truth, wm, s_aff)
    npt.assert_(good_err < 0.5)
    # With the wrong sign the registration drifts away, or runs out of
    # boundary points with a usable signal.
    try:
        _, bad = bbr_registration(
            epi, t1, m_aff, s_aff, boundary_contrast="gm_brighter", **kwargs
        )
    except ValueError:
        return
    npt.assert_(_error_mm(bad, truth, wm, s_aff) > 2 * good_err)


def test_bbr_dropout():
    epi, t1, m_aff, s_aff, wm, truth = _setup()
    # Signal dropout in a slab of the EPI.
    epi[:, :, :8] = 0
    _, affine = bbr_registration(
        epi, t1, m_aff, s_aff, wm_mask=wm, starting_affine=np.eye(4)
    )
    npt.assert_(_error_mm(affine, truth, wm, s_aff) < 0.75)


def test_bbr_hmrf_fallback():
    epi, t1, m_aff, s_aff, wm, truth = _setup()
    # No white-matter mask: it is segmented from the masked T1.
    _, affine = bbr_registration(
        epi, t1, m_aff, s_aff, static_mask=t1 > 0, starting_affine=np.eye(4)
    )
    npt.assert_(_error_mm(affine, truth, wm, s_aff) < 0.75)


def test_bbr_errors():
    epi, t1, m_aff, s_aff, wm, _ = _setup()
    kwargs = {"wm_mask": wm, "starting_affine": np.eye(4)}
    npt.assert_raises(
        ValueError,
        bbr_registration,
        np.stack([epi, epi], -1),
        t1,
        m_aff,
        s_aff,
        **kwargs,
    )
    npt.assert_raises(
        ValueError,
        bbr_registration,
        epi,
        t1,
        m_aff,
        s_aff,
        boundary_contrast="other",
        **kwargs,
    )
    npt.assert_raises(
        ValueError,
        bbr_registration,
        epi,
        t1,
        m_aff,
        s_aff,
        wm_mask=np.zeros_like(wm),
        starting_affine=np.eye(4),
    )
    npt.assert_raises(
        ValueError,
        bbr_registration,
        epi,
        t1,
        m_aff,
        s_aff,
        starting_affine=np.eye(4),
    )
    far = np.eye(4)
    far[:3, 3] = 500.0
    npt.assert_raises(
        ValueError,
        bbr_registration,
        epi,
        t1,
        m_aff,
        s_aff,
        wm_mask=wm,
        starting_affine=far,
    )
