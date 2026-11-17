"""Boundary-based registration (BBR) of an EPI reference volume to an
anatomical image :footcite:p:`Greve2009`.

References
----------
.. footbibliography::
"""

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    gaussian_filter,
    map_coordinates,
)
from scipy.optimize import minimize

from dipy.align._public import affine_registration
from dipy.align.imaffine import AffineMap
from dipy.align.reslice import reslice
from dipy.segment.tissue import TissueClassifierHMRF
from dipy.utils.logging import logger

_CONTRAST_SIGN = {"gm_brighter": 1.0, "wm_brighter": -1.0}


def _wm_probability_hmrf(static, static_affine, static_mask, *, zoom=1.0):
    """White-matter probability of a T1 image from a 3-class HMRF.

    The image is cropped to the brain mask and resampled to `zoom` mm before
    the segmentation: the HMRF time grows with the number of voxels, and the
    boundary is located below the voxel size anyway.

    Parameters
    ----------
    static : ndarray (X, Y, Z)
        T1-weighted image.
    static_affine : ndarray (4, 4)
        Voxel-to-world matrix of `static`.
    static_mask : ndarray (X, Y, Z)
        Brain mask on the grid of `static`.
    zoom : float, optional
        Isotropic voxel size, in mm, of the segmentation grid.

    Returns
    -------
    wm : ndarray
        White-matter partial-volume map on the segmentation grid.
    wm_affine : ndarray (4, 4)
        Voxel-to-world matrix of the segmentation grid.

    Raises
    ------
    ValueError
        If `static_mask` is empty.
    """
    mask = np.asarray(static_mask) > 0
    if not mask.any():
        raise ValueError("The brain mask is empty.")
    ijk = np.argwhere(mask)
    lo = np.maximum(ijk.min(axis=0) - 2, 0)
    hi = np.minimum(ijk.max(axis=0) + 3, mask.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    crop = np.asarray(static, dtype=np.float64)[box] * mask[box]
    crop_affine = static_affine.copy()
    crop_affine[:3, 3] = static_affine[:3, :3] @ lo + static_affine[:3, 3]

    zooms = np.sqrt(np.sum(static_affine[:3, :3] ** 2, axis=0))
    if np.allclose(zooms, zoom, atol=1e-3):
        image, image_affine = crop, crop_affine
    else:
        image, image_affine = reslice(crop, crop_affine, zooms, new_zooms=(zoom,) * 3)
    _, _, pve = TissueClassifierHMRF(verbose=False).classify(image, 3, 0.1)
    return np.ascontiguousarray(pve[..., 2], dtype=np.float64), image_affine


def _boundary_points(wm, wm_affine, *, n_points, rng):
    """Points and outward normals of the white-matter boundary.

    Parameters
    ----------
    wm : ndarray (X, Y, Z)
        White-matter probability.
    wm_affine : ndarray (4, 4)
        Voxel-to-world matrix of `wm`.
    n_points : int
        Maximal number of points to return.
    rng : numpy.random.Generator
        Generator used to subsample the points.

    Returns
    -------
    points : ndarray (N, 3)
        Boundary points in world coordinates.
    normals : ndarray (N, 3)
        Unit normals in world coordinates, pointing from white to gray matter.
    """
    p = gaussian_filter(np.asarray(wm, dtype=np.float64), sigma=1.0)
    inside = p > 0.5
    if not inside.any() or inside.all():
        raise ValueError("The white-matter mask has no boundary.")
    edge = binary_dilation(inside) & ~binary_erosion(inside)
    ijk = np.argwhere(edge).astype(np.float64)

    grad_vox = np.stack(np.gradient(p), axis=-1)[edge]
    # Gradient with respect to world coordinates, in units per mm.
    lin = wm_affine[:3, :3]
    grad = grad_vox @ np.linalg.inv(lin)
    norm = np.linalg.norm(grad, axis=1)
    keep = norm > 1e-3
    ijk, grad, norm = ijk[keep], grad[keep], norm[keep]
    if len(ijk) == 0:
        raise ValueError("The white-matter mask has no boundary.")

    xyz = ijk @ lin.T + wm_affine[:3, 3]
    value = p[edge][keep]
    xyz = xyz - ((value - 0.5) / norm**2)[:, None] * grad
    normals = -grad / norm[:, None]

    if len(xyz) > n_points:
        sel = rng.choice(len(xyz), size=n_points, replace=False)
        xyz, normals = xyz[sel], normals[sel]
    return xyz, normals


def _rigid_matrix(params, center):
    """Rigid transform rotating about `center`, then translating.

    Parameters
    ----------
    params : ndarray (6,)
        Rotations about x, y, z in degrees, then translations in mm. With
        these units a unit step moves a point of the brain by about 1 mm.
    center : ndarray (3,)
        Center of rotation in world coordinates.

    Returns
    -------
    matrix : ndarray (4, 4)
        The rigid transform.
    """
    rx, ry, rz = np.deg2rad(params[:3])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rot = rot_z @ rot_y @ rot_x
    matrix = np.eye(4)
    matrix[:3, :3] = rot
    matrix[:3, 3] = center - rot @ center + params[3:]
    return matrix


class _BBRCost:
    """Boundary-based cost of a rigid transform."""

    def __init__(
        self,
        moving,
        moving_affine,
        points,
        normals,
        starting_affine,
        center,
        sample_distance,
        slope,
        sign,
        threshold,
    ):
        self.moving = moving
        self.world2moving = np.linalg.inv(moving_affine)
        self.wm_pts = points - sample_distance[0] * normals
        self.gm_pts = points + sample_distance[1] * normals
        self.start = starting_affine
        self.center = center
        self.slope = slope
        self.sign = sign
        self.threshold = threshold
        self.min_valid = max(500, int(0.05 * len(points)))
        self.n_valid = 0

    def affine(self, params):
        matrix = _rigid_matrix(params[:6], self.center)
        if len(params) > 6:
            # Scaling along the axes of the static image, about the center.
            zoom = np.eye(4)
            zoom[:3, :3] = np.diag(1.0 + np.asarray(params[6:9]) / 100.0)
            zoom[:3, 3] = self.center - zoom[:3, :3] @ self.center
            matrix = matrix @ zoom
        return self.start @ matrix

    def _sample(self, pts, mat):
        vox = pts @ mat[:3, :3].T + mat[:3, 3]
        values = map_coordinates(self.moving, vox.T, order=1, mode="constant", cval=0.0)
        shape = np.array(self.moving.shape) - 1
        inside = np.all((vox >= 0) & (vox <= shape), axis=1)
        return values, inside

    def __call__(self, params):
        mat = self.world2moving @ self.affine(params)
        wm, wm_in = self._sample(self.wm_pts, mat)
        gm, gm_in = self._sample(self.gm_pts, mat)
        valid = wm_in & gm_in & (wm > self.threshold) & (gm > self.threshold)
        self.n_valid = int(valid.sum())
        if self.n_valid < self.min_valid:
            # Out of the field of view: the worst cost, so that the optimizer
            # steps back.
            return 2.0
        wm, gm = wm[valid], gm[valid]
        q = 100.0 * (gm - wm) / (0.5 * (gm + wm))
        return float(np.mean(1.0 - np.tanh(self.slope * self.sign * q)))


def bbr_registration(
    moving,
    static,
    moving_affine,
    static_affine,
    *,
    wm_mask=None,
    static_mask=None,
    starting_affine=None,
    boundary_contrast="gm_brighter",
    scaling=False,
    sample_distance=(2.0, 1.5),
    slope=0.5,
    n_points=20000,
    random_seed=0,
):
    """Boundary-based registration of an EPI volume to an anatomical image.

    The transform maximizes the intensity contrast of the EPI across the
    white-matter boundary of the anatomical image :footcite:p:`Greve2009`.

    Parameters
    ----------
    moving : ndarray (X, Y, Z)
        EPI reference volume, such as a b0, a mean BOLD or an SBRef volume.
    static : ndarray (X, Y, Z)
        Anatomical (T1-weighted) image.
    moving_affine : ndarray (4, 4)
        Voxel-to-world matrix of `moving`.
    static_affine : ndarray (4, 4)
        Voxel-to-world matrix of `static`.
    wm_mask : ndarray (X, Y, Z), optional
        White-matter mask or probability map on the grid of `static`. If
        None, it is estimated from `static` with a 3-class HMRF, which needs
        `static_mask`.
    static_mask : ndarray (X, Y, Z), optional
        Brain mask on the grid of `static`. Only used when `wm_mask` is None.
    starting_affine : ndarray (4, 4), optional
        Initial transform from the world coordinates of `static` to those of
        `moving`. If None, a rigid mutual-information registration is run.
    boundary_contrast : {'gm_brighter', 'wm_brighter'}, optional
        Whether gray matter is brighter than white matter in `moving` (b0,
        T2*-weighted BOLD) or darker (T1-weighted EPI).
    scaling : bool, optional
        If True, a scaling along each axis of `static` is estimated on top of
        the rigid transform (9 parameters). Use it when the two images do not
        share the same geometry, for instance when only one of them was
        corrected for gradient nonlinearity. A scaling along the
        phase-encoding axis also absorbs a uniform stretch due to the
        distortion of the EPI.
    sample_distance : tuple of float, optional
        Distances in mm from the boundary, along its normal, at which the
        white-matter and the gray-matter intensities are sampled.
    slope : float, optional
        Slope of the hyperbolic tangent of the cost.
    n_points : int, optional
        Maximal number of boundary points.
    random_seed : int, optional
        Seed of the subsampling of the boundary points.

    Returns
    -------
    resampled : ndarray
        `moving` resampled on the grid of `static`.
    affine : ndarray (4, 4)
        Transform from the world coordinates of `static` to those of
        `moving`. It is rigid, unless `scaling` is True.

    Examples
    --------
    Align a b0 volume to a T1-weighted image of the same subject, with the
    white-matter boundary estimated from the skull-stripped T1:

    >>> from dipy.align import bbr_registration
    >>> from dipy.io.image import load_nifti
    >>> b0, b0_affine = load_nifti("b0.nii.gz")  # doctest: +SKIP
    >>> t1, t1_affine = load_nifti("t1.nii.gz")  # doctest: +SKIP
    >>> brain_mask, _ = load_nifti("t1_brain_mask.nii.gz")  # doctest: +SKIP
    >>> b0_in_t1, t1_to_b0 = bbr_registration(
    ...     b0, t1, b0_affine, t1_affine, static_mask=brain_mask
    ... )  # doctest: +SKIP

    References
    ----------
    .. footbibliography::
    """
    moving = np.asarray(moving, dtype=np.float64)
    static = np.asarray(static)
    if moving.ndim != 3:
        raise ValueError(
            "moving must be a 3D reference volume (e.g. a b0 or a mean BOLD "
            f"volume). Got shape {moving.shape}."
        )
    if static.ndim != 3:
        raise ValueError(f"static must be a 3D image. Got shape {static.shape}.")
    if boundary_contrast not in _CONTRAST_SIGN:
        raise ValueError(
            "boundary_contrast must be 'gm_brighter' or 'wm_brighter', got "
            f"{boundary_contrast!r}."
        )

    if wm_mask is None:
        if static_mask is None:
            raise ValueError("Either wm_mask or static_mask is required.")
        wm_mask, wm_affine = _wm_probability_hmrf(static, static_affine, static_mask)
    else:
        wm_mask = np.asarray(wm_mask, dtype=np.float64)
        if wm_mask.shape != static.shape:
            raise ValueError("wm_mask must be on the grid of static.")
        wm_affine = static_affine

    if starting_affine is None:
        # The EPI has the smaller field of view, so it is the static image.
        _, epi_to_anat = affine_registration(
            static.astype(np.float64),
            moving,
            moving_affine=static_affine,
            static_affine=moving_affine,
            metric="MI",
            pipeline=["center_of_mass", "translation", "rigid"],
        )
        starting_affine = np.linalg.inv(epi_to_anat)
    starting_affine = np.asarray(starting_affine, dtype=np.float64)

    rng = np.random.default_rng(random_seed)
    points, normals = _boundary_points(wm_mask, wm_affine, n_points=n_points, rng=rng)
    center = points.mean(axis=0)
    threshold = 0.05 * np.percentile(moving, 99)
    sign = _CONTRAST_SIGN[boundary_contrast]

    n_params = 9 if scaling else 6

    def run(image, pts, nrm, x0):
        cost = _BBRCost(
            image,
            moving_affine,
            pts,
            nrm,
            starting_affine,
            center,
            sample_distance,
            slope,
            sign,
            threshold,
        )
        start_cost = cost(x0)
        if cost.n_valid < cost.min_valid:
            raise ValueError(
                f"Only {cost.n_valid} boundary points fall in the moving image: "
                "the starting transform or the field of view is unusable."
            )
        res = minimize(
            cost,
            x0,
            method="Powell",
            options={"xtol": 1e-3, "ftol": 1e-5, "maxiter": 50},
        )
        params = res.x if res.fun <= start_cost else x0
        final_cost = cost(params)
        return params, final_cost, cost.n_valid, cost

    # Stage 1: smoothed moving image and a subset of the points.
    n_coarse = min(5000, len(points))
    sub = rng.choice(len(points), size=n_coarse, replace=False)
    params, _, _, _ = run(
        gaussian_filter(moving, sigma=1.0),
        points[sub],
        normals[sub],
        np.zeros(n_params),
    )
    # Stage 2: full resolution and all the points.
    params, final_cost, n_valid, cost = run(moving, points, normals, params)

    affine = cost.affine(params)
    logger.info(
        f"BBR: cost {final_cost:.4f}, {n_valid} valid points, rotations "
        f"{np.round(params[:3], 2)} deg, translations "
        f"{np.round(params[3:6], 2)} mm from the start"
        + (f", scalings {np.round(params[6:], 2)} %" if scaling else "")
    )
    resampled = AffineMap(
        affine,
        domain_grid_shape=static.shape,
        domain_grid2world=static_affine,
        codomain_grid_shape=moving.shape,
        codomain_grid2world=moving_affine,
    ).transform(moving)
    return resampled, affine
