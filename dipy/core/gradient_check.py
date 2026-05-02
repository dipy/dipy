"""Automatic verification of the diffusion gradient table.

Implements the tractography-free fiber-continuity criterion of
:footcite:p:`Aganj2018` to detect (and undo) the axis permutation and
single-axis flip that brings the diffusion-gradient coordinate frame
into agreement with the image coordinate frame.

The criterion is the fiber-continuity error

.. math::

    \\epsilon(T) = \\int_{\\Omega} \\int_{S^2}
        \\bigl(T(\\hat n) \\cdot \\nabla_x \\psi_G(x, \\hat n)\\bigr)^2
        \\, d\\hat n \\, dx,

where :math:`\\psi_G` is the orientation distribution function (ODF)
reconstructed once from the input gradient table :math:`G`, :math:`T` is a
candidate axis permutation/flip applied to the unit-vector sampling
directions, and :math:`\\Omega` is a fibrous-tissue mask.

The transform :math:`T^*` minimizing :math:`\\epsilon` is returned; applying
:math:`T^*` to the input b-vectors yields a corrected gradient table.

References
----------
.. footbibliography::
"""

from itertools import permutations
import warnings

import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.reconst.shm import CsaOdfModel
from dipy.testing.decorators import warning_for_keywords

# Six axis permutations and four canonical flip configurations.
# Together they enumerate the 24 distinct (permutation, flip) gradient-table
# transforms identified in :footcite:p:`Aganj2018`.  Flipping all three axes is
# equivalent to no flip thanks to the antipodal symmetry of the ODF, so we
# only enumerate "no flip" plus the three single-axis flips.
PERMUTATIONS = tuple(permutations(range(3)))
FLIPS = (
    (1, 1, 1),
    (-1, 1, 1),
    (1, -1, 1),
    (1, 1, -1),
)


def _flip_label(flip):
    if flip == (1, 1, 1):
        return "no flip"
    axis = "xyz"[[i for i, v in enumerate(flip) if v == -1][0]]
    return f"flip {axis}"


def _perm_label(perm):
    return "[" + " ".join("XYZ"[i] for i in perm) + "]"


def transform_label(perm, flip):
    """Human-readable label for a (permutation, flip) configuration."""
    return f"perm={_perm_label(perm)}, {_flip_label(flip)}"


def apply_transform(vecs, perm, flip):
    """Apply a permutation followed by an axis flip to the last axis of `vecs`.

    Parameters
    ----------
    vecs : array_like, shape (..., 3)
        Vectors to transform.
    perm : sequence of int, length 3
        Axis permutation. ``new[i] = old[perm[i]]``.
    flip : sequence of int, length 3
        Per-axis sign multiplier from ``{-1, +1}`` applied after the
        permutation.

    Returns
    -------
    out : ndarray, shape (..., 3)
        Transformed vectors.
    """
    vecs = np.asarray(vecs)
    out = vecs[..., list(perm)] * np.asarray(flip, dtype=vecs.dtype)
    return out


def _spatial_gradient(odf_vol, voxel_size=None):
    """Spatial gradient of an ODF volume sampled on a sphere.

    Parameters
    ----------
    odf_vol : ndarray, shape (X, Y, Z, M)
        ODF samples; the trailing axis indexes sphere directions.
    voxel_size : sequence of float, length 3, optional
        Physical voxel size along each spatial axis. Defaults to unit
        spacing, which is sufficient because :math:`\\epsilon` is compared
        across configurations sharing the same scale factor.

    Returns
    -------
    grad : ndarray, shape (X, Y, Z, M, 3)
        Spatial gradient of the ODF volume; the trailing axis indexes the
        spatial direction (x, y, z).
    """
    if voxel_size is None:
        gx, gy, gz = np.gradient(odf_vol, axis=(0, 1, 2))
    else:
        gx, gy, gz = np.gradient(
            odf_vol, voxel_size[0], voxel_size[1], voxel_size[2], axis=(0, 1, 2)
        )
    return np.stack([gx, gy, gz], axis=-1)


def _make_default_mask(data, gtab, *, gfa_threshold=0.4, adc_threshold=0.01):
    """Approximate white-matter mask following :footcite:p:`Aganj2018`.

    A voxel is kept when its mean apparent diffusion coefficient is below
    ``adc_threshold`` (in mm^2/s, assuming bvals in s/mm^2) and its
    generalized fractional anisotropy exceeds ``gfa_threshold``. This
    matches the simple anisotropy heuristic used in the paper when no
    external white-matter segmentation is available.
    """
    from dipy.reconst.shm import CsaOdfModel

    b0_mask = gtab.b0s_mask
    dwi_mask = ~b0_mask
    if not np.any(dwi_mask):
        raise ValueError("Gradient table contains no diffusion-weighted volumes.")
    if not np.any(b0_mask):
        raise ValueError("Gradient table contains no b=0 volume.")

    s0 = data[..., b0_mask].mean(axis=-1).astype(np.float64)
    sd = data[..., dwi_mask].astype(np.float64)
    bvals_dwi = gtab.bvals[dwi_mask].astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(s0[..., None] > 0, sd / s0[..., None], 0.0)
        ratio = np.clip(ratio, 1e-8, None)
        adc_per_dir = -np.log(ratio) / bvals_dwi[None, None, None, :]
    mean_adc = np.mean(adc_per_dir, axis=-1)
    mean_adc = np.where(np.isfinite(mean_adc), mean_adc, np.inf)

    csa = CsaOdfModel(gtab, sh_order_max=4)
    coarse_mask = (s0 > 0) & (mean_adc < adc_threshold) & (mean_adc > 0)
    fit = csa.fit(data, mask=coarse_mask)
    gfa = np.nan_to_num(fit.gfa, nan=0.0, posinf=0.0, neginf=0.0)

    return coarse_mask & (gfa > gfa_threshold)


@warning_for_keywords()
def fiber_continuity_error(
    data,
    gtab,
    *,
    mask=None,
    sphere=None,
    sh_order_max=4,
    voxel_size=None,
    smooth=0.006,
):
    """Fiber-continuity error for every (permutation, flip) configuration.

    Reconstructs the ODF volume once with the supplied gradient table, then
    evaluates :math:`\\epsilon(T)` for each of the 24 candidate transforms.

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z, N)
        Diffusion-weighted volume series.
    gtab : GradientTable
        Original gradient table associated with ``data``.
    mask : array_like of bool, shape (X, Y, Z), optional
        Mask of fibrous tissue. When ``None`` an internal ADC/GFA-based
        white-matter approximation is used.
    sphere : Sphere, optional
        Discrete sphere on which the ODF is sampled. Defaults to
        ``dipy.data.default_sphere``.
    sh_order_max : int, optional
        Maximum spherical-harmonic order for the CSA-ODF reconstruction.
    voxel_size : sequence of float, length 3, optional
        Physical voxel size; used as the spacing for the spatial gradient.
        Only the relative scale matters; if ``None`` unit spacing is used.
    smooth : float, optional
        Laplace-Beltrami regularization for the CSA-ODF model.

    Returns
    -------
    errors : dict
        Mapping ``(perm, flip) -> epsilon(T)`` for the 24 configurations.
    """
    data = np.asarray(data)
    if data.ndim != 4:
        raise ValueError("`data` must be a 4-D array (X, Y, Z, N).")
    if sphere is None:
        sphere = default_sphere

    if mask is None:
        mask = _make_default_mask(data, gtab)
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        raise ValueError(
            "Fiber-continuity mask is empty. Provide an explicit `mask` or "
            "loosen the default ADC/GFA thresholds."
        )

    csa = CsaOdfModel(gtab, sh_order_max=sh_order_max, smooth=smooth)
    fit = csa.fit(data)
    # Sample the ODF on the sphere for every voxel (we need the spatial
    # gradient in voxels neighbouring the mask, so we sample everywhere).
    odf_vol = fit.odf(sphere)

    grad_vol = _spatial_gradient(odf_vol, voxel_size=voxel_size)
    grad_masked = grad_vol[mask]

    vertices = np.asarray(sphere.vertices, dtype=np.float64)

    errors = {}
    for perm in PERMUTATIONS:
        for flip in FLIPS:
            transformed = apply_transform(vertices, perm, flip)
            dots = np.einsum("vmd,md->vm", grad_masked, transformed)
            errors[(perm, flip)] = float(np.sum(dots * dots))
    return errors


@warning_for_keywords()
def check_gradient_table(
    data,
    gtab,
    *,
    mask=None,
    sphere=None,
    sh_order_max=4,
    voxel_size=None,
    smooth=0.006,
    return_errors=False,
):
    """Suggest the permutation/flip required to align ``gtab`` with ``data``.

    This implements the Aganj 2018 fiber-continuity criterion for automatic
    gradient-table verification :footcite:p:`Aganj2018`. The reported
    transform :math:`T^*` is the one minimizing the fiber-continuity error;
    applying it to the b-vectors yields a gradient table consistent with the
    image coordinate frame.

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z, N)
        Diffusion-weighted volume series.
    gtab : GradientTable
        Gradient table to verify.
    mask : array_like of bool, shape (X, Y, Z), optional
        Fibrous-tissue mask.
    sphere : Sphere, optional
        Sphere on which the ODF is sampled.
    sh_order_max : int, optional
        Maximum spherical-harmonic order. The paper recommends an SH order
        below the theoretical maximum for the dataset to keep the ODF
        denoised; a value of 4 is a sensible default for typical HARDI.
    voxel_size : sequence of float, length 3, optional
        Voxel size; only the relative scale matters.
    smooth : float, optional
        Laplace-Beltrami regularization for the CSA-ODF fit.
    return_errors : bool, optional
        If True, also return the dictionary of fiber-continuity errors for
        every (permutation, flip) configuration.

    Returns
    -------
    perm : tuple of int, length 3
        Axis permutation; ``new_axis[i] = old_axis[perm[i]]``.
    flip : tuple of int, length 3
        Per-axis sign multipliers in ``{-1, +1}`` applied after the
        permutation.
    errors : dict, optional
        Returned only when ``return_errors`` is True.

    References
    ----------
    .. footbibliography::
    """
    errors = fiber_continuity_error(
        data,
        gtab,
        mask=mask,
        sphere=sphere,
        sh_order_max=sh_order_max,
        voxel_size=voxel_size,
        smooth=smooth,
    )
    perm, flip = min(errors, key=errors.get)
    if return_errors:
        return perm, flip, errors
    return perm, flip


def correct_bvecs(bvecs, perm, flip):
    """Apply a (permutation, flip) configuration to a b-vector array.

    Parameters
    ----------
    bvecs : array_like, shape (N, 3)
        Original b-vectors.
    perm : sequence of int, length 3
        Axis permutation.
    flip : sequence of int, length 3
        Per-axis sign multipliers in ``{-1, +1}``.

    Returns
    -------
    bvecs_out : ndarray, shape (N, 3)
        Corrected b-vectors.
    """
    bvecs = np.asarray(bvecs, dtype=np.float64)
    if bvecs.ndim != 2 or bvecs.shape[1] != 3:
        raise ValueError("`bvecs` must have shape (N, 3).")
    return apply_transform(bvecs, perm, flip)


@warning_for_keywords()
def correct_gradient_table(
    data,
    gtab,
    *,
    mask=None,
    sphere=None,
    sh_order_max=4,
    voxel_size=None,
    smooth=0.006,
):
    """Return a gradient table corrected by the Aganj 2018 criterion.

    This is a convenience wrapper that runs :func:`check_gradient_table` and
    builds a new ``GradientTable`` whose b-vectors have been transformed by
    the recommended permutation and flip. The b-values are unchanged because
    permuting and flipping b-vectors does not affect their magnitudes.

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z, N)
        Diffusion-weighted volume series.
    gtab : GradientTable
        Gradient table to verify.
    mask, sphere, sh_order_max, voxel_size, smooth :
        Forwarded to :func:`check_gradient_table`.

    Returns
    -------
    gtab_corrected : GradientTable
        Gradient table with corrected b-vectors. If the input gradient
        table was already aligned with the image, this is functionally
        equivalent to ``gtab``.
    perm : tuple of int
        Axis permutation that was applied.
    flip : tuple of int
        Sign multipliers that were applied.
    """
    perm, flip = check_gradient_table(
        data,
        gtab,
        mask=mask,
        sphere=sphere,
        sh_order_max=sh_order_max,
        voxel_size=voxel_size,
        smooth=smooth,
    )

    bvecs_corr = correct_bvecs(gtab.bvecs, perm, flip)

    if (perm, flip) == ((0, 1, 2), (1, 1, 1)):
        warnings.warn(
            "check_gradient_table found no permutation or flip is needed; "
            "the gradient table appears already aligned with the image.",
            stacklevel=2,
        )

    gtab_corrected = gradient_table(
        gtab.bvals,
        bvecs=bvecs_corr,
        b0_threshold=gtab.b0_threshold,
    )
    return gtab_corrected, perm, flip
