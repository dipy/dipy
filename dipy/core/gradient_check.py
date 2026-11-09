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

import numpy as np
from scipy import ndimage as ndi

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.reconst.shm import CsaOdfModel
from dipy.segment.threshold import otsu
from dipy.utils.deprecator import warning_for_keywords

# Six axis permutations and four canonical flip configurations.
# Together they enumerate the 24 distinct (permutation, flip) gradient-table
# transforms identified in :footcite:p:`Aganj2018`.  Flipping all three axes is
# equivalent to no flip thanks to the antipodal symmetry of the ODF, so we
# only enumerate "no flip" plus the three single-axis flips.
_PERMUTATIONS = tuple(permutations(range(3)))
_FLIPS = (
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


def _foreground_mask(data, gtab):
    """Foreground voxels from the mean b0 volume (Otsu threshold).

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z, N)
        Diffusion-weighted volume series.
    gtab : GradientTable
        Gradient table associated with ``data``.

    Returns
    -------
    mask : ndarray of bool, shape (X, Y, Z)
        Largest connected foreground component, holes filled.
    """
    mean_b0 = data[..., gtab.b0s_mask].mean(axis=-1, dtype=np.float32)
    mean_b0 = ndi.median_filter(mean_b0, size=3)
    mask = mean_b0 > otsu(mean_b0)
    labels, nb_labels = ndi.label(mask)
    if nb_labels > 1:
        sizes = np.bincount(labels.ravel())[1:]
        mask = labels == (1 + np.argmax(sizes))
    return ndi.binary_fill_holes(mask)


def _white_matter_mask(signal, gtab, gfa, gfa_threshold, adc_threshold):
    """Fibrous-tissue voxels from GFA and mean apparent diffusion coefficient.

    Parameters
    ----------
    signal : ndarray, shape (V, N)
        Diffusion signal of V voxels.
    gtab : GradientTable
        Gradient table associated with ``signal``.
    gfa : ndarray, shape (V,)
        Generalized fractional anisotropy of each voxel.
    gfa_threshold : float
        Minimum GFA.
    adc_threshold : float
        Maximum mean ADC, in mm^2/s.

    Returns
    -------
    keep : ndarray of bool, shape (V,)
        Voxels that satisfy both criteria.
    """
    b0s_mask = gtab.b0s_mask
    s0 = signal[:, b0s_mask].mean(axis=-1, dtype=np.float32)
    ratio = signal[:, ~b0s_mask] / np.where(s0 > 0, s0, 1)[:, None]
    bvals = gtab.bvals[~b0s_mask].astype(np.float32)
    adc = -np.log(np.clip(ratio, 1e-8, None)) / bvals
    mean_adc = adc.mean(axis=-1)
    return (
        (np.nan_to_num(gfa) > gfa_threshold)
        & (s0 > 0)
        & (mean_adc > 0)
        & (mean_adc < adc_threshold)
    )


def _continuity_tensor(coef_gradient, sampling_matrix, vertices):
    r"""Accumulate the transform-independent part of the continuity error.

    With :math:`g_d(\hat n) = \sum_k B_{\hat n k} \partial_d c_k` the spatial
    gradient of the ODF, the error of a transform :math:`T` is
    :math:`\sum_{d,e,a,b} T_{da} T_{eb} A_{deab}` where
    :math:`A_{deab} = \sum_{x,\hat n} g_d g_e \hat n_a \hat n_b`.

    Parameters
    ----------
    coef_gradient : ndarray, shape (V, K, 3)
        Spatial gradient of the SH coefficients of V voxels.
    sampling_matrix : ndarray, shape (M, K)
        SH basis sampled on M sphere vertices.
    vertices : ndarray, shape (M, 3)
        Sphere vertices.

    Returns
    -------
    tensor : ndarray, shape (3, 3, 3, 3)
        Tensor :math:`A`.
    """
    nb_voxels, nb_coef, _ = coef_gradient.shape
    flat = coef_gradient.reshape(nb_voxels, nb_coef * 3)
    gram = (flat.T @ flat).astype(np.float64).reshape(nb_coef, 3, nb_coef, 3)
    basis = np.asarray(sampling_matrix, dtype=np.float64)
    per_vertex = np.einsum("mk,ml,kdle->mde", basis, basis, gram, optimize=True)
    return np.einsum("mde,ma,mb->deab", per_vertex, vertices, vertices)


@warning_for_keywords()
def fiber_continuity_error(
    data,
    gtab,
    *,
    mask=None,
    brain_mask=None,
    sphere=None,
    sh_order_max=4,
    voxel_size=None,
    smooth=0.006,
    gfa_threshold=0.3,
    adc_threshold=0.0015,
):
    """Fiber-continuity error for every (permutation, flip) configuration.

    The CSA-ODF is fitted once, on the brain voxels only, and the error of the
    24 candidate transforms is obtained from a single accumulated tensor. Peak
    memory scales with the number of brain voxels and the number of SH
    coefficients, not with the number of sphere vertices.

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z, N)
        Diffusion-weighted volume series.
    gtab : GradientTable
        Original gradient table associated with ``data``.
    mask : array_like of bool, shape (X, Y, Z), optional
        Mask of fibrous tissue used for scoring. When ``None`` a mask is
        derived from ``gfa_threshold`` and ``adc_threshold``. Voxels that
        have a neighbor outside ``brain_mask`` are not scored.
    brain_mask : array_like of bool, shape (X, Y, Z), optional
        Voxels on which the CSA-ODF is fitted. When ``None``, the foreground
        of the mean b0 volume (Otsu threshold) is used.
    sphere : Sphere, optional
        Discrete sphere used to integrate over directions. Defaults to
        ``dipy.data.default_sphere``.
    sh_order_max : int, optional
        Maximum spherical-harmonic order for the CSA-ODF reconstruction.
    voxel_size : sequence of float, length 3, optional
        Voxel size used as spacing of the spatial gradient. Unit spacing is
        used when ``None``.
    smooth : float, optional
        Laplace-Beltrami regularization for the CSA-ODF model.
    gfa_threshold : float, optional
        Minimum GFA of the default fibrous-tissue mask.
    adc_threshold : float, optional
        Maximum mean apparent diffusion coefficient (mm^2/s) of the default
        fibrous-tissue mask.

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

    if brain_mask is None:
        brain_mask = _foreground_mask(data, gtab)
    brain_mask = np.asarray(brain_mask, dtype=bool)

    signal = data[brain_mask]
    model = CsaOdfModel(gtab, sh_order_max=sh_order_max, smooth=smooth)
    fit = model.fit(signal)

    if mask is None:
        keep = _white_matter_mask(signal, gtab, fit.gfa, gfa_threshold, adc_threshold)
        mask = np.zeros(brain_mask.shape, dtype=bool)
        mask[brain_mask] = keep
    mask = np.asarray(mask, dtype=bool)

    # Central differences must only read fitted voxels.
    interior = ndi.binary_erosion(np.pad(brain_mask, 1))[1:-1, 1:-1, 1:-1]
    mask = mask & interior
    if not np.any(mask):
        raise ValueError(
            "Fiber-continuity mask is empty. Provide an explicit `mask` or "
            "loosen `gfa_threshold` and `adc_threshold`."
        )

    coef = np.zeros(brain_mask.shape + (fit.shm_coeff.shape[-1],), dtype=np.float32)
    coef[brain_mask] = fit.shm_coeff
    spacing = () if voxel_size is None else tuple(voxel_size)
    gradient = np.gradient(coef, *spacing, axis=(0, 1, 2))
    coef_gradient = np.stack([g[mask] for g in gradient], axis=-1)
    del coef, gradient

    vertices = np.asarray(sphere.vertices, dtype=np.float64)
    tensor = _continuity_tensor(coef_gradient, model.sampling_matrix(sphere), vertices)

    errors = {}
    for perm in _PERMUTATIONS:
        for flip in _FLIPS:
            transform = apply_transform(np.eye(3), perm, flip).T
            errors[(perm, flip)] = float(
                np.einsum("da,eb,deab->", transform, transform, tensor)
            )
    return errors


@warning_for_keywords()
def check_gradient_table(
    data,
    gtab,
    *,
    mask=None,
    brain_mask=None,
    sphere=None,
    sh_order_max=4,
    voxel_size=None,
    smooth=0.006,
    gfa_threshold=0.3,
    adc_threshold=0.0015,
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
        Fibrous-tissue mask used for scoring.
    brain_mask : array_like of bool, shape (X, Y, Z), optional
        Voxels on which the CSA-ODF is fitted. When ``None``, the foreground
        of the mean b0 volume (Otsu threshold) is used.
    sphere : Sphere, optional
        Sphere on which the ODF is sampled.
    sh_order_max : int, optional
        Maximum spherical-harmonic order. The paper recommends an SH order
        below the theoretical maximum for the dataset to keep the ODF
        denoised; a value of 4 is a sensible default for typical HARDI.
    voxel_size : sequence of float, length 3, optional
        Voxel size used as spacing of the spatial gradient. Unit spacing is
        used when ``None``.
    smooth : float, optional
        Laplace-Beltrami regularization for the CSA-ODF fit.
    gfa_threshold : float, optional
        Minimum GFA of the default fibrous-tissue mask.
    adc_threshold : float, optional
        Maximum mean apparent diffusion coefficient (mm^2/s) of the default
        fibrous-tissue mask.
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
        brain_mask=brain_mask,
        sphere=sphere,
        sh_order_max=sh_order_max,
        voxel_size=voxel_size,
        smooth=smooth,
        gfa_threshold=gfa_threshold,
        adc_threshold=adc_threshold,
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
    brain_mask=None,
    sphere=None,
    sh_order_max=4,
    voxel_size=None,
    smooth=0.006,
    gfa_threshold=0.3,
    adc_threshold=0.0015,
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
    mask, brain_mask, sphere, sh_order_max, voxel_size, smooth :
        Forwarded to :func:`check_gradient_table`.
    gfa_threshold, adc_threshold : float, optional
        Forwarded to :func:`check_gradient_table`.

    Returns
    -------
    gtab_corrected : GradientTable
        Gradient table with corrected b-vectors. If the recommended
        transform is the identity, the input ``gtab`` is returned
        unchanged.
    perm : tuple of int
        Axis permutation that was applied.
    flip : tuple of int
        Sign multipliers that were applied.
    """
    perm, flip = check_gradient_table(
        data,
        gtab,
        mask=mask,
        brain_mask=brain_mask,
        sphere=sphere,
        sh_order_max=sh_order_max,
        voxel_size=voxel_size,
        smooth=smooth,
        gfa_threshold=gfa_threshold,
        adc_threshold=adc_threshold,
    )

    if (perm, flip) == ((0, 1, 2), (1, 1, 1)):
        return gtab, perm, flip

    bvecs_corr = correct_bvecs(gtab.bvecs, perm, flip)
    gtab_corrected = gradient_table(
        gtab.bvals,
        bvecs=bvecs_corr,
        b0_threshold=gtab.b0_threshold,
    )
    return gtab_corrected, perm, flip
