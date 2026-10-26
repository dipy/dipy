"""Fieldmap-less susceptibility distortion correction with SyN.

Echo-planar images (EPI) such as diffusion-weighted images are geometrically
distorted along the phase-encoding (PE) direction by off-resonance fields
near air-tissue interfaces. When neither a fieldmap nor a reverse
phase-encoded acquisition is available, the distortion can be estimated by
nonlinearly registering the distorted b0 to an undistorted anatomical image
of the same subject, with the deformation restricted to the PE axis
:footcite:p:`Wang2017`, :footcite:p:`Huang2008`.

The anatomical reference may be a T1-weighted image, whose intensities are
inverted so that cerebrospinal fluid is bright and white matter dark as in a
b0 :footcite:p:`Huang2008`, or a T2-weighted image, whose contrast already
resembles that of a b0 and whose intensities are mapped to the b0 through
their joint histogram.

References
----------
.. footbibliography::
"""

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    gaussian_filter,
)

from dipy.align import vector_fields as vfu
from dipy.align.imaffine import (
    AffineRegistration,
    MutualInformationMetric,
    transform_centers_of_mass,
)
from dipy.align.imwarp import DiffeomorphicMap, SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
from dipy.align.transforms import RigidTransform3D, TranslationTransform3D
from dipy.utils.logging import logger

__all__ = ["PhaseEncodingSyN", "syn_sdc"]


class PhaseEncodingSyN(SymmetricDiffeomorphicRegistration):
    """Symmetric diffeomorphic registration restricted to one direction.

    Every update of the forward and backward transforms is projected onto a
    fixed world-space direction before it is composed with the current
    transforms. The composition and inversion of displacements that are
    parallel to a single direction stay parallel to it, so the resulting
    diffeomorphism only moves points along that direction, as expected for a
    susceptibility-induced EPI distortion.

    Parameters
    ----------
    metric : SimilarityMetric
        The metric to be optimized.
    direction : array-like (3,)
        World-space direction the deformation is restricted to (e.g. the
        phase-encoding axis of the EPI grid). It does not need to be
        normalized.
    **kwargs : dict
        Keyword arguments passed to
        :class:`~dipy.align.imwarp.SymmetricDiffeomorphicRegistration`.
    """

    def __init__(self, metric, direction, **kwargs):
        super().__init__(metric, **kwargs)
        direction = np.asarray(direction, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if direction.shape != (3,) or norm == 0:
            raise ValueError("direction must be a nonzero 3-vector")
        self.direction = direction / norm

    def update(
        self, current_displacement, new_displacement, disp_world2grid, time_scaling
    ):
        """Compose the current displacement with a projected update.

        See :meth:`SymmetricDiffeomorphicRegistration.update`; the only
        difference is that ``new_displacement`` is first projected onto
        ``direction``.
        """
        new_displacement = np.asarray(new_displacement)
        proj = new_displacement @ self.direction
        new_displacement = np.ascontiguousarray(
            proj[..., None] * self.direction, dtype=new_displacement.dtype
        )
        return super().update(
            current_displacement, new_displacement, disp_world2grid, time_scaling
        )


def _pe_axis_index(pe_axis):
    if isinstance(pe_axis, str):
        axes = {"i": 0, "j": 1, "k": 2, "x": 0, "y": 1, "z": 2}
        pe_axis = axes.get(pe_axis.lower().rstrip("-"))
    if pe_axis not in (0, 1, 2):
        raise ValueError("pe_axis must be 0, 1, 2 or one of 'i', 'j', 'k'.")
    return pe_axis


def _scale(img, mask):
    """Scale an image to [0, 100] using the 99th percentile inside a mask."""
    vals = img[mask]
    hi = np.percentile(vals, 99) if vals.size else 0.0
    if hi <= 0:
        return np.zeros_like(img)
    return np.clip(img / hi, 0, 1) * 100.0


def _rigid_anat_to_epi(epi, epi_affine, anat, anat_affine, anat_mask, starting_affine):
    """Rigid MI alignment of the anatomy (moving) to the EPI (static).

    MI is sampled on the static grid, so using the EPI as the static image
    keeps anatomy outside a cropped EPI field of view out of the metric.
    Without a starting affine, the registration is started both from the
    world alignment (images of one session share scanner coordinates) and
    from the centers of mass (which a cropped field of view biases), and the
    result with the best metric is kept.
    """
    reg = AffineRegistration(
        metric=MutualInformationMetric(nbins=32, sampling_proportion=None),
        level_iters=[100, 50, 25],
        sigmas=[3.0, 1.0, 0.0],
        factors=[4, 2, 1],
    )
    kw = {
        "static_grid2world": epi_affine,
        "moving_grid2world": anat_affine,
        "moving_mask": anat_mask.astype(np.int32),
        "ret_metric": True,
    }
    if starting_affine is not None:
        starts = [(starting_affine, False)]
    else:
        com = transform_centers_of_mass(epi, epi_affine, anat, anat_affine).affine
        starts = [(np.eye(4), False), (np.eye(4), True), (com, True)]
    best = None
    for start, translate_first in starts:
        if translate_first:
            start = reg.optimize(
                epi, anat, TranslationTransform3D(), None, starting_affine=start, **kw
            )[0].affine
        amap, _, fopt = reg.optimize(
            epi, anat, RigidTransform3D(), None, starting_affine=start, **kw
        )
        if best is None or fopt < best[1]:
            best = (amap, fopt)
    return best[0]


def _anat_reference(anat_in_epi, mask, contrast, epi):
    """Map the anatomical image to a b0-like contrast inside the brain.

    A T1w image is intensity-inverted, so that cerebrospinal fluid is bright
    and white matter dark as in a b0. The contrast of a T2w image already
    resembles that of a b0, and its intensities are mapped to the
    conditional mean of ``epi`` (see :func:`_intensity_map`) to match the
    relative brightness of the tissues. Voxels outside ``mask`` are set to 0.
    """
    if contrast == "t1":
        # Inverting about the maximum keeps the relation with the b0 linear
        # over the whole brain, which the local cross-correlation needs.
        ref = anat_in_epi[mask].max() - anat_in_epi
    else:
        ref = _intensity_map(anat_in_epi, epi, mask)
    return np.where(mask, ref, 0.0)


def _epi_fov(epi):
    """Region of the EPI grid that was actually acquired.

    Voxels outside the field of view of the acquisition (e.g. after oblique
    cropping) are exactly zero, while background noise inside it is not.
    """
    return binary_fill_holes(binary_closing(epi != 0, iterations=2))


def _intensity_map(anat, epi, mask, nbins=64):
    """Map anatomical intensities to the expected EPI intensity.

    The conditional mean of ``epi`` given ``anat`` is estimated inside
    ``mask`` on quantile bins of ``anat`` and applied with linear
    interpolation, which captures nonlinear contrast relations such as
    partial-volume effects at the brain boundary.
    """
    a = anat[mask]
    e = epi[mask]
    edges = np.unique(np.percentile(a, np.linspace(0, 100, nbins + 1)))
    if edges.size < 3:
        return anat
    idx = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, edges.size - 2)
    counts = np.bincount(idx, minlength=edges.size - 1)
    sums = np.bincount(idx, weights=e, minlength=edges.size - 1)
    centers = np.bincount(idx, weights=a, minlength=edges.size - 1)
    ok = counts > 0
    centers = centers[ok] / counts[ok]
    means = sums[ok] / counts[ok]
    return np.interp(anat, centers, means)


def syn_sdc(
    dwi,
    dwi_affine,
    anat,
    anat_affine,
    *,
    pe_axis,
    anat_contrast="t1",
    anat_mask=None,
    b0_index=0,
    n_outer=2,
    level_iters=(100, 100, 50),
    step_length=0.25,
    radius=4,
    sigma_diff=5.0,
    ss_sigma_factor=0.2,
    mask_dilation=4,
    field_smooth=1.0,
    return_field=False,
    return_mapping=False,
):
    """Fieldmap-less susceptibility distortion correction (SyN-SDC).

    The distorted b0 is rigidly aligned to an undistorted, skull-stripped
    anatomical image of the same subject, which is mapped to a b0-like
    contrast and resampled onto the b0 grid. Anatomy outside the acquired
    field of view of the b0 (where it is exactly zero) is ignored. The b0 is then registered to it
    with symmetric diffeomorphic registration (SyN) and the local
    cross-correlation metric, with the deformation restricted to the
    phase-encoding axis :footcite:p:`Avants2008`, :footcite:p:`Wang2017`. The
    rigid alignment is re-estimated on the corrected b0 ``n_outer`` times, so
    that the distortion does not bias the coregistration. The estimated
    displacement is finally applied to every volume of ``dwi``.

    Parameters
    ----------
    dwi : ndarray (X, Y, Z) or (X, Y, Z, N)
        Distorted EPI data. If 4D, the volume ``b0_index`` is used to estimate
        the distortion and the correction is applied to all volumes.
    dwi_affine : ndarray (4, 4)
        Voxel-to-world affine of ``dwi``.
    anat : ndarray (X', Y', Z')
        Undistorted anatomical image of the same subject. It should be
        skull-stripped, or a brain mask should be given in ``anat_mask``.
    anat_affine : ndarray (4, 4)
        Voxel-to-world affine of ``anat``.
    pe_axis : int or str
        Voxel axis of ``dwi`` along which phase encoding was performed:
        0, 1, 2 or one of 'i', 'j', 'k' (the polarity is irrelevant).
    anat_contrast : {'t1', 't2'}, optional
        Contrast of ``anat``. A T1-weighted image is intensity-inverted inside
        the brain mask to resemble a b0. The intensities of a T2-weighted
        image are mapped to the mean b0 intensity observed for them inside the
        brain, re-estimated as the correction improves.
    anat_mask : ndarray (X', Y', Z'), optional
        Brain mask on the ``anat`` grid. Defaults to ``anat > 0``.
    b0_index : int, optional
        Index of the b0 volume of a 4D ``dwi`` used to estimate the
        distortion.
    n_outer : int, optional
        Number of alternations between rigid alignment and SyN.
    level_iters : sequence of int, optional
        SyN iterations at each resolution, coarsest first.
    step_length : float, optional
        Maximum SyN update per iteration, in voxels.
    radius : int, optional
        Radius of the cross-correlation neighborhood, in voxels.
    sigma_diff : float, optional
        Standard deviation (in voxels) of the Gaussian smoothing of the SyN
        updates. Larger values give a smoother field.
    ss_sigma_factor : float, optional
        Smoothing factor of the SyN scale space.
    mask_dilation : int, optional
        Dilation (in voxels) of the brain mask applied to the b0, so that
        displaced brain boundaries remain inside the similarity region.
    field_smooth : float, optional
        Standard deviation (in mm) of the Gaussian smoothing applied to the
        final displacement. 0 disables it.
    return_field : bool, optional
        Also return the displacement along ``pe_axis``.
    return_mapping : bool, optional
        Also return the :class:`~dipy.align.imwarp.DiffeomorphicMap`, which
        can be used to correct other images sharing the distortion of ``dwi``.

    Returns
    -------
    corrected : ndarray
        Distortion-corrected ``dwi``, with the same shape.
    field : ndarray (X, Y, Z)
        Displacement, in voxels, along ``pe_axis``: the corrected image at
        voxel ``x`` is the input sampled at ``x + field[x]`` along
        ``pe_axis``. Only returned if ``return_field`` is True.
    mapping : DiffeomorphicMap
        Only returned if ``return_mapping`` is True.

    References
    ----------
    .. footbibliography::
    """
    dwi = np.asarray(dwi)
    anat = np.asarray(anat, dtype=np.float64)
    if dwi.ndim not in (3, 4):
        raise ValueError(f"dwi must be 3D or 4D, got shape {dwi.shape}.")
    if anat.ndim != 3:
        raise ValueError(f"anat must be 3D, got shape {anat.shape}.")
    if dwi.ndim == 4 and not 0 <= b0_index < dwi.shape[-1]:
        raise ValueError(
            f"b0_index {b0_index} out of range for {dwi.shape[-1]} volumes."
        )
    contrast = str(anat_contrast).lower()
    if contrast not in ("t1", "t2"):
        raise ValueError("anat_contrast must be 't1' or 't2'.")
    pe_axis = _pe_axis_index(pe_axis)
    if n_outer < 1:
        raise ValueError("n_outer must be at least 1.")

    b0 = np.asarray(dwi[..., b0_index] if dwi.ndim == 4 else dwi, dtype=np.float64)
    if anat_mask is None:
        anat_mask = anat > 0
    anat_mask = np.asarray(anat_mask) > 0
    if not anat_mask.any():
        raise ValueError("The anatomical brain mask is empty.")

    zooms = np.sqrt((dwi_affine[:3, :3] ** 2).sum(0))
    direction = dwi_affine[:3, pe_axis]
    corrected_b0 = b0
    # Anatomy outside the acquired EPI field of view has no counterpart in
    # the b0 and must not attract the registration.
    fov = _epi_fov(b0)
    rigid = None
    mapping = None
    for outer in range(n_outer):
        rigid = _rigid_anat_to_epi(
            corrected_b0,
            dwi_affine,
            anat,
            anat_affine,
            anat_mask,
            None if rigid is None else rigid.affine,
        )
        mask = (rigid.transform(anat_mask.astype(np.float64)) > 0.5) & fov
        ref = _anat_reference(rigid.transform(anat), mask, contrast, corrected_b0)
        support = binary_dilation(mask, iterations=mask_dilation)
        static = _scale(ref, mask)
        moving = _scale(np.where(support, b0, 0.0), mask)

        sdr = PhaseEncodingSyN(
            CCMetric(3, sigma_diff=sigma_diff, radius=radius),
            direction,
            level_iters=list(level_iters),
            step_length=step_length,
            ss_sigma_factor=ss_sigma_factor,
        )
        mapping = sdr.optimize(
            static, moving, static_grid2world=dwi_affine, moving_grid2world=dwi_affine
        )
        corrected_b0 = mapping.transform(b0)
        logger.info(f"SyN-SDC iteration {outer + 1}/{n_outer} done.")

    forward = np.asarray(mapping.get_forward_field(), dtype=np.float64)
    # Displacements are world vectors parallel to ``direction``; express them
    # in voxels along the PE axis.
    field = forward @ direction / zooms[pe_axis] ** 2
    if field_smooth > 0:
        field = gaussian_filter(field, field_smooth / zooms)
    mapping = _pe_mapping(field, direction, dwi_affine, b0.shape)

    if dwi.ndim == 4:
        corrected = np.stack(
            [mapping.transform(dwi[..., i]) for i in range(dwi.shape[-1])], axis=-1
        )
    else:
        corrected = mapping.transform(dwi)
    corrected = corrected.astype(dwi.dtype, copy=False)

    out = (corrected,)
    if return_field:
        out += (field.astype(np.float32),)
    if return_mapping:
        out += (mapping,)
    return out[0] if len(out) == 1 else out


def _pe_mapping(disp_vox, axis_vector, affine, shape):
    """DiffeomorphicMap of a displacement given in voxels along one axis.

    ``axis_vector`` is the world-space step of one voxel along that axis
    (the corresponding column of ``affine``).
    """
    forward = np.ascontiguousarray(disp_vox[..., None] * axis_vector, dtype=np.float32)
    mapping = DiffeomorphicMap(
        3,
        shape,
        disp_grid2world=affine,
        domain_shape=shape,
        domain_grid2world=affine,
        codomain_shape=shape,
        codomain_grid2world=affine,
    )
    mapping.forward = forward
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(0))
    mapping.backward = np.asarray(
        vfu.invert_vector_field_fixed_point_3d(
            forward, mapping.disp_world2grid, spacing, 40, 1e-4
        )
    )
    return mapping
