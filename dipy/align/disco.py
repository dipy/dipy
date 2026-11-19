"""Tools for correcting distortions in diffusion MRI data, including
susceptibility-induced distortions using the Synb0-SyN method
:footcite:p:`Chigurupati2024`, :footcite:p:`Schilling2019`,
:footcite:p:`Schilling2020`.

References
----------
.. footbibliography::
"""

from pathlib import Path

from nibabel.affines import voxel_sizes
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

from dipy.align import affine_registration
from dipy.align.bbr import bbr_registration
from dipy.align.imaffine import AffineMap
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
from dipy.align.reslice import reslice
from dipy.data import get_fnames
from dipy.io.image import load_nifti, save_nifti
from dipy.nn.deepn4 import DeepN4
from dipy.nn.mindgrab import MindGrab
from dipy.nn.synb0 import Synb0
from dipy.nn.utils import normalize
from dipy.utils.logging import logger
from dipy.utils.volume import select_3d_volume


def _limit_resolution(image, affine, *, mask=None, zoom=1.0):
    """Resample an image, and its mask, that is finer than `zoom` mm.

    Parameters
    ----------
    image : ndarray (X, Y, Z)
        Image to resample.
    affine : ndarray (4, 4)
        Voxel-to-world matrix of `image`.
    mask : ndarray (X, Y, Z), optional
        Binary mask on the grid of `image`.
    zoom : float, optional
        Smallest voxel size, in mm, to keep. Coarser axes are not changed.

    Returns
    -------
    image : ndarray
        Resampled image, or `image` itself when no axis is finer than `zoom`.
    mask : ndarray or None
        Resampled boolean mask, or `mask` itself.
    affine : ndarray (4, 4)
        Voxel-to-world matrix of the returned image.
    """
    zooms = voxel_sizes(affine)
    if np.all(zooms >= zoom - 1e-3):
        return image, mask, affine
    new_zooms = tuple(np.maximum(zooms, zoom))
    resampled, new_affine = reslice(image, affine, zooms, new_zooms=new_zooms)
    if mask is not None:
        mask, _ = reslice(
            np.asarray(mask, dtype=np.float32),
            affine,
            zooms,
            new_zooms=new_zooms,
            order=0,
        )
        mask = mask > 0.5
    return resampled, mask, new_affine


def _warp_image(mapping, image, *, num_threads=None):
    """Apply a deformation mapping to a 3D or 4D image.

    Parameters
    ----------
    mapping : DiffeomorphicMap
        Mapping whose ``transform`` method warps `image`.
    image : ndarray
        Input array, either 3D (X, Y, Z) or 4D (X, Y, Z, N).
    num_threads : int, optional
        Number of OpenMP threads used for warping. If None, DIPY's default
        thread count is used.

    Returns
    -------
    warped : ndarray
        Warped image with the same shape as `image`.

    Raises
    ------
    ValueError
        If `image` is not 3D or 4D.
    """
    image = np.asarray(image)

    if image.ndim == 3:
        return mapping.transform(image, num_threads=num_threads)

    if image.ndim != 4:
        raise ValueError(
            f"DWI image must be a 3D or 4D array. Got shape {image.shape}."
        )

    warped_0 = mapping.transform(image[..., 0], num_threads=num_threads)
    warped = np.empty(image.shape, dtype=warped_0.dtype)
    warped[..., 0] = warped_0

    for vol_idx in range(1, image.shape[-1]):
        warped[..., vol_idx] = mapping.transform(
            image[..., vol_idx], num_threads=num_threads
        )

    return warped


def _pe_direction(pe_axis, dwi_affine):
    """World direction of the phase-encoding voxel axis of the DWI.

    Parameters
    ----------
    pe_axis : int or str
        Voxel axis of the DWI along which the phase is encoded: 0, 1, 2, or
        a BIDS ``PhaseEncodingDirection`` ('i', 'j', 'k', 'i-', 'j-', 'k-');
        'x', 'y', 'z' are accepted as synonyms. The sign is ignored.
    dwi_affine : ndarray (4, 4)
        Voxel-to-world matrix of the DWI.

    Returns
    -------
    pe_axis : int
        The voxel axis as an integer.
    direction : ndarray (3,)
        Unit vector, in world coordinates, pointing along increasing voxel
        index on that axis.

    Raises
    ------
    ValueError
        If `pe_axis` is not a valid axis.
    """
    if isinstance(pe_axis, str):
        pe_axis = {"i": 0, "j": 1, "k": 2, "x": 0, "y": 1, "z": 2}.get(
            pe_axis.lower().removesuffix("-")
        )
    if isinstance(pe_axis, bool) or pe_axis not in (0, 1, 2):
        raise ValueError("pe_axis must be 0, 1, 2 or 'i', 'j', 'k' (or 'i-', ...).")
    direction = np.asarray(dwi_affine, dtype=np.float64)[:3, pe_axis]
    return int(pe_axis), direction / np.linalg.norm(direction)


def _restrict_field_to_pe(field, direction, sigma):
    """Keep the component of a displacement field along one direction.

    Parameters
    ----------
    field : ndarray (X, Y, Z, 3)
        Displacement field in world coordinates (mm).
    direction : ndarray (3,)
        Unit vector in world coordinates.
    sigma : float or sequence of float
        Standard deviation, in voxels, of the Gaussian smoothing applied to
        the kept component.

    Returns
    -------
    field : ndarray (X, Y, Z, 3)
        Field with displacements along `direction` only.
    pe_disp : ndarray (X, Y, Z)
        Signed displacement (mm) along `direction`.
    """
    direction = np.asarray(direction, dtype=field.dtype)
    pe_disp = gaussian_filter(field @ direction, sigma=sigma)
    return pe_disp[..., None] * direction, pe_disp


def _set_forward_field(mapping, field):
    """Replace the field that ``mapping.transform`` uses.

    ``SymmetricDiffeomorphicRegistration.optimize`` returns an inverted map
    (``is_inverse=True``): its ``transform`` reads ``mapping.backward``, not
    ``mapping.forward``. Only the field of ``transform`` is replaced, so
    ``transform_inverse`` no longer matches it.

    Parameters
    ----------
    mapping : DiffeomorphicMap
        Map to modify in place.
    field : ndarray (X, Y, Z, 3)
        Displacement field, in world coordinates (mm), to use when warping
        with ``mapping.transform``.
    """
    field = np.asarray(field, dtype=np.float32)
    if mapping.is_inverse:
        mapping.backward = field
    else:
        mapping.forward = field


def _refine_b0_to_t1(
    b0, t1, dwi_affine, t1_affine, brain_mask, starting_affine, *, scaling=False
):
    """Refine the b0 to T1 alignment with a boundary-based registration.

    Parameters
    ----------
    b0 : ndarray (X, Y, Z)
        The b0 volume.
    t1 : ndarray (X, Y, Z)
        The T1-weighted image.
    dwi_affine : ndarray (4, 4)
        Voxel-to-world matrix of `b0`.
    t1_affine : ndarray (4, 4)
        Voxel-to-world matrix of `t1`.
    brain_mask : ndarray (X, Y, Z)
        Brain mask on the grid of `t1`.
    starting_affine : ndarray (4, 4)
        Initial transform from the world coordinates of `t1` to those of
        `b0`.
    scaling : bool, optional
        If True, a scaling along each axis is also estimated.

    Returns
    -------
    affine : ndarray (4, 4)
        The refined transform, or `starting_affine` when the boundary-based
        registration cannot be run (for instance when the white matter
        cannot be segmented).
    """
    try:
        _, affine = bbr_registration(
            b0,
            t1,
            dwi_affine,
            t1_affine,
            static_mask=brain_mask,
            starting_affine=starting_affine,
            scaling=scaling,
        )
    except ValueError as error:
        logger.warning(
            f"The boundary-based registration failed: {error} The b0 to T1 "
            "alignment is not refined."
        )
        return starting_affine
    return affine


def _normalize_t1_for_synb0(t1):
    """Rescale a T1 image to match the intensity distribution expected by Synb0.

    The Synb0 model was trained on T1 images normalised to a scale which
    places white-matter around 110. The subsequent model normalisation clips
    at 150 and maps to [-1, 1]. Feeding a raw T1 (values up to ~3000) causes
    everything to clip, destroying all contrast.

    The white-matter peak is estimated as the 90th percentile of brain voxels
    (voxels above the 5th percentile of non-zero values).

    Parameters
    ----------
    t1 : ndarray
        Input T1-weighted image with arbitrary intensity scale.

    Returns
    -------
    t1_normalized : ndarray
        Rescaled T1 image with white-matter intensity near 110.
    """
    brain = t1[t1 > np.percentile(t1[t1 > 0], 5)] if np.any(t1 > 0) else t1.ravel()
    wm_proxy = np.percentile(brain, 90)
    if wm_proxy < 1e-6:
        return t1.copy()
    return t1 * (110.0 / wm_proxy)


def _save_debug_volume(debug_dir, name, data, affine):
    """Save an array as a NIfTI file for debugging, if debug_dir is set.

    Parameters
    ----------
    debug_dir : str, Path, or None
        Directory where the file will be saved. If None, does nothing.
    name : str
        Base name of the output file (without extension).
    data : ndarray
        Array to save.
    affine : ndarray (4, 4)
        Affine transformation matrix for the NIfTI file.
    """
    if debug_dir is None:
        return
    arr = np.asarray(data)
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
    save_nifti(Path(debug_dir) / f"{name}.nii.gz", arr, affine)


def synb0_syn(
    dwi,
    T1,
    dwi_affine,
    T1_affine,
    *,
    b0_index=0,
    dwi_mask=None,
    T1_mask=None,
    pe_axis=1,
    return_field=False,
    apply_deepn4=False,
    apply_skull_strip=False,
    b0_to_t1_method="bbr",
    debug_dir=None,
    verbose=True,
    num_threads=None,
    **kwargs,
):
    """
    Perform Synb0-SyN distortion correction on a diffusion MRI dataset.

    This function synthesizes an undistorted b0 image using the Synb0
    deep learning model and applies nonrigid registration to correct
    distortions in the diffusion MRI data. It provides a simplified
    interface to the full Synb0-SyN pipeline.

    Parameters
    ----------
    dwi : ndarray (X, Y, Z, N) or (X, Y, Z)
        The input diffusion MRI data. If 4D, `b0_index` is used to select
        the volume for estimating the distortion field, and the estimated
        field is then applied to all volumes.
    T1 : ndarray (X, Y, Z) or (X, Y, Z, N)
        The T1-weighted structural image. If 4D, its first volume is used.
        Axes finer than 1 mm are resampled to 1 mm.
    dwi_affine : ndarray (4, 4)
        Affine transformation matrix for the DWI image.
    T1_affine : ndarray (4, 4)
        Affine transformation matrix for the T1 image.
    b0_index : int, optional
        The index of the b0 volume used to estimate the field when `dwi` (and
        `dwi_mask`) are 4D. Ignored when `dwi` is 3D.
    dwi_mask : ndarray (X, Y, Z), optional
        A binary mask for the DWI data. If None, no masking is applied.
    T1_mask : ndarray (X, Y, Z), optional
        A binary mask for the T1 image. If None, no masking is applied.
    pe_axis : int or str, optional
        The phase-encoding axis, as a voxel axis of `dwi` (not a world axis):
        0, 1, 2, or a BIDS ``PhaseEncodingDirection`` such as 'j' or 'j-'
        (the sign is ignored; 'x', 'y', 'z' are accepted as synonyms). The
        field is restricted to the direction of that axis, which `dwi_affine`
        gives in world coordinates. If None, the field is not restricted.
    return_field : bool, optional
        Whether to return the estimated distortion field along with the
        corrected DWI data.
    apply_deepn4 : bool, optional
        If True, apply DeepN4 bias field correction to the T1 image
        before the first affine registration. DeepN4 requires PyTorch or
        TensorFlow.
    apply_skull_strip : bool, optional
        If True, apply MindGrab skull stripping to both the T1 and DWI b0
        images before registration. The resulting masks are used as
        ``T1_mask`` and ``dwi_mask`` respectively.
    b0_to_t1_method : {'bbr', 'bbr_scaling', 'mi'}, optional
        How the b0 is aligned to the T1. 'mi' is a rigid registration that
        maximizes mutual information. 'bbr' refines it with a rigid
        boundary-based registration :footcite:p:`Greve2009`, which is more
        robust to the distortion of the b0. 'bbr_scaling' also estimates a
        scaling along each axis: use it when the T1 and the DWI do not share
        the same geometry (for instance when only the T1 was corrected for
        gradient nonlinearity), otherwise the difference in size ends up in
        the estimated field.
    debug_dir : str or Path, optional
        Directory to save intermediate debug NIfTI volumes.
    verbose : bool, optional
        Whether the MindGrab, DeepN4 and Synb0 models log their progress.
    num_threads : int, optional
        Number of threads used by the registrations, MindGrab, Synb0 and the
        final warping. If None, the default of each step is used; a negative
        value uses all the threads minus $|num_threads + 1|$.
    **kwargs :
        Additional keyword arguments to pass to Synb0.predict.

    Returns
    -------
    corrected_dwi : ndarray
        Distortion-corrected DWI image. Same dimensionality as `dwi` (3D or
        4D). If `dwi` is 4D, the field estimated from `dwi[..., b0_index]` is
        applied to all volumes.
    field : ndarray
        Estimated deformation field. Only returned when `return_field=True`.
        If `pe_axis` is not None, shape is (X, Y, Z) and the values are the
        displacements in mm along the phase-encoding voxel axis, positive
        towards increasing voxel index; otherwise shape is (X, Y, Z, 3), in
        world coordinates.

    References
    ----------
    .. footbibliography::
    """
    dwi = np.asarray(dwi)
    T1 = np.asarray(T1)

    if b0_to_t1_method not in ("bbr", "bbr_scaling", "mi"):
        raise ValueError(
            "b0_to_t1_method must be 'bbr', 'bbr_scaling' or 'mi', got "
            f"{b0_to_t1_method!r}."
        )
    if dwi.ndim not in (3, 4):
        raise ValueError(f"dwi must be a 3D or 4D array. Got shape {dwi.shape}.")
    if pe_axis is not None:
        pe_axis, pe_direction = _pe_direction(pe_axis, dwi_affine)
    if T1.ndim not in (3, 4):
        raise ValueError(f"T1 must be a 3D or 4D array. Got shape {T1.shape}.")

    dwi_for_field = select_3d_volume(dwi, index=b0_index, name="dwi")
    T1_for_reg = select_3d_volume(T1, name="T1")

    if dwi_mask is not None:
        dwi_mask = select_3d_volume(dwi_mask, index=b0_index, name="dwi_mask")
    if T1_mask is not None:
        T1_mask = select_3d_volume(T1_mask, name="T1_mask")
    # Registration and Synb0 do not use sub-millimetre detail, and a 0.5 mm
    # T1 has 8 times more voxels than at 1 mm.
    T1_for_reg, T1_mask, T1_affine = _limit_resolution(
        T1_for_reg, T1_affine, mask=T1_mask
    )

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    _save_debug_volume(
        debug_dir=debug_dir,
        name="00_dwi_b0_input",
        data=dwi_for_field,
        affine=dwi_affine,
    )
    _save_debug_volume(
        debug_dir=debug_dir, name="01_t1_input", data=T1_for_reg, affine=T1_affine
    )

    _, _, mni_mask_path, mni_t1_path = get_fnames(name="mni_templates")
    mni_t1, mni_t1_affine = load_nifti(mni_t1_path)
    mni_mask, _ = load_nifti(mni_mask_path)
    mni_mask = mni_mask.astype(np.int32)

    # ── Optional: MindGrab skull stripping on T1 and DWI ─────────────────
    if apply_skull_strip:
        logger.info("Applying MindGrab skull stripping to T1 and DWI b0...")
        mindgrab_model = MindGrab(verbose=verbose, num_threads=num_threads)
        T1_mask = mindgrab_model.predict(T1_for_reg, T1_affine).astype(bool)
        dwi_mask = mindgrab_model.predict(dwi_for_field, dwi_affine).astype(bool)
        _save_debug_volume(
            debug_dir=debug_dir,
            name="00a_t1_skull_strip_mask",
            data=T1_mask,
            affine=T1_affine,
        )
        _save_debug_volume(
            debug_dir=debug_dir,
            name="00a_dwi_skull_strip_mask",
            data=dwi_mask,
            affine=dwi_affine,
        )
    if dwi_mask is not None:
        dwi_for_field = dwi_for_field * dwi_mask
    if T1_mask is not None:
        T1_for_reg = T1_for_reg * T1_mask
    # ── Optional: DeepN4 bias field correction on T1 ────────────────────
    if apply_deepn4:
        logger.info("Applying DeepN4 bias field correction to T1...")
        deepn4_model = DeepN4(verbose=verbose)
        T1_for_reg = deepn4_model.predict(T1_for_reg, T1_affine)
        _save_debug_volume(
            debug_dir=debug_dir,
            name="00b_t1_deepn4_corrected",
            data=T1_for_reg,
            affine=T1_affine,
        )

    # ── Step 1: Affine T1 → MNI T1 (sparse sampling for speed) ──────────
    logger.info("Performing affine registration of T1 to MNI template...")
    T1_affine_to_template, t1_reg_affine = affine_registration(
        T1_for_reg,
        mni_t1,
        moving_affine=T1_affine,
        static_affine=mni_t1_affine,
        nbins=32,
        num_threads=num_threads,
        metric="MI",
        pipeline=["center_of_mass", "translation", "rigid", "affine"],
        level_iters=[100, 50, 25],
        sigmas=[4.0, 2.0, 1.0],
        factors=[8, 4, 2],
        static_mask=mni_mask,
        moving_mask=T1_mask.astype(bool) if T1_mask is not None else None,
    )
    _save_debug_volume(
        debug_dir=debug_dir,
        name="01d_t1_affine_to_template",
        data=T1_affine_to_template,
        affine=mni_t1_affine,
    )
    t1_affine_map = AffineMap(
        t1_reg_affine,
        domain_grid_shape=mni_t1.shape,
        domain_grid2world=mni_t1_affine,
        codomain_grid_shape=T1_for_reg.shape,
        codomain_grid2world=T1_affine,
    )
    if T1_mask is not None:
        T1_mask_f = np.ascontiguousarray(np.asarray(T1_mask, dtype=np.float32))
        T1_mask_in_template = (
            t1_affine_map.transform(T1_mask_f, interpolation="nearest") > 0.5
        )
        masked_T1_reg_to_template = T1_affine_to_template * T1_mask_in_template
    else:
        masked_T1_reg_to_template = T1_affine_to_template

    # ── Step 2: Rigid b0 → T1, compose with T1 → MNI ────────────────────
    logger.info("Performing rigid registration of DWI b0 to T1...")
    dwi_p99 = np.percentile(dwi_for_field, 99)
    dwi_norm = normalize(dwi_for_field, min_v=0, max_v=dwi_p99, new_min=0, new_max=100)
    t1_p99 = np.percentile(T1_for_reg, 99)
    t1_norm = normalize(T1_for_reg, min_v=0, max_v=t1_p99, new_min=0, new_max=100)
    # NOTE: masks are intentionally NOT passed here. ``dwi_norm``/``t1_norm`` are
    # already computed from the pre-masked ``dwi_for_field``/``T1_for_reg``, and
    # additionally restricting the MI metric to the mask drives this rigid step to
    # a grossly wrong optimum (a ~150 mm translation that pushes the b0 out of the
    # template), leaving the b0-in-template and the back-warped synthetic b0 empty.
    _, b0_to_t1_affine = affine_registration(
        dwi_norm,
        t1_norm,
        moving_affine=dwi_affine,
        static_affine=T1_affine,
        nbins=32,
        num_threads=num_threads,
        metric="MI",
        pipeline=["center_of_mass", "translation", "rigid"],
        level_iters=[100, 50, 25],
        sigmas=[4.0, 2.0, 1.0],
        factors=[8, 4, 2],
    )
    if b0_to_t1_method in ("bbr", "bbr_scaling"):
        logger.info("Refining the b0 to T1 alignment with BBR...")
        if T1_mask is not None:
            brain_mask = np.asarray(T1_mask) > 0
        else:
            # Brain mask of the template, pulled back to the T1.
            brain_mask = (
                t1_affine_map.transform_inverse(
                    mni_mask.astype(np.float32), interpolation="nearest"
                )
                > 0.5
            )
        b0_to_t1_affine = _refine_b0_to_t1(
            dwi_for_field,
            T1_for_reg,
            dwi_affine,
            T1_affine,
            brain_mask,
            b0_to_t1_affine,
            scaling=b0_to_t1_method == "bbr_scaling",
        )
    # Template world -> T1 world -> DWI world.
    dwi_reg_affine = b0_to_t1_affine @ t1_reg_affine

    dwi_to_mni_map = AffineMap(
        dwi_reg_affine,
        domain_grid_shape=mni_t1.shape,
        domain_grid2world=mni_t1_affine,
        codomain_grid_shape=dwi_for_field.shape,
        codomain_grid2world=dwi_affine,
    )
    dwi_affine_to_template = dwi_to_mni_map.transform(dwi_for_field)
    _save_debug_volume(
        debug_dir=debug_dir,
        name="04_b0_affine_to_template",
        data=dwi_affine_to_template,
        affine=mni_t1_affine,
    )

    if dwi_mask is not None:
        dwi_mask_f = np.ascontiguousarray(np.asarray(dwi_mask, dtype=np.float32))
        dwi_mask_in_template = (
            dwi_to_mni_map.transform(dwi_mask_f, interpolation="nearest") > 0.5
        )
        masked_dwi_reg_to_template = dwi_affine_to_template * dwi_mask_in_template
    else:
        dwi_mask_f = None
        masked_dwi_reg_to_template = dwi_affine_to_template

    # ── Step 3: Reslice to 2.5 mm for Synb0 ─────────────────────────────
    synb0_shape = (77, 91, 77)
    synb0_zooms = (2.5, 2.5, 2.5)
    mni_t1_zooms = tuple(voxel_sizes(mni_t1_affine))

    masked_T1_reg_to_template = _normalize_t1_for_synb0(masked_T1_reg_to_template)

    masked_T1_reg_resized, resized_affine = reslice(
        masked_T1_reg_to_template,
        mni_t1_affine,
        mni_t1_zooms,
        new_zooms=synb0_zooms,
        new_shape=synb0_shape,
    )
    masked_dwi_reg_resized, _ = reslice(
        masked_dwi_reg_to_template,
        mni_t1_affine,
        mni_t1_zooms,
        new_zooms=synb0_zooms,
        new_shape=synb0_shape,
    )
    mni_mask_resized, _ = reslice(
        mni_mask.astype(np.float32),
        mni_t1_affine,
        mni_t1_zooms,
        new_zooms=synb0_zooms,
        new_shape=synb0_shape,
        order=0,
    )
    mni_mask_resized = (mni_mask_resized > 0.5).astype(np.int32)

    # ── Step 4: Synb0 synthesis ──────────────────────────────────────────
    synb0 = Synb0(verbose=verbose, num_threads=num_threads)
    logger.info("Synthesizing undistorted b0 with Synb0 model...")
    binf = synb0.predict(masked_dwi_reg_resized, masked_T1_reg_resized, **kwargs)
    binf = binf * mni_mask_resized
    _save_debug_volume(
        debug_dir=debug_dir, name="06_binf_template", data=binf, affine=resized_affine
    )

    # ── Step 5: Bring synthetic b0 back to native DWI space ──────────────
    dwi_inv_affine_map = AffineMap(
        dwi_reg_affine,
        domain_grid_shape=synb0_shape,
        domain_grid2world=resized_affine,
        codomain_grid_shape=dwi_for_field.shape,
        codomain_grid2world=dwi_affine,
    )
    ori_binf = dwi_inv_affine_map.transform_inverse(binf)
    _save_debug_volume(
        debug_dir=debug_dir, name="06b_binf_in_dwi", data=ori_binf, affine=dwi_affine
    )
    ori_binf = np.clip(ori_binf, a_min=0, a_max=None)
    if dwi_mask_f is not None:
        ori_binf = ori_binf * dwi_mask_f
    _save_debug_volume(
        debug_dir=debug_dir, name="07_binf_in_dwi", data=ori_binf, affine=dwi_affine
    )

    # ── Step 6: SyN registration (synthetic b0 ↔ distorted b0) ──────────
    logger.info("Performing nonlinear registration of synthesized b0 to DWI...")
    if dwi_mask_f is not None:
        masked_dwi_for_sdr = dwi_for_field * dwi_mask_f
    else:
        masked_dwi_for_sdr = dwi_for_field

    dwi_zooms = voxel_sizes(dwi_affine)

    binf_coverage = binary_dilation(ori_binf > 0, iterations=5).astype(np.float32)
    binf_coverage = gaussian_filter(binf_coverage, sigma=3.0)
    masked_dwi_for_sdr = masked_dwi_for_sdr * binf_coverage

    p99_s = np.percentile(ori_binf[ori_binf > 0], 99) if np.any(ori_binf > 0) else 1.0
    p99_m = (
        np.percentile(masked_dwi_for_sdr[masked_dwi_for_sdr > 0], 99)
        if np.any(masked_dwi_for_sdr > 0)
        else 1.0
    )
    static_norm = ori_binf / max(p99_s, 1e-8) * 100.0
    moving_norm = masked_dwi_for_sdr / max(p99_m, 1e-8) * 100.0

    sdr = SymmetricDiffeomorphicRegistration(
        metric=CCMetric(3),
        level_iters=[200, 200, 100],
        step_length=0.25,
        ss_sigma_factor=0.1,
        opt_tol=1e-6,
        num_threads=num_threads,
    )
    mapping = sdr.optimize(
        static=static_norm,
        moving=moving_norm,
        static_grid2world=dwi_affine,
        moving_grid2world=dwi_affine,
    )

    # ── Step 7: Extract PE-restricted field and post-smooth ──────────────
    field_smooth_mm = 0.5
    field_smooth_vox = field_smooth_mm / dwi_zooms

    if pe_axis is not None:
        # The field is in world coordinates, the phase encoding is along a
        # voxel axis: project on the world direction of that axis.
        warp_field, field = _restrict_field_to_pe(
            mapping.get_forward_field(), pe_direction, field_smooth_vox
        )
    else:
        warp_field = gaussian_filter(
            mapping.get_forward_field(), sigma=(*field_smooth_vox, 0)
        )
        field = warp_field
    _set_forward_field(mapping, warp_field)

    # ── Step 8: Apply correction to DWI ──────────────────────────────────
    dwi_to_binf = _warp_image(mapping=mapping, image=dwi, num_threads=num_threads)
    _save_debug_volume(
        debug_dir=debug_dir,
        name="08_dwi_corrected_b0",
        data=select_3d_volume(dwi_to_binf, index=b0_index, name="dwi_corrected"),
        affine=dwi_affine,
    )

    if return_field:
        _save_debug_volume(
            debug_dir=debug_dir, name="09_final_field", data=field, affine=dwi_affine
        )
        return dwi_to_binf, field
    else:
        return dwi_to_binf
