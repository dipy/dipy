"""Tools for correcting distortions in diffusion MRI data, including
susceptibility-induced distortions using the Synb0-SyN method
:footcite:p:`Chigurupati2024`, :footcite:p:`Schilling2019`,
:footcite:p:`Schilling2020`.

References
----------
.. footbibliography::
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

from dipy.align import affine_registration
from dipy.align.imaffine import AffineMap
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
from dipy.align.reslice import reslice
from dipy.data import get_fnames
from dipy.io.image import load_nifti, save_nifti
from dipy.nn.utils import normalize
from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package

_, _have_torch, _ = optional_package("torch", min_version="2.2.0")
_, _have_tf, _ = optional_package("tensorflow", min_version="2.18.0")

try:
    from dipy.nn.synb0 import Synb0

    HAVE_SYNB0 = _have_torch or _have_tf
    if not HAVE_SYNB0:
        logger.warning(
            "Synb0 model not available. Install PyTorch or TensorFlow to use. "
            "(pip install dipy[ml]) "
            "Synb0-DISCO distortion correction."
        )
except ImportError:
    HAVE_SYNB0 = False
    logger.warning(
        "Synb0 model not available. Install PyTorch or TensorFlow to use. "
        "(pip install dipy[ml]) "
        "Synb0-DISCO distortion correction."
    )


def _validate_b0_index(b0_index, n_volumes, image_name):
    """Validate that b0_index is a valid integer index into a volume array.

    Parameters
    ----------
    b0_index : int
        Index to validate.
    n_volumes : int
        Total number of volumes in the image.
    image_name : str
        Name of the image (used in error messages).

    Returns
    -------
    b0_index : int
        The validated index as a Python int.

    Raises
    ------
    TypeError
        If b0_index is not an integer.
    ValueError
        If b0_index is out of range [0, n_volumes - 1].
    """
    if not isinstance(b0_index, (int, np.integer)):
        raise TypeError("b0_index must be an integer.")

    if b0_index < 0 or b0_index >= n_volumes:
        raise ValueError(
            f"b0_index ({b0_index}) must be in [0, {n_volumes - 1}] "
            f"for {image_name} with {n_volumes} volumes."
        )

    return int(b0_index)


def _select_3d_volume(image, image_name, b0_index):
    """Select a 3D volume from a 3D or 4D array.

    Parameters
    ----------
    image : ndarray
        Input array, either 3D (X, Y, Z) or 4D (X, Y, Z, N).
    image_name : str
        Name of the image (used in error messages).
    b0_index : int
        Index of the volume to select when `image` is 4D.

    Returns
    -------
    volume : ndarray (X, Y, Z)
        Selected 3D volume.

    Raises
    ------
    ValueError
        If `image` is not 3D or 4D.
    """
    image = np.asarray(image)

    if image.ndim == 3:
        return image

    if image.ndim != 4:
        raise ValueError(
            f"{image_name} must be a 3D or 4D array. Got shape {image.shape}."
        )

    idx = _validate_b0_index(
        b0_index=b0_index, n_volumes=image.shape[-1], image_name=image_name
    )
    return image[..., idx]


def _warp_image(mapping, image):
    """Apply a deformation mapping to a 3D or 4D image.

    Parameters
    ----------
    mapping : object
        A mapping object with a ``transform(image)`` method.
    image : ndarray
        Input array, either 3D (X, Y, Z) or 4D (X, Y, Z, N).

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
        return mapping.transform(image)

    if image.ndim != 4:
        raise ValueError(
            f"DWI image must be a 3D or 4D array. Got shape {image.shape}."
        )

    warped_0 = mapping.transform(image[..., 0])
    warped = np.empty(image.shape, dtype=warped_0.dtype)
    warped[..., 0] = warped_0

    for vol_idx in range(1, image.shape[-1]):
        warped[..., vol_idx] = mapping.transform(image[..., vol_idx])

    return warped


def _normalize_t1_for_synb0(t1):
    """
    The Synb0 model was trained on T1 images normalised to a scale
     which places white-matter around 110.
    The subsequent model normalisation clips at 150 and maps to [-1, 1].
    Feeding a raw T1 (values up to ~3000) causes everything to clip,
    destroying all contrast.

    This function rescales the T1 so that the white-matter peak (estimated
    as the 90th percentile of brain voxels) sits at 110, matching the
    expected distribution.
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
    debug_dir=None,
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
        The T1-weighted structural image. If 4D, `b0_index` is used to
        select the volume for registration.
    dwi_affine : ndarray (4, 4)
        Affine transformation matrix for the DWI image.
    T1_affine : ndarray (4, 4)
        Affine transformation matrix for the T1 image.
    b0_index : int, optional
        The index of the volume used for field estimation/registration when
        `dwi` and/or `T1` are 4D. Must be less than the number of volumes in
        each 4D input.
    dwi_mask : ndarray (X, Y, Z), optional
        A binary mask for the DWI data. If None, no masking is applied.
    T1_mask : ndarray (X, Y, Z), optional
        A binary mask for the T1 image. If None, no masking is applied.
    pe_axis : int or str, optional
        The phase-encoding axis. Can be specified as an integer (0 for 'x',
        1 for 'y', 2 for 'z') or a string ('x', 'y', 'z').
        If provided, the returned field will be restricted to this direction.
    return_field : bool, optional
        Whether to return the estimated distortion field along with the
        corrected DWI data.
    debug_dir : str or Path, optional
        Directory to save intermediate debug NIfTI volumes.
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
        If `pe_axis` is not None, shape is (X, Y, Z); otherwise (X, Y, Z, 3).
    """
    dwi = np.asarray(dwi)
    T1 = np.asarray(T1)

    if not HAVE_SYNB0:
        raise ImportError(
            "Synb0 model not available. Install PyTorch or TensorFlow to use "
            "Synb0-DISCO distortion correction."
        )

    if dwi.ndim not in (3, 4):
        raise ValueError(f"dwi must be a 3D or 4D array. Got shape {dwi.shape}.")
    if T1.ndim not in (3, 4):
        raise ValueError(f"T1 must be a 3D or 4D array. Got shape {T1.shape}.")

    if dwi.ndim == 4:
        _validate_b0_index(b0_index=b0_index, n_volumes=dwi.shape[-1], image_name="dwi")
    if T1.ndim == 4:
        _validate_b0_index(b0_index=b0_index, n_volumes=T1.shape[-1], image_name="T1")

    dwi_for_field = _select_3d_volume(dwi, image_name="dwi", b0_index=b0_index)
    T1_for_reg = _select_3d_volume(T1, image_name="T1", b0_index=b0_index)

    if dwi_mask is not None:
        dwi_mask = _select_3d_volume(dwi_mask, image_name="dwi_mask", b0_index=b0_index)
    if T1_mask is not None:
        T1_mask = _select_3d_volume(T1_mask, image_name="T1_mask", b0_index=b0_index)

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

    # ── Step 1: Affine T1 → MNI T1 (sparse sampling for speed) ──────────
    logger.info("Performing affine registration of T1 to MNI template...")
    T1_affine_to_template, t1_reg_affine = affine_registration(
        T1_for_reg,
        mni_t1,
        moving_affine=T1_affine,
        static_affine=mni_t1_affine,
        nbins=32,
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
    if T1_mask is not None:
        T1_mask_f = np.ascontiguousarray(np.asarray(T1_mask, dtype=np.float32))
        t1_affine_map = AffineMap(
            t1_reg_affine,
            domain_grid_shape=mni_t1.shape,
            domain_grid2world=mni_t1_affine,
            codomain_grid_shape=T1_for_reg.shape,
            codomain_grid2world=T1_affine,
        )
        T1_mask_in_template = (
            t1_affine_map.transform(T1_mask_f, interpolation="nearest") > 0.5
        )
        masked_T1_reg_to_template = T1_affine_to_template * T1_mask_in_template
    else:
        T1_mask_f = None
        masked_T1_reg_to_template = T1_affine_to_template

    # ── Step 2: Rigid b0 → T1, compose with T1 → MNI ────────────────────
    logger.info("Performing rigid registration of DWI b0 to T1...")
    dwi_p99 = np.percentile(dwi_for_field, 99)
    dwi_norm = normalize(dwi_for_field, min_v=0, max_v=dwi_p99, new_min=0, new_max=100)
    t1_p99 = np.percentile(T1_for_reg, 99)
    t1_norm = normalize(T1_for_reg, min_v=0, max_v=t1_p99, new_min=0, new_max=100)
    _, b0_to_t1_affine = affine_registration(
        dwi_norm,
        t1_norm,
        moving_affine=dwi_affine,
        static_affine=T1_affine,
        nbins=32,
        metric="MI",
        pipeline=["center_of_mass", "translation", "rigid", "affine"],
        level_iters=[100, 50, 25],
        sigmas=[4.0, 2.0, 1.0],
        factors=[8, 4, 2],
        moving_mask=dwi_mask.astype(bool) if dwi_mask is not None else None,
        static_mask=T1_mask.astype(bool) if T1_mask is not None else None,
    )
    dwi_reg_affine = t1_reg_affine @ b0_to_t1_affine

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
    mni_t1_zooms = tuple(np.sqrt(np.sum(mni_t1_affine[:3, :3] ** 2, axis=0)))

    masked_T1_reg_to_template = _normalize_t1_for_synb0(masked_T1_reg_to_template)

    masked_T1_reg_resized, resized_affine = reslice(
        masked_T1_reg_to_template,
        mni_t1_affine,
        mni_t1_zooms,
        synb0_zooms,
        new_shape=synb0_shape,
    )
    masked_dwi_reg_resized, _ = reslice(
        masked_dwi_reg_to_template,
        mni_t1_affine,
        mni_t1_zooms,
        synb0_zooms,
        new_shape=synb0_shape,
    )
    mni_mask_resized, _ = reslice(
        mni_mask.astype(np.float32),
        mni_t1_affine,
        mni_t1_zooms,
        synb0_zooms,
        new_shape=synb0_shape,
        order=0,
    )
    mni_mask_resized = (mni_mask_resized > 0.5).astype(np.int32)

    # ── Step 4: Synb0 synthesis ──────────────────────────────────────────
    synb0 = Synb0()
    logger.info("Synthesizing undistorted b0 with Synb0 model...")
    binf = synb0.predict(masked_dwi_reg_resized, masked_T1_reg_resized, **kwargs)
    binf = binf * mni_mask_resized
    _save_debug_volume(
        debug_dir=debug_dir, name="06_binf_template", data=binf, affine=resized_affine
    )

    # ── Step 5: Bring synthetic b0 back to native DWI space ──────────────
    binf_fullres, _ = reslice(
        binf,
        resized_affine,
        synb0_zooms,
        mni_t1_zooms,
        new_shape=mni_t1.shape,
    )
    dwi_inv_affine_map = AffineMap(
        dwi_reg_affine,
        domain_grid_shape=mni_t1.shape,
        domain_grid2world=mni_t1_affine,
        codomain_grid_shape=dwi_for_field.shape,
        codomain_grid2world=dwi_affine,
    )
    ori_binf = dwi_inv_affine_map.transform_inverse(binf_fullres)
    _save_debug_volume(
        debug_dir=debug_dir, name="06b_binf_in_dwi", data=ori_binf, affine=dwi_affine
    )
    ori_binf = np.clip(ori_binf, a_min=0, a_max=None)
    if dwi_mask_f is not None:
        ori_binf = ori_binf * dwi_mask_f
    _save_debug_volume(
        debug_dir=debug_dir, name="07_binf_in_dwi", data=ori_binf, affine=dwi_affine
    )

    # ── Step 6: SyN registration  (synthetic b0 ↔ distorted b0) ─────────
    logger.info("Performing nonlinear registration of synthesized b0 to DWI...")
    if dwi_mask_f is not None:
        masked_dwi_for_sdr = dwi_for_field * dwi_mask_f
    else:
        masked_dwi_for_sdr = dwi_for_field

    voxel_sizes = np.sqrt(np.sum(dwi_affine[:3, :3] ** 2, axis=0))

    binf_coverage = binary_dilation(ori_binf > 0, iterations=5).astype(np.float32)
    binf_coverage = gaussian_filter(binf_coverage, sigma=3.0)
    masked_dwi_for_sdr = masked_dwi_for_sdr * binf_coverage

    smooth_sigma_vox = 1.15 / voxel_sizes
    binf_smooth = gaussian_filter(ori_binf, sigma=smooth_sigma_vox)

    p99_s = (
        np.percentile(binf_smooth[binf_smooth > 0], 99)
        if np.any(binf_smooth > 0)
        else 1.0
    )
    p99_m = (
        np.percentile(masked_dwi_for_sdr[masked_dwi_for_sdr > 0], 99)
        if np.any(masked_dwi_for_sdr > 0)
        else 1.0
    )
    static_norm = binf_smooth / max(p99_s, 1e-8) * 100.0
    moving_norm = masked_dwi_for_sdr / max(p99_m, 1e-8) * 100.0

    sdr = SymmetricDiffeomorphicRegistration(
        metric=CCMetric(3, sigma_diff=3.5, radius=6),
        level_iters=[200, 100, 20],
        step_length=0.25,
    )
    mapping = sdr.optimize(
        static=static_norm,
        moving=moving_norm,
        static_grid2world=dwi_affine,
        moving_grid2world=dwi_affine,
        prealign=np.eye(4),
    )

    # ── Step 7: Extract PE-restricted field and post-smooth ──────────────
    field_smooth_mm = 10.0
    field_smooth_vox = field_smooth_mm / voxel_sizes

    if pe_axis is not None:
        field = mapping.get_forward_field()
        if isinstance(pe_axis, str):
            pe_axis = {"x": 0, "y": 1, "z": 2}.get(pe_axis.lower())
        if pe_axis in [0, 1, 2]:
            field[..., [i for i in range(3) if i != pe_axis]] = 0
            pe_disp = gaussian_filter(field[..., pe_axis], sigma=field_smooth_vox)
            field[..., pe_axis] = pe_disp
            mapping.forward = field
            field = pe_disp
        else:
            raise ValueError("pe_axis must be 'x', 'y', 'z' or 0, 1, 2.")
    elif return_field:
        field = mapping.get_forward_field()
        for c in range(3):
            field[..., c] = gaussian_filter(field[..., c], sigma=field_smooth_vox)
        mapping.forward = field

    dwi_to_binf = _warp_image(mapping=mapping, image=dwi)
    _save_debug_volume(
        debug_dir=debug_dir,
        name="08_dwi_corrected_b0",
        data=_select_3d_volume(
            dwi_to_binf, image_name="dwi_corrected", b0_index=b0_index
        ),
        affine=dwi_affine,
    )

    if return_field:
        _save_debug_volume(
            debug_dir=debug_dir, name="09_final_field", data=field, affine=dwi_affine
        )
        return dwi_to_binf, field
    else:
        return dwi_to_binf
