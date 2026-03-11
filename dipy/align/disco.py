"""
Distortion Correction Module

This module provides tools for correcting various types of distortions
in diffusion MRI data, including susceptibility-induced distortions
using the Synb0-SyN method.

References
----------
.. [1] Chigurupati, S., et al. (2024). "Fast susceptibility distortion correction
       for diffusion MRI using style transfer and nonrigid registration."
       Proceedings of ISMRM.
.. [2] Schilling, K. G., et al. (2019). "Synthesized b0 for diffusion
       distortion correction (Synb0-DisCo)." Magnetic Resonance Imaging,
       64, 62-70.
.. [3] Schilling, K. G., et al. (2020). "Distortion correction of
       diffusion weighted MRI without reverse phase-encoding scans or
       field-maps." PLOS ONE, 15(7), e0236659.
"""

from pathlib import Path

import numpy as np

from dipy.align import affine_registration
from dipy.align.imaffine import AffineMap
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
from dipy.align.reslice import reslice
from dipy.data import get_fnames
from dipy.io.image import load_nifti, save_nifti
from dipy.nn.utils import normalize
from dipy.utils.logging import logger

# Try to import Synb0 - the backend (torch/tf) is selected automatically
# based on DIPY_NN_BACKEND environment variable
try:
    from dipy.nn.synb0 import Synb0

    HAVE_SYNB0 = True
except ImportError:
    HAVE_SYNB0 = False
    logger.warning(
        "Synb0 model not available. Install PyTorch or TensorFlow to use."
        "(pip install dipy[ml])"
        "Synb0-DISCO distortion correction."
    )


def _validate_b0_index(b0_index, n_volumes, image_name):
    if not isinstance(b0_index, (int, np.integer)):
        raise TypeError("b0_index must be an integer.")

    if b0_index < 0 or b0_index >= n_volumes:
        raise ValueError(
            f"b0_index ({b0_index}) must be in [0, {n_volumes - 1}] "
            f"for {image_name} with {n_volumes} volumes."
        )

    return int(b0_index)


def _select_3d_volume(image, *, image_name, b0_index):
    image = np.asarray(image)

    if image.ndim == 3:
        return image

    if image.ndim != 4:
        raise ValueError(
            f"{image_name} must be a 3D or 4D array. Got shape {image.shape}."
        )

    idx = _validate_b0_index(b0_index, image.shape[-1], image_name)
    return image[..., idx]


def _warp_image(mapping, image):
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


def _save_debug_volume(debug_dir, name, data, affine):
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

    b0_index : int, optional
        The index of the volume used for field estimation/registration when
        `dwi` and/or `T1` are 4D. Must be less than the number of volumes in
        each 4D input. Default is 0.

    dwi_mask : ndarray (X, Y, Z), optional
        A binary mask for the DWI data. If None, no masking is applied.
        Default is None.

    T1_mask : ndarray (X, Y, Z), optional
        A binary mask for the T1 image. If None, no masking is applied.
        Default is None.

    pe_axis : int or str, optional
        The phase-encoding axis. Can be specified as an integer (0 for 'x',
        1 for 'y', 2 for 'z') or a string ('x', 'y', 'z'). Default is 1 ('y').
        If provided, the returned field will be restricted to this direction.

    return_field : bool, optional
        Whether to return the estimated distortion field along with the
        corrected DWI data and synthesized undistorted b0. Default is False.

    debug_dir : str or Path, optional
        Directory to save intermediate debug NIfTI volumes.

    kwargs :
        Additional keyword arguments to pass to Synb0.predict.

    Returns
    -------
    corrected_dwi : ndarray
        Distortion-corrected DWI image. Same dimensionality as `dwi` (3D or
        4D). If `dwi` is 4D, the field estimated from `dwi[..., b0_index]` is
        applied to all volumes.
    field : ndarray, optional
        Estimated deformation field, returned only when `return_field=True`.
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
        _validate_b0_index(b0_index, dwi.shape[-1], "dwi")
    if T1.ndim == 4:
        _validate_b0_index(b0_index, T1.shape[-1], "T1")

    dwi_for_field = _select_3d_volume(dwi, image_name="dwi", b0_index=b0_index)
    T1_for_reg = _select_3d_volume(T1, image_name="T1", b0_index=b0_index)

    if dwi_mask is not None:
        dwi_mask = _select_3d_volume(dwi_mask, image_name="dwi_mask", b0_index=b0_index)
    if T1_mask is not None:
        T1_mask = _select_3d_volume(T1_mask, image_name="T1_mask", b0_index=b0_index)

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    _save_debug_volume(debug_dir, "00_dwi_b0_input", dwi_for_field, dwi_affine)
    _save_debug_volume(debug_dir, "01_t1_input", T1_for_reg, T1_affine)
    if dwi_mask is not None:
        _save_debug_volume(debug_dir, "01b_dwi_mask_input", dwi_mask, dwi_affine)
    if T1_mask is not None:
        _save_debug_volume(debug_dir, "01c_t1_mask_input", T1_mask, T1_affine)

    _, mni_t2_path, mni_mask_path, mni_t1_path = get_fnames("mni_templates")
    mni_t1, mni_t1_affine = load_nifti(mni_t1_path)  # 09c T1
    mni_t2, mni_t2_affine = load_nifti(mni_t2_path)  # 09a T2
    mni_mask, _ = load_nifti(mni_mask_path)  # 09c mask
    mni_mask = mni_mask.astype(np.int32)

    # Resample mask to T2 grid if shapes differ (09c mask vs 09a T2)
    if mni_mask.shape != mni_t2.shape:
        mask_to_t2 = AffineMap(
            np.eye(4),
            domain_grid_shape=mni_t2.shape,
            domain_grid2world=mni_t2_affine,
            codomain_grid_shape=mni_mask.shape,
            codomain_grid2world=mni_t1_affine,
        )
        mni_mask_t2 = (
            mask_to_t2.transform(mni_mask.astype(np.float32), interpolation="nearest")
            > 0.5
        ).astype(np.int32)
    else:
        mni_mask_t2 = mni_mask

    if not np.allclose(mni_t1_affine, mni_t2_affine):
        logger.warning(
            "MNI T1 and T2 template affines differ. "
            "Using MNI T1 affine for DWI->template registration to match "
            "the Synb0-DISCO pipeline."
        )

    level_iters = [10000, 1000]
    sigmas = [3.0, 1.0]
    factors = [4, 2]
    pipeline = ["center_of_mass", "translation", "rigid", "affine"]
    T1_affine_to_template, t1_reg_affine = affine_registration(
        T1_for_reg,
        mni_t1,
        moving_affine=T1_affine,
        static_affine=mni_t1_affine,
        nbins=32,
        metric="MI",
        pipeline=pipeline,
        level_iters=level_iters,
        sigmas=sigmas,
        factors=factors,
        moving_mask=T1_mask,
        static_mask=mni_mask,
    )
    _save_debug_volume(
        debug_dir,
        "01d_t1_affine_to_template",
        T1_affine_to_template,
        mni_t1_affine,
    )
    masked_mni_t1 = mni_t1 * mni_mask
    if T1_mask is not None:
        T1_mask_f = np.ascontiguousarray(np.asarray(T1_mask, dtype=np.float32))
        masked_t1 = T1_for_reg * T1_mask_f
    else:
        T1_mask_f = None
        masked_t1 = T1_for_reg

    t1_template_sdr = SymmetricDiffeomorphicRegistration(
        metric=CCMetric(3), level_iters=[200, 200, 100]
    )
    t1_template_mapping = t1_template_sdr.optimize(
        static=masked_mni_t1,
        moving=masked_t1,
        static_grid2world=mni_t1_affine,
        moving_grid2world=T1_affine,
        prealign=t1_reg_affine,
    )
    T1_reg_to_template = t1_template_mapping.transform(T1_for_reg)
    _save_debug_volume(
        debug_dir, "02_t1_reg_to_template", T1_reg_to_template, mni_t1_affine
    )

    if T1_mask_f is not None:
        T1_mask_reg_to_template = (
            t1_template_mapping.transform(T1_mask_f, interpolation="nearest") > 0.5
        )
        masked_T1_reg_to_template = T1_reg_to_template * T1_mask_reg_to_template
        _save_debug_volume(
            debug_dir,
            "02b_t1_mask_reg_to_template",
            T1_mask_reg_to_template,
            mni_t1_affine,
        )
    else:
        masked_T1_reg_to_template = T1_reg_to_template
    _save_debug_volume(
        debug_dir,
        "03_t1_reg_to_template_masked",
        masked_T1_reg_to_template,
        mni_t1_affine,
    )

    dwi_p99 = np.percentile(dwi_for_field, 99)
    dwi_norm = normalize(dwi_for_field, min_v=0, max_v=dwi_p99, new_min=0, new_max=100)
    mni_t2_p99 = np.percentile(mni_t2, 99)
    mni_t2_norm = normalize(mni_t2, min_v=0, max_v=mni_t2_p99, new_min=0, new_max=100)

    dwi_affine_to_template, dwi_reg_affine = affine_registration(
        dwi_norm,
        mni_t2_norm,
        moving_affine=dwi_affine,
        static_affine=mni_t2_affine,
        nbins=32,
        metric="MI",
        pipeline=pipeline,
        level_iters=level_iters,
        sigmas=sigmas,
        factors=factors,
        moving_mask=dwi_mask,
        static_mask=mni_mask_t2,
    )
    _save_debug_volume(
        debug_dir,
        "04_b0_affine_to_template",
        dwi_affine_to_template,
        mni_t2_affine,
    )

    if dwi_mask is not None:
        dwi_mask_f = np.ascontiguousarray(np.asarray(dwi_mask, dtype=np.float32))
    else:
        dwi_mask_f = None
    masked_mni_t2 = mni_t2_norm * mni_mask_t2

    dwi_template_sdr = SymmetricDiffeomorphicRegistration(
        metric=CCMetric(3), level_iters=[200, 200, 100]
    )
    dwi_template_mapping = dwi_template_sdr.optimize(
        static=masked_mni_t2,
        moving=dwi_norm,
        static_grid2world=mni_t2_affine,
        moving_grid2world=dwi_affine,
        prealign=dwi_reg_affine,
    )
    dwi_reg_to_template = dwi_template_mapping.transform(dwi_for_field)
    _save_debug_volume(
        debug_dir, "04_b0_reg_to_template", dwi_reg_to_template, mni_t2_affine
    )

    if dwi_mask_f is not None:
        dwi_mask_reg_to_template = (
            dwi_template_mapping.transform(dwi_mask_f, interpolation="nearest") > 0.5
        )
        masked_dwi_reg_to_template = dwi_reg_to_template * dwi_mask_reg_to_template
        _save_debug_volume(
            debug_dir,
            "04b_dwi_mask_reg_to_template",
            dwi_mask_reg_to_template,
            mni_t1_affine,
        )
    else:
        masked_dwi_reg_to_template = dwi_reg_to_template
    _save_debug_volume(
        debug_dir,
        "05_b0_reg_to_template_masked",
        masked_dwi_reg_to_template,
        mni_t1_affine,
    )
    # Reslice registered images from full-res to 2.5mm for Synb0 input
    synb0_shape = (77, 91, 77)
    synb0_zooms = (2.5, 2.5, 2.5)
    mni_t1_zooms = tuple(np.sqrt(np.sum(mni_t1_affine[:3, :3] ** 2, axis=0)))
    mni_t2_zooms = tuple(np.sqrt(np.sum(mni_t2_affine[:3, :3] ** 2, axis=0)))

    masked_T1_reg_resized, resized_affine = reslice(
        masked_T1_reg_to_template,
        mni_t1_affine,
        mni_t1_zooms,
        synb0_zooms,
        new_shape=synb0_shape,
    )
    masked_dwi_reg_resized, _ = reslice(
        masked_dwi_reg_to_template,
        mni_t2_affine,
        mni_t2_zooms,
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

    synb0 = Synb0()
    binf = synb0.predict(masked_dwi_reg_resized, masked_T1_reg_resized, **kwargs)
    binf = binf * mni_mask_resized
    _save_debug_volume(debug_dir, "06_binf_template", binf, resized_affine)
    binf_fullres, _ = reslice(
        binf,
        resized_affine,
        synb0_zooms,
        mni_t1_zooms,
        new_shape=mni_t1.shape,
    )
    binf_in_t1 = t1_template_mapping.transform_inverse(binf_fullres)
    _save_debug_volume(debug_dir, "06b_binf_in_t1", binf_in_t1, T1_affine)
    ori_binf, _ = affine_registration(
        binf_in_t1,
        dwi_for_field,
        moving_affine=T1_affine,
        static_affine=dwi_affine,
        nbins=32,
        metric="MI",
        pipeline=["center_of_mass", "translation", "rigid", "affine"],
        level_iters=level_iters,
        sigmas=sigmas,
        factors=factors,
    )
    ori_binf = np.clip(ori_binf, a_min=0, a_max=None)
    if dwi_mask_f is not None:
        ori_binf = ori_binf * dwi_mask_f
    _save_debug_volume(debug_dir, "07_binf_in_dwi", ori_binf, dwi_affine)

    sdr = SymmetricDiffeomorphicRegistration(
        metric=CCMetric(3), level_iters=[200, 200, 100]
    )
    pre_align = np.eye(4)
    if dwi_mask_f is not None:
        masked_dwi_for_sdr = dwi_for_field * dwi_mask_f
    else:
        masked_dwi_for_sdr = dwi_for_field
    mapping = sdr.optimize(
        static=ori_binf,
        moving=masked_dwi_for_sdr,
        static_grid2world=dwi_affine,
        moving_grid2world=dwi_affine,
        prealign=pre_align,
    )
    if pe_axis is not None:
        # Make other axes zero in the forward field
        #  to restrict correction to the PE direction
        field = mapping.get_forward_field()
        if isinstance(pe_axis, str):
            pe_axis = {"x": 0, "y": 1, "z": 2}.get(pe_axis.lower())
        if pe_axis in [0, 1, 2]:
            field[..., [i for i in range(3) if i != pe_axis]] = 0
            mapping.forward = field
            field = field[..., pe_axis]
        else:
            raise ValueError("pe_axis must be 'x', 'y', 'z' or 0, 1, 2.")
    dwi_to_binf = _warp_image(mapping, dwi)
    _save_debug_volume(
        debug_dir,
        "08_dwi_corrected_b0",
        _select_3d_volume(dwi_to_binf, image_name="dwi_corrected", b0_index=b0_index),
        dwi_affine,
    )

    if return_field:
        _save_debug_volume(debug_dir, "09_final_field", field, dwi_affine)

        return dwi_to_binf, field
    else:
        return dwi_to_binf
