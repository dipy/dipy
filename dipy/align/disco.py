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

import numpy as np

from dipy.align import affine_registration
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
from dipy.data import get_fnames
from dipy.io.image import load_nifti
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


def synb0_syn(
    dwi,
    T1,
    dwi_affine,
    T1_affine,
    b0_index=0,
    dwi_mask=None,
    T1_mask=None,
    return_field=False,
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

    return_field : bool, optional
        Whether to return the estimated distortion field along with the
        corrected DWI data and synthesized undistorted b0. Default is False.

    kwargs :
        Additional keyword arguments to pass to synb0_predict and
        nonrigid registration functions.

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

    mni_t1_path, mni_t2_path, mni_mask_path = get_fnames("mni_resized_templates")
    mni_t1, mni_t1_affine = load_nifti(mni_t1_path)
    mni_t2, mni_t2_affine = load_nifti(mni_t2_path)
    mni_mask, mni_mask_affine = load_nifti(mni_mask_path)
    mni_mask = mni_mask.astype(np.int32)
    masked_mni_t1 = mni_t1 * mni_mask
    level_iters = [10000, 1000]
    sigmas = [3.0, 1.0]
    factors = [4, 2]
    pipeline = ["center_of_mass", "translation", "rigid", "affine"]
    xformed_data, reg_affine = affine_registration(
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

    metric = CCMetric(3)
    level_iters_syn = [200, 200, 100]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters_syn)
    mapping = sdr.optimize(
        masked_mni_t1, T1_for_reg, mni_t1_affine, T1_affine, reg_affine
    )
    T1_reg_to_template = mapping.transform(T1_for_reg)
    if T1_mask is not None:
        T1_mask = np.ascontiguousarray(np.asarray(T1_mask, dtype=np.float32))
        T1_mask_reg_to_template = (
            mapping.transform(T1_mask, interpolation="nearest") > 0.5
        )
        masked_T1_reg_to_template = T1_reg_to_template * T1_mask_reg_to_template
    else:
        masked_T1_reg_to_template = T1_reg_to_template

    xformed_data, reg_affine = affine_registration(
        dwi_for_field,
        mni_t2,
        moving_affine=dwi_affine,
        static_affine=mni_t1_affine,
        nbins=32,
        metric="MI",
        pipeline=pipeline,
        level_iters=level_iters,
        sigmas=sigmas,
        factors=factors,
        moving_mask=dwi_mask,
        static_mask=mni_mask,
    )

    if dwi_mask is not None:
        masked_dwi = dwi_for_field * dwi_mask
    else:
        masked_dwi = dwi_for_field
    masked_mni_t2 = mni_t2 * mni_mask

    metric = CCMetric(3)
    level_iters_syn = [100, 100, 50]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters_syn)
    mapping = sdr.optimize(
        masked_mni_t2, masked_dwi, mni_t1_affine, dwi_affine, reg_affine
    )
    dwi_reg_to_template = mapping.transform(dwi_for_field)
    if dwi_mask is not None:
        dwi_mask_f = np.ascontiguousarray(np.asarray(dwi_mask, dtype=np.float32))
        dwi_mask_reg_to_template = (
            mapping.transform(dwi_mask_f, interpolation="nearest") > 0.5
        )
        masked_dwi_reg_to_template = dwi_reg_to_template * dwi_mask_reg_to_template
    else:
        masked_dwi_reg_to_template = dwi_reg_to_template
    synb0 = Synb0()
    binf = synb0.predict(masked_dwi_reg_to_template, masked_T1_reg_to_template)
    ori_binf = mapping.transform_inverse(binf)

    sdr = SymmetricDiffeomorphicRegistration(
        metric=CCMetric(3), level_iters=[100, 100, 50]
    )
    pre_align = np.eye(4)
    mapping = sdr.optimize(
        static=ori_binf,
        moving=dwi_for_field,
        static_grid2world=dwi_affine,
        moving_grid2world=dwi_affine,
        prealign=pre_align,
    )
    dwi_to_binf = _warp_image(mapping, dwi)

    if return_field:
        field = mapping.get_forward_field()
        return dwi_to_binf, field
    else:
        return dwi_to_binf
