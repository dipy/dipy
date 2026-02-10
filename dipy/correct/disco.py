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
    dwi : ndarray (X, Y, Z, N) / (X, Y, X)
        The input diffusion MRI data with shape (X, Y, Z) where N is
        the number of volumes.

    T1 : ndarray (X, Y, Z)
        The T1-weighted structural image that should be in the same space
        as the desired undistorted b0.

    b0_index : int, optional
        The index of the b0 volume in the DWI data. Default is 0.

    dwi_mask : ndarray (X, Y, Z), optional
        A binary mask for the DWI data. If None, no masking is applied.
        Default is None.

    T1_mask : ndarray (X, Y, Z), optional
        A binary mask for the T1 image. If None, no masking is applied.
        Default is None.

    return_field : bool, optional
        Whether to return the estimated distortion field along with the
        corrected DWI data. Default is False.

    kwargs :
        Additional keyword arguments to pass to synb0_predict and
        nonrigid registration functions.
    """
    if dwi.ndim == 4:
        dwi = dwi[..., b0_index]
    mni_t1_path, mni_t2_path, mni_mask_path = get_fnames("mni_resized_templates")
    mni_t1, mni_t1_affine = load_nifti(mni_t1_path)
    mni_t2, mni_t2_affine = load_nifti(mni_t2_path)
    mni_mask, mni_mask_affine = load_nifti(mni_mask_path)
    mni_mask = mni_mask.astype(np.int32)
    print(
        "Shapes of T1, DWI, MNI T1, and MNI mask:",
        T1.shape,
        dwi.shape,
        mni_t1.shape,
        mni_mask.shape,
    )
    masked_mni_t1 = mni_t1 * mni_mask
    print("Shapes of T1 and MNI T1:", T1.shape, mni_t1.shape)
    level_iters = [10000, 1000]
    sigmas = [3.0, 1.0]
    factors = [4, 2]
    pipeline = ["center_of_mass", "translation", "rigid", "affine"]
    xformed_data, reg_affine = affine_registration(
        T1,
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
    mapping = sdr.optimize(masked_mni_t1, T1, mni_t1_affine, T1_affine, reg_affine)
    T1_reg_to_template = mapping.transform(T1)
    if T1_mask is not None:
        T1_mask = np.ascontiguousarray(np.asarray(T1_mask, dtype=np.float32))
        T1_mask_reg_to_template = (
            mapping.transform(T1_mask, interpolation="nearest") > 0.5
        )
        masked_T1_reg_to_template = T1_reg_to_template * T1_mask_reg_to_template
    else:
        masked_T1_reg_to_template = T1_reg_to_template

    xformed_data, reg_affine = affine_registration(
        dwi,
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
        masked_dwi = dwi * dwi_mask
    else:
        masked_dwi = dwi
    masked_mni_t2 = mni_t2 * mni_mask

    metric = CCMetric(3)
    level_iters_syn = [200, 200, 100]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters_syn)
    mapping = sdr.optimize(
        masked_mni_t2, masked_dwi, mni_t1_affine, dwi_affine, reg_affine
    )
    dwi_reg_to_template = mapping.transform(dwi)
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
        metric=CCMetric(3), level_iters=[200, 200, 100]
    )
    pre_align = np.eye(4)
    mapping = sdr.optimize(
        static=ori_binf,
        moving=dwi,
        static_grid2world=dwi_affine,
        moving_grid2world=dwi_affine,
        prealign=pre_align,
    )
    dwi_to_binf = dwi.copy()
    dwi_to_binf = mapping.transform(dwi)

    if return_field:
        field = mapping.get_forward_field()
        return dwi_to_binf, field
    else:
        return dwi_to_binf
