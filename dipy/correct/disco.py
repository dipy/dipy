"""
Distortion Correction Module

This module provides tools for correcting various types of distortions
in diffusion MRI data, including susceptibility-induced distortions
using the Synb0-DISCO method.

References
----------
.. [1] Schilling, K. G., et al. (2019). "Synthesized b0 for diffusion
       distortion correction (Synb0-DisCo)." Magnetic Resonance Imaging,
       64, 62-70.
.. [2] Schilling, K. G., et al. (2020). "Distortion correction of
       diffusion weighted MRI without reverse phase-encoding scans or
       field-maps." PLOS ONE, 15(7), e0236659.
"""

import numpy as np

from dipy.testing.decorators import warning_for_keywords
from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package

# Try to import Synb0 - the backend (torch/tf) is selected automatically
# based on DIPY_NN_BACKEND environment variable
try:
    from dipy.nn.synb0 import Synb0

    HAVE_SYNB0 = True
except ImportError:
    HAVE_SYNB0 = False
    logger.warning(
        "Synb0 model not available. Install PyTorch or TensorFlow to use "
        "Synb0-DISCO distortion correction."
    )


def dummy_distortion_correction(data, affine=None, b0_threshold=50):
    """
    Dummy function for distortion correction.

    This is a placeholder function that will be replaced with actual
    distortion correction algorithms. Currently, it just returns the
    input data unchanged.

    Parameters
    ----------
    data : ndarray
        The input diffusion MRI data to be corrected.
        Shape should be (X, Y, Z, N) where N is the number of volumes.
    affine : ndarray, optional
        The 4x4 affine transformation matrix. If None, an identity
        matrix is used.
    b0_threshold : float, optional
        The threshold below which a b-value is considered as b0.
        Default is 50.

    Returns
    -------
    corrected_data : ndarray
        The corrected data (currently just a copy of input data).
    corrected_affine : ndarray
        The corrected affine transformation matrix.

    Notes
    -----
    This is a dummy implementation. Future versions will include:
    - Susceptibility-induced distortion correction
    - Eddy current correction
    - Motion correction integration

    Examples
    --------
    >>> import numpy as np
    >>> from dipy.correct.disco import dummy_distortion_correction
    >>> data = np.random.rand(10, 10, 10, 32)
    >>> corrected_data, corrected_affine = dummy_distortion_correction(data)
    >>> corrected_data.shape
    (10, 10, 10, 32)
    """
    # Create default affine if not provided
    if affine is None:
        affine = np.eye(4)

    # For now, just return a copy of the data and affine
    # This is where actual distortion correction would happen
    corrected_data = np.copy(data)
    corrected_affine = np.copy(affine)

    # Placeholder for future implementation
    # TODO: Implement actual distortion correction algorithm
    # - Estimate distortion field
    # - Apply correction
    # - Update affine transformation if needed

    return corrected_data, corrected_affine


def estimate_distortion_field(b0_image, phase_encoding_direction='y'):
    """
    Dummy function to estimate the distortion field.

    This is a placeholder for distortion field estimation.

    Parameters
    ----------
    b0_image : ndarray
        The b0 (non-diffusion weighted) image.
    phase_encoding_direction : str, optional
        The phase encoding direction ('x', 'y', or 'z').
        Default is 'y'.

    Returns
    -------
    distortion_field : ndarray
        The estimated distortion field (currently zeros).

    Notes
    -----
    This is a dummy implementation. Future versions will include
    actual field estimation algorithms.
    """
    # Return a zero field for now
    distortion_field = np.zeros_like(b0_image)

    return distortion_field


@warning_for_keywords()
def synb0_predict(b0, T1, *, batch_size=None, average=True, verbose=False):
    """
    Synthesize an undistorted b0 image using Synb0-DISCO.

    This function uses the Synb0 deep learning model to synthesize an
    undistorted b0 image from a distorted b0 and a T1-weighted image.
    The backend (PyTorch or TensorFlow) is selected based on the
    DIPY_NN_BACKEND environment variable.

    Parameters
    ----------
    b0 : ndarray (batch, 77, 91, 77) or (77, 91, 77)
        Distorted b0 (non-diffusion weighted) image.
        For a single image, input should be a 3D array. If multiple images,
        there should also be a batch dimension.

    T1 : ndarray (batch, 77, 91, 77) or (77, 91, 77)
        T1-weighted structural image that should be in the same space
        as the desired undistorted b0.
        For a single image, input should be a 3D array. If multiple images,
        there should also be a batch dimension.

    batch_size : int, optional
        Number of images per prediction pass. Only available if data
        is provided with a batch dimension.
        Consider lowering it if you get an out of memory error.
        Increase it if you want it to be faster and have a lot of data.
        If None, batch_size will be set to 1.
        Default is None.

    average : bool, optional
        Whether to average the prediction of 5 different models as in
        the original Synb0-Disco pipeline. If False, uses only the loaded
        weights for prediction.
        Default is True.

    verbose : bool, optional
        Whether to show information about the processing.
        Default is False.

    Returns
    -------
    pred_output : ndarray (...) or (batch, ...)
        Synthesized undistorted b0 image(s) with the same shape as input.

    Raises
    ------
    ImportError
        If neither PyTorch nor TensorFlow is available.
    ValueError
        If input shapes are incorrect.

    Notes
    -----
    The input images should be pre-processed and registered to MNI space
    (77, 91, 77) with the standard Synb0-DISCO preprocessing pipeline.
    This function performs only the neural network inference part.

    The backend selection priority is:
    1. DIPY_NN_BACKEND environment variable ('torch' or 'tf')
    2. PyTorch if available and DIPY_NN_BACKEND not set
    3. TensorFlow if PyTorch not available

    Examples
    --------
    >>> import numpy as np
    >>> from dipy.correct.disco import synb0_predict
    >>> # Create dummy data (in practice, use real pre-processed images)
    >>> b0 = np.random.rand(77, 91, 77) * 1000
    >>> T1 = np.random.rand(77, 91, 77) * 150
    >>> # Predict undistorted b0 (requires trained weights)
    >>> # b0_corrected = synb0_predict(b0, T1, average=False)

    References
    ----------
    .. [1] Schilling, K. G., et al. (2019). "Synthesized b0 for diffusion
           distortion correction (Synb0-DisCo)." Magnetic Resonance
           Imaging, 64, 62-70.
    """
    if not HAVE_SYNB0:
        raise ImportError(
            "Synb0 model not available. Install PyTorch or TensorFlow:\n"
            "  pip install torch  # for PyTorch backend\n"
            "  pip install tensorflow  # for TensorFlow backend\n"
            "Set DIPY_NN_BACKEND='torch' or 'tf' to choose backend."
        )

    # Create model and run prediction
    model = Synb0(verbose=verbose)

    prediction = model.predict(b0, T1, batch_size=batch_size, average=average)

    return prediction


@warning_for_keywords()
def synb0_distortion_correction(
    b0_image,
    T1_image,
    *,
    average=True,
    verbose=False,
):
    """
    Perform susceptibility distortion correction using Synb0-DISCO.

    This function synthesizes an undistorted b0 image using the Synb0
    deep learning model. It provides a simplified interface to the
    synb0_predict function.

    Parameters
    ----------
    b0_image : ndarray (77, 91, 77)
        A distorted b0 (non-diffusion weighted) image in MNI space.

    T1_image : ndarray (77, 91, 77)
        A T1-weighted structural image registered to MNI space.

    average : bool, optional
        Whether to average predictions from 5 different models as in
        the original Synb0-DISCO pipeline.
        Default is True.

    verbose : bool, optional
        Whether to show processing information.
        Default is False.

    Returns
    -------
    b0_undistorted : ndarray (77, 91, 77)
        The synthesized undistorted b0 image.

    Raises
    ------
    ImportError
        If neither PyTorch nor TensorFlow is available.
    ValueError
        If input shapes are incorrect.

    Notes
    -----
    This is a simplified interface. The full Synb0-DISCO pipeline involves:

    1. Skull stripping and normalization of both b0 and T1
    2. Registration to MNI152 template space (77, 91, 77)
    3. Neural network inference (this function)
    4. Transformation back to native space
    5. Optional TOPUP integration for final correction

    This function performs only step 3. For the complete pipeline,
    additional preprocessing and postprocessing steps are required.

    Examples
    --------
    >>> import numpy as np
    >>> from dipy.correct.disco import synb0_distortion_correction
    >>> # Assuming preprocessed images in MNI space
    >>> b0_mni = np.random.rand(77, 91, 77) * 1000
    >>> T1_mni = np.random.rand(77, 91, 77) * 150
    >>> # Get undistorted b0 (requires trained weights)
    >>> # b0_undistorted = synb0_distortion_correction(b0_mni, T1_mni,
    >>> #                                               average=False)

    References
    ----------
    .. [1] Schilling, K. G., et al. (2019). "Synthesized b0 for diffusion
           distortion correction (Synb0-DisCo)." Magnetic Resonance
           Imaging, 64, 62-70.
    .. [2] Schilling, K. G., et al. (2020). "Distortion correction of
           diffusion weighted MRI without reverse phase-encoding scans or
           field-maps." PLOS ONE, 15(7), e0236659.
    """
    # Input validation
    if b0_image.shape != (77, 91, 77):
        raise ValueError(
            f"b0_image must have shape (77, 91, 77) (MNI space), "
            f"got {b0_image.shape}. Please preprocess and register "
            f"your data to MNI space first."
        )

    if T1_image.shape != (77, 91, 77):
        raise ValueError(
            f"T1_image must have shape (77, 91, 77) (MNI space), "
            f"got {T1_image.shape}. Please preprocess and register "
            f"your data to MNI space first."
        )

    # Synthesize undistorted b0
    b0_undistorted = synb0_predict(
        b0_image, T1_image, average=average, verbose=verbose
    )

    return b0_undistorted
