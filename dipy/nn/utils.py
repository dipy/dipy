import numpy as np
from scipy.ndimage import affine_transform
from dipy.align.reslice import reslice

def normalize(image, min_v=None, max_v=None, new_min=-1, new_max=1):
    r"""
    normalization function

    Parameters
    ----------
    image : np.ndarray
    min_v : int or float (optional)
        minimum value range for normalization
        intensities below min_v will be clipped
        if None it is set to min value of image
        Default : None
    max_v : int or float (optional)
        maximum value range for normalization
        intensities above max_v will be clipped
        if None it is set to max value of image
        Default : None
    new_min : int or float (optional)
        new minimum value after normalization
        Default : 0
    new_max : int or float (optional)
        new maximum value after normalization
        Default : 1

    Returns
    -------
    np.ndarray
        Normalized image from range new_min to new_max
    """
    if min_v is None:
        min_v = np.min(image)
    if max_v is None:
        max_v = np.max(image)
    return np.interp(image, (min_v, max_v), (new_min, new_max))

def unnormalize(image, norm_min, norm_max, min_v, max_v):
    r"""
    unnormalization function

    Parameters
    ----------
    image : np.ndarray
    norm_min : int or float
        minimum value of normalized image
    norm_max : int or float
        maximum value of normalized image
    min_v : int or float
        minimum value of unnormalized image
    max_v : int or float
        maximum value of unnormalized image

    Returns
    -------
    np.ndarray
        unnormalized image from range min_v to max_v
    """
    return (image - norm_min) / (norm_max-norm_min) * \
           (max_v - min_v) + min_v


def set_logger_level(log_level, logger):
    """ Change the logger to one of the following:
    DEBUG, INFO, WARNING, CRITICAL, ERROR

    Parameters
    ----------
    log_level : str
        Log level for the logger
    """
    logger.setLevel(level=log_level)

def transform_img(image, affine, init_shape=(256, 256, 256), scale=2):
    r"""
    Function to reshape image as an input to the model

    Parameters
    ----------
    image : np.ndarray
        Image to transform to voxelspace
    affine : np.ndarray
        Affine matrix provided by the file
    init_shape : tuple
        Initial shape to transform the image to
        Default is (256, 256, 256)
    scale : float
        How much we want to scale the image
        Default is 2

    Returns
    -------
    transformed_img : np.ndarray
    """
    affine2 = affine.copy()
    affine2[:3, 3] += np.array([init_shape[0]//2,
                                init_shape[1]//2,
                                init_shape[2]//2])
    inv_affine = np.linalg.inv(affine2)
    transformed_img = affine_transform(image, inv_affine, output_shape=init_shape)
    transformed_img, _ = reslice(transformed_img, np.eye(4), (1, 1, 1),
                                 (scale, scale, scale))
    return transformed_img, affine2
    
def recover_img(image, affine, ori_shape, scale=2):
    r"""
    Function to recover image back to its original shape

    Parameters
    ----------
    image : np.ndarray
        Image to recover
    affine : np.ndarray
        Affine matrix provided from transform_img
    ori_shape : tuple
        Original shape of image
    scale : float
        Scale that was used in transform_img
        Default is 2

    Returns
    -------
    recovered_img : np.ndarray
    """
    new_image, _ = reslice(image, np.eye(4), (scale, scale, scale), (1, 1, 1))
    recovered_img = affine_transform(new_image, affine, output_shape=ori_shape)
    return recovered_img