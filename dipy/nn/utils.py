import numpy as np

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
    return (image-norm_min)/(norm_max-norm_min)*(max_v-min_v) + min_v
