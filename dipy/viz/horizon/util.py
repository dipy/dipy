import warnings
import numpy as np


def check_img_shapes(images):
    """Check if the images have same shapes.

    Parameters
    ----------
    images : list

    Returns
    -------
    boolean
        True, if shapes are equal.
    """

    if len(images) < 2:
        return True
    base_shape = images[0][0].shape[:3]
    for img in images:
        data, _ = img
        if base_shape != data.shape[:3]:
            return False
    return True


def check_img_dtype(images):
    """Check for supported dtype. If not supported numerical type, fallback to
    supported numerical types (either int32 or float 32). If non-numerical
    type, skip the data.

    Parameters
    ----------
    images : list
        Each image is tuple of (data, affine).

    Returns
    -------
    list
        Valid images from the provided images.
    """

    valid_images = []

    for idx, img in enumerate(images):
        data, affine = img
        if np.issubdtype(data.dtype, np.integer):
            if data.dtype != np.int32:
                msg = '{} is not supported, falling back to int32'
                warnings.warn(msg.format(data.dtype))
                img = (data.astype(np.int32), affine)
            valid_images.append(img)
        elif np.issubdtype(data.dtype, np.floating):
            if data.dtype != np.float64 and data.dtype != np.float32:
                msg = '{} is not supported, falling back to float32'
                warnings.warn(msg.format(data.dtype))
                img = (data.astype(np.float32), affine)
            valid_images.append(img)
        else:
            msg = 'skipping image {}, passed image is not in numerical format'
            warnings.warn(msg.format(idx + 1))

    return valid_images
