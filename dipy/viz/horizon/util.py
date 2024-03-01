import warnings
import numpy as np


def check_img_shapes(images):
    """Check if the images have same shapes. It also provides details about the
    volumes are same or not. If the shapes are not equal it will return False
    for both shape and volume.

    Parameters
    ----------
    images : list

    Returns
    -------
    tuple
        tuple[0] = True, if shapes are equal.
        tuple[1] = True, if volumes are equal.
    """

    if len(images) < 2:
        return (True, False)
    base_shape = images[0][0].shape[:3]
    volumed_data_shapes = []
    for img in images:
        data, _ = img
        if len(data.shape) == 4:
            volumed_data_shapes.append(data.shape[3])
        if base_shape != data.shape[:3]:
            return (False, False)

    return (True, len(set(volumed_data_shapes)) == 1)


def check_img_dtype(images):
    """Check supplied image dtype.

    If not supported numerical type, fallback to supported numerical types
    (either int32 or float 32). If non-numerical type, skip the data.

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


def is_binary_image(data, unique_points=100):
    """Check if an image is binary image.

    Parameters
    ----------
    data : ndarray
    unique_points : int, optional
        number of points to sample the check, by default 100

    Returns
    -------
    boolean
        Whether the image is binary or not
    """
    indices = []

    rng = np.random.default_rng()

    for dim in data.shape:
        indices.append(rng.integers(0, dim - 1, size=unique_points))

    return np.unique(np.take(data, indices)).shape[0] <= 2
