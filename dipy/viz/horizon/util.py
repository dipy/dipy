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
        data, _, _ = unpack_image(img)
        if base_shape != data.shape[:3]:
            return False
    return True


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
        data, affine, fname = unpack_image(img)
        if np.issubdtype(data.dtype, np.integer):
            if data.dtype != np.int32:
                msg = '{} is not supported, falling back to int32'
                warnings.warn(msg.format(data.dtype))
                img = (data.astype(np.int32), affine, fname)
            valid_images.append(img)
        elif np.issubdtype(data.dtype, np.floating):
            if data.dtype != np.float64 and data.dtype != np.float32:
                msg = '{} is not supported, falling back to float32'
                warnings.warn(msg.format(data.dtype))
                img = (data.astype(np.float32), affine, fname)
            valid_images.append(img)
        else:
            msg = 'skipping image {}, passed image is not in numerical format'
            warnings.warn(msg.format(idx + 1))

    return valid_images


def show_ellipsis(text, text_size, available_size):
    """Apply ellipsis to the text.

    Parameters
    ----------
    text : string
        Text required to be check for ellipsis.
    text_size : float
        Current size of the text in pixels.
    available_size : float
        Size available to fit the text. This will be used to truncate the text
        and show ellipsis.

    Returns
    -------
    string
        Text after processing for ellipsis.
    """
    if text_size > available_size:
        max_chars = int((available_size / text_size) * len(text))
        ellipsis_text = "..." + text[-(max_chars - 3):]
        return ellipsis_text
    return text


def unpack_image(img):
    """Unpack provided image data.

    Standard way to handle different value images.

    Parameters
    ----------
    img : tuple
        An image can contain either (data, affine) or (data, affine, fname).

    Returns
    -------
    tuple
        If img with (data, affine) it will convert to (data, affine, None).
        Otherwise it will be passed as it is.
    """

    if len(img) < 3:
        data, affine = img
        return data, affine, None

    data, affine, fname = img
    return data, affine, fname


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
