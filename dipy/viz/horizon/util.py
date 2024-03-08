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
        data, _, _ = unpack_image(img)
        data_shape = data.shape[:3]
        if data.ndim == 4:
            volumed_data_shapes.append(data.shape[3])
        if base_shape != data_shape:
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
    return _unpack_data(img)


def unpack_surface(surface):
    """Unpack surface data.

    Parameters
    ----------
    surface : tuple
        It either contains (vertices, faces) or (vertices, faces, fname).

    Returns
    -------
    tuple
        If surface with (vertices, faces) it will convert to (vertices, faces,
        None). Otherwise it will be passed as it is.
    """
    data = _unpack_data(surface)

    if data[0].shape[-1] != 3:
        raise ValueError('Vertices do not have correct shape:' +
                         f' {data[0].shape}')
    if data[1].shape[-1] != 3:
        raise ValueError('Faces do not have correct shape:' +
                         f' {data[1].shape}')
    return data


def _unpack_data(data, return_size=3):
    result = [*data]
    if len(data) < return_size:
        result += (return_size - len(data)) * [None]

    return result


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


def check_peak_size(pams, ref_img_shape=None, sync_imgs=False):
    """Check shape of peaks.

    Parameters
    ----------
    pams : PeaksAndMetrics
    ref_img_shape : tuple, optional
        3D shape of the image, by default None.
    sync_imgs : bool, optional
        True if the images are synchronized, by default False.

    Returns
    -------
    bool
        If the peaks are aligned with images and other peaks.
    """
    base_shape = pams[0].peak_dirs.shape[:3]

    for pam in pams:
        if pam.peak_dirs.shape[:3] != base_shape:
            return False

    if not ref_img_shape:
        return True

    return base_shape == ref_img_shape and sync_imgs
