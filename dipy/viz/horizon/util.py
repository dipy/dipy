import logging

import numpy as np

from dipy.testing.decorators import warning_for_keywords


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
        data, _, _ = unpack_data(img, return_size=3)
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
        data, affine, fname = unpack_data(img, return_size=3)
        if np.issubdtype(data.dtype, np.integer):
            if data.dtype != np.int32:
                msg = f"{data.dtype} is not supported, falling back to int32"
                logging.warning(msg)
                img = (data.astype(np.int32), affine, fname)
            valid_images.append(img)
        elif np.issubdtype(data.dtype, np.floating):
            if data.dtype != np.float64 and data.dtype != np.float32:
                msg = f"{data.dtype} is not supported, falling back to float32"
                logging.warning(msg)
                img = (data.astype(np.float32), affine, fname)
            valid_images.append(img)
        else:
            msg = f"skipping image {idx + 1}, passed image is not in numerical format"
            logging.warning(msg)

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
        ellipsis_text = f"...{text[-(max_chars - 3) :]}"
        return ellipsis_text
    return text


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
    data = unpack_data(surface)

    if data[0].shape[-1] != 3:
        raise ValueError(f"Vertices do not have correct shape: {data[0].shape}")
    if data[1].shape[-1] != 3:
        raise ValueError(f"Faces do not have correct shape: {data[1].shape}")
    return data


def unpack_data(data, *, return_size=3):
    if not isinstance(data, tuple):
        data = (data, None)
    if len(data) < return_size:
        result = [*data]
        result += (return_size - len(data)) * [None]
        return result

    return data


@warning_for_keywords()
def check_peak_size(pams, *, ref_img_shape=None, sync_imgs=False):
    """Check shape of peaks.

    Parameters
    ----------
    pams : tuple
        (PeaksAndMetrics, fname).
    ref_img_shape : tuple, optional
        3D shape of the image, by default None.
    sync_imgs : bool, optional
        True if the images are synchronized, by default False.

    Returns
    -------
    bool
        If the peaks are aligned with images and other peaks.
    """
    base_data = unpack_data(pams[0], return_size=2)
    base_shape = base_data[0].peak_dirs.shape[:3]

    for pam, _ in pams:
        if pam.peak_dirs.shape[:3] != base_shape:
            return False

    if not ref_img_shape:
        return True

    return base_shape == ref_img_shape and sync_imgs
