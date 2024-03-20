import warnings

import numpy as np
from scipy.ndimage import label


def remove_holes_and_islands(binary_img):
    """
    Remove any small mask chunks or holes
    that could be in the segmentation output.

    Parameters
    ----------
    binary_img : np.ndarray
        Binary image

    Returns
    -------
    largest_img : np.ndarray
    """
    largest_img = np.zeros_like(binary_img)
    chunks, _ = label(np.abs(1 - binary_img))
    u, c = np.unique(chunks[chunks != 0], return_counts=True)
    try:
        target = u[np.argmax(c)]
    except ValueError:
        warnings.warn('The mask has no background. \
                      Returning the original mask',
                      UserWarning, stacklevel=2)
        return binary_img

    largest_img = np.where(chunks == target, 0, 1)

    chunks, _ = label(largest_img)
    u, c = np.unique(chunks[chunks != 0], return_counts=True)
    try:
        target = u[np.argmax(c)]
    except ValueError:
        warnings.warn('The mask has no foreground. \
                      Returning the original mask',
                      UserWarning, stacklevel=2)
        return binary_img

    largest_img = np.where(chunks == target, 1, 0)

    return largest_img
