import warnings

import numpy as np
from scipy.ndimage import label


def remove_holes_and_islands(binary_img, *, slice_wise=False):
    """
    Remove any small mask chunks or holes
    that could be in the segmentation output.

    Parameters
    ----------
    binary_img : np.ndarray
        Binary image
    slice_wise : bool, optional
        Whether to run slice wise background correction as well

    Returns
    -------
    largest_img : np.ndarray
    """
    if len(np.unique(binary_img)) != 2:
        warnings.warn(
            "The mask is not binary. \
                      Returning the original mask",
            UserWarning,
            stacklevel=2,
        )
        return binary_img

    largest_img = np.zeros_like(binary_img)
    chunks, _ = label(np.abs(1 - binary_img))
    u, c = np.unique(chunks[chunks != 0], return_counts=True)
    if len(u) == 0:
        warnings.warn(
            "The mask has no background. \
                      Returning the original mask",
            UserWarning,
            stacklevel=2,
        )
        return binary_img
    target = u[np.argmax(c)]

    largest_img = np.where(chunks == target, 0, 1)

    chunks, _ = label(largest_img)
    u, c = np.unique(chunks[chunks != 0], return_counts=True)
    if len(u) == 0:
        warnings.warn(
            "The mask has no foreground. \
                      Returning the original mask",
            UserWarning,
            stacklevel=2,
        )
        return binary_img
    target = u[np.argmax(c)]

    largest_img = np.where(chunks == target, 1, 0)

    if slice_wise:
        for x in range(largest_img.shape[0]):
            chunks, n_chunk = label(np.abs(1 - largest_img[x]))
            u, c = np.unique(chunks[chunks != 0], return_counts=True)
            target = u[np.argmax(c)]
            largest_img[x] = np.where(chunks == target, 0, 1)
        for y in range(largest_img.shape[1]):
            chunks, n_chunk = label(np.abs(1 - largest_img[:, y]))
            u, c = np.unique(chunks[chunks != 0], return_counts=True)
            target = u[np.argmax(c)]
            largest_img[:, y] = np.where(chunks == target, 0, 1)
        for z in range(largest_img.shape[2]):
            chunks, n_chunk = label(np.abs(1 - largest_img[..., z]))
            u, c = np.unique(chunks[chunks != 0], return_counts=True)
            target = u[np.argmax(c)]
            largest_img[..., z] = np.where(chunks == target, 0, 1)

    return largest_img
