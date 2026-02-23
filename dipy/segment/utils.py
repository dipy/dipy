import warnings

import numpy as np
from scipy.ndimage import label


def remove_holes_and_islands(
    binary_img, *, remove_holes=True, remove_islands=True, slice_wise=False
):
    """
    Remove any small mask chunks or holes
    that could be in the segmentation output.

    Parameters
    ----------
    binary_img : np.ndarray
        Binary image
    slice_wise : bool, optional
        Whether to run slice wise background correction as well
    remove_holes : bool, optional
        Whether to remove holes from the binary image
    remove_islands : bool, optional
        Whether to remove islands from the binary image

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

    if not remove_holes and not remove_islands:
        warnings.warn(
            "Both remove_holes and remove_islands is False. \
                      Returning the original mask",
            UserWarning,
            stacklevel=2,
        )
        return binary_img

    shape = binary_img.shape
    new_binary_img = binary_img.copy()

    if remove_islands:
        components, _ = label(new_binary_img)
        new_binary_img = components == np.argmax(np.bincount(components.flat)[1:]) + 1
        new_binary_img = new_binary_img.astype(int)
    if remove_holes:
        components, _ = label(1 - new_binary_img)
        new_binary_img = components == np.argmax(np.bincount(components.flat)[1:]) + 1
        new_binary_img = 1 - new_binary_img.astype(int)

    if slice_wise:
        for x in range(shape[0]):
            components, _ = label(1 - new_binary_img[x])
            temp = components == np.argmax(np.bincount(components.flat)[1:]) + 1
            new_binary_img[x] = 1 - temp.astype(int)
        for y in range(shape[1]):
            components, _ = label(1 - new_binary_img[:, y])
            temp = components == np.argmax(np.bincount(components.flat)[1:]) + 1
            new_binary_img[:, y] = 1 - temp.astype(int)
        for z in range(shape[2]):
            components, _ = label(1 - new_binary_img[..., z])
            temp = components == np.argmax(np.bincount(components.flat)[1:]) + 1
            new_binary_img[..., z] = 1 - temp.astype(int)

    return new_binary_img
