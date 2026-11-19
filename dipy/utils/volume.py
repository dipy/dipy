"""Utilities for image volumes: volume selection, voxel adjacencies and
neighbourhoods."""

import numpy as np
from scipy.spatial.distance import pdist, squareform

from dipy.utils.deprecator import warning_for_keywords


@warning_for_keywords()
def adjacency_calc(img_shape, *, mask=None, cutoff=1.99):
    """Create adjacency list for voxels, accounting for the mask.

    Parameters
    ----------

    img_shape : list
        Spatial shape of the image data.

    mask : array, optional
        A boolean array used to mark the coordinates in the data that
        should be analyzed that should have the same shape as the images.

    cutoff : float, optional
        Cutoff distance in voxel coordinates for finding adjacent voxels.
        e.g. cutoff=2 gives a 3x3 patch, cutoff=2.1 gives a 3x3 patch
        with additional 4 voxels above, below, left, right of this patch.

    Returns
    -------

    adj : list
        List, one entry per voxel, giving the indices of adjacent voxels.
        The list will correspond to the data array once masked and flattened.

    """

    # Image-space coordinates of voxels, flattened
    XYZ = np.meshgrid(*[range(ds) for ds in img_shape], indexing="ij")
    XYZ = np.column_stack([xyz.ravel() for xyz in XYZ])
    dists = squareform(pdist(XYZ))
    dists = dists < cutoff  # adjacency list contains current voxel
    adj = []
    if mask is not None:
        flat_mask = mask.reshape(-1)
        for idx in range(dists.shape[0]):
            if flat_mask[idx]:
                cond = dists[idx, :]
                cond = cond * flat_mask
                cond = cond[flat_mask == 1]  # so indices will match masked array
                adj.append(np.argwhere(cond).flatten().tolist())
    else:
        for idx in range(dists.shape[0]):
            cond = dists[idx, :]
            adj.append(np.argwhere(cond).flatten().tolist())

    return adj


def select_3d_volume(image, *, index=0, name="image"):
    """Select a 3D volume from a 3D or 4D array.

    Parameters
    ----------
    image : ndarray (X, Y, Z) or (X, Y, Z, N)
        Input array.
    index : int, optional
        Index, along the last axis, of the volume to select when `image` is
        4D. Ignored when `image` is 3D.
    name : str, optional
        Name of the image, used in error messages.

    Returns
    -------
    volume : ndarray (X, Y, Z)
        `image` itself when it is 3D, otherwise ``image[..., index]``.

    Raises
    ------
    ValueError
        If `image` is not 3D or 4D, or if `index` is out of range.
    TypeError
        If `image` is 4D and `index` is not an integer.
    """
    image = np.asarray(image)
    if image.ndim == 3:
        return image
    if image.ndim != 4:
        raise ValueError(f"{name} must be a 3D or 4D array. Got shape {image.shape}.")
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise TypeError(f"The volume index of {name} must be an integer.")
    n_volumes = image.shape[-1]
    if not 0 <= index < n_volumes:
        raise ValueError(
            f"The volume index ({index}) must be in [0, {n_volumes - 1}] for "
            f"{name} with {n_volumes} volumes."
        )
    return image[..., int(index)]
