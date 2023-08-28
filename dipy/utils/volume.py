""" Utilities for calculating voxel adjacencies and neighbourhoods """

import numpy as np
from scipy.spatial.distance import pdist, squareform


def adjacency_calc(img_shape, mask=None, cutoff=1.99):
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
    XYZ = np.meshgrid(*[range(ds) for ds in img_shape], indexing='ij')
    XYZ = np.column_stack([xyz.ravel() for xyz in XYZ])
    dists = squareform(pdist(XYZ))
    dists = (dists < cutoff)  # adjacency list contains current voxel
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

