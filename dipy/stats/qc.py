import numpy as np
from dipy.core.geometry import cart_distance


def find_qspace_neighbors(gtab):
    """Create a mapping of dwi volume index to its nearest neighbor.

    An approximate q-space is used (the deltas are not included).
    Note that neighborhood is not necessarily bijective. One neighbor
    is found per dwi volume.

    Parameters
    ----------

    gtab: dipy.core.gradients.GradientTable
        Gradient table.

    Returns
    -------

    neighbors: list of tuple
        A list of 2-tuples indicating the nearest q-space neighbor
        of each dwi volume.

    Examples
    --------
    >>> from dipy.core.gradients import gradient_table
    >>> import numpy as np
    >>> gtab = gradient_table(
    ...     np.array([0, 1000, 1000, 2000]),
    ...     np.array([
    ...         [1, 0, 0],
    ...         [1, 0, 0],
    ...         [0.99, 0.0001, 0.0001],
    ...         [1, 0, 0]]))
    >>> find_qspace_neighbors(gtab)
    [(1, 2), (2, 1), (3, 1)]

    """
    dwi_neighbors = []

    # Only correlate the b>0 images
    dwi_mask = np.logical_not(gtab.b0s_mask)
    dwi_indices = np.flatnonzero(dwi_mask)

    # Get a pseudo-qspace value for b>0s
    qvecs = np.sqrt(gtab.bvals)[:, np.newaxis] * gtab.bvecs

    for dwi_index in dwi_indices:
        qvec = qvecs[dwi_index]

        # Calculate distance in q-space, accounting for symmetry
        pos_dist = cart_distance(qvec[np.newaxis, :], qvecs)
        neg_dist = cart_distance(qvec[np.newaxis, :], -qvecs)
        distances = np.min(np.column_stack([pos_dist, neg_dist]), axis=1)

        # Be sure we don't select the image as its own neighbor
        distances[dwi_index] = np.inf
        # Or a b=0
        distances[gtab.b0s_mask] = np.inf
        neighbor_index = np.argmin(distances)
        dwi_neighbors.append((dwi_index, neighbor_index))

    return dwi_neighbors


def neighboring_dwi_correlation(dwi_data, gtab, mask=None):
    """Calculate the Neighboring DWI Correlation (NDC) from dMRI data.

    Using a mask is highly recommended, otherwise the FOV will influence the
    correlations. According to [Yeh2019], an NDC less than 0.4 indicates a
    low quality image.

    Parameters
    ----------
    dwi_data : 4D ndarray
        dwi data on which to calculate NDC
    gtab : dipy.core.gradients.GradientTable
        Gradient table.
    mask : 3D ndarray, optional
        optional mask of voxels to include in the NDC calculation

    Returns
    -------
    ndc : float
        The neighboring DWI correlation

    References
    ----------

    .. [Yeh2019] Yeh, Fang-Cheng, et al. "Differential tractography as a
                 track-based biomarker for neuronal injury."
                 NeuroImage 202 (2019): 116131.

    """

    neighbor_indices = find_qspace_neighbors(gtab)
    neighbor_correlations = []

    if mask is not None:
        binary_mask = mask > 0

    for from_index, to_index in neighbor_indices:

        # Flatten the dwi images
        if mask is not None:
            flat_from_image = dwi_data[..., from_index][binary_mask]
            flat_to_image = dwi_data[..., to_index][binary_mask]
        else:
            flat_from_image = dwi_data[..., from_index].flatten()
            flat_to_image = dwi_data[..., to_index].flatten()

        neighbor_correlations.append(
            np.corrcoef(flat_from_image, flat_to_image)[0, 1])

    return np.mean(neighbor_correlations)
