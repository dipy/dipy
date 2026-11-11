import numpy as np

from dipy.core.geometry import cart_distance
from dipy.utils.deprecator import warning_for_keywords


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
    ...     bvecs=np.array([
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


@warning_for_keywords()
def neighboring_dwi_correlation(dwi_data, gtab, *, mask=None):
    """Calculate the Neighboring DWI Correlation (NDC) from dMRI data.

    Using a mask is highly recommended, otherwise the FOV will influence the
    correlations. According to :footcite:t:`Yeh2019`, an NDC less than 0.4
    indicates a low quality image.

    Parameters
    ----------
    dwi_data : 4D ndarray
        dwi data on which to calculate NDC
    gtab : dipy.core.gradients.GradientTable
        Gradient table.
    mask : 3D ndarray, optional
        Mask of voxels to include in the NDC calculation

    Returns
    -------
    ndc : float
        The neighboring DWI correlation

    References
    ----------
    .. footbibliography::

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

        neighbor_correlations.append(np.corrcoef(flat_from_image, flat_to_image)[0, 1])

    return np.mean(neighbor_correlations)


def find_qspace_contrast(gtab):
    """Create a mapping of DWI volume index to its contrast DWI.

    For each DWI volume, the contrast DWI is selected using the q-space
    criterion described by Yeh. For each candidate q-space vector, its
    component perpendicular to the reference q-space vector is calculated
    and normalized to the magnitude of the reference vector. The candidate
    closest to this normalized perpendicular vector is selected.

    Parameters
    ----------
    gtab : dipy.core.gradients.GradientTable
        Gradient table.

    Returns
    -------
    contrast : list of tuple
        A list of 2-tuples indicating the contrast DWI for each
        DWI volume.
    """
    dwi_contrast = []

    # Only compare the b>0 images
    dwi_mask = np.logical_not(gtab.b0s_mask)
    dwi_indices = np.flatnonzero(dwi_mask)

    # Approximate q-space coordinates
    qvecs = np.sqrt(gtab.bvals)[:, np.newaxis] * gtab.bvecs

    for dwi_index in dwi_indices:
        qvec = qvecs[dwi_index]

        qvec_norm_sq = np.dot(qvec, qvec)
        qvec_norm = np.sqrt(qvec_norm_sq)

        min_distance = np.inf
        contrast_index = None

        for candidate_index in dwi_indices:
            if candidate_index == dwi_index:
                continue

            candidate = qvecs[candidate_index]

            # Projection of the candidate q-vector onto the
            # reference q-vector.
            parallel = qvec * (np.dot(qvec, candidate) / qvec_norm_sq)

            # Component of the candidate perpendicular to the
            # reference q-vector.
            perpendicular = candidate - parallel

            # A parallel candidate has no defined perpendicular
            # direction and cannot serve as the contrast DWI.
            if np.allclose(perpendicular, 0.0):
                continue

            # Normalize the perpendicular component to the magnitude
            # of the reference q-vector, matching Yeh's implementation.
            perpendicular_norm = np.linalg.norm(perpendicular)
            perpendicular *= qvec_norm / perpendicular_norm

            distance = np.linalg.norm(candidate - perpendicular)

            if distance < min_distance:
                min_distance = distance
                contrast_index = candidate_index

        if contrast_index is None:
            raise ValueError(
                "Unable to find a contrast DWI for volume "
                f"{dwi_index}. At least one non-parallel DWI "
                "direction is required."
            )

        dwi_contrast.append((dwi_index, contrast_index))

    return dwi_contrast


def dwi_contrast(dwi_data, gtab, *, mask=None):
    """Calculate the DWI contrast quality metric from dMRI data.

    Refer to https://dsi-studio.labsolver.org/doc/gui_t1.html#step-t1a-quality-control-optional
    And see :footcite:p:`Yeh2025` for further details about DSI Studio which originated the metric.

    For each DWI volume, the voxel-wise correlation with its nearest
    q-space neighbor and with its contrast DWI are calculated. The DWI
    contrast metric is the mean neighboring DWI correlation divided by
    the mean contrast DWI correlation.

    The contrast DWI is selected by finding the candidate closest to a
    normalized q-space vector perpendicular to the reference DWI.

    Using a mask is highly recommended, otherwise the FOV will influence
    the correlations.

    Conventional thresholds are:

    * < 1.1: Poor
    * 1.1-1.3: Fair
    * > 1.3: Good

    Parameters
    ----------
    dwi_data : 4D ndarray
        DWI data on which to calculate DWI contrast.
    gtab : dipy.core.gradients.GradientTable
        Gradient table.
    mask : 3D ndarray, optional
        Mask of voxels to include in the DWI contrast calculation.

    Returns
    -------
    contrast : float
        DWI contrast, calculated as the mean neighboring DWI
        correlation divided by the mean contrast DWI correlation.

    References
    ----------
    .. footbibliography::

    """
    neighbor_indices = find_qspace_neighbors(gtab)
    contrast_indices = find_qspace_contrast(gtab)

    neighbor_correlations = []
    contrast_correlations = []

    if mask is not None:
        binary_mask = mask > 0

    for (from_index, neighbor_index), (
        contrast_from_index,
        contrast_index,
    ) in zip(neighbor_indices, contrast_indices):
        if from_index != contrast_from_index:
            raise RuntimeError(
                "Neighbor and contrast q-space mappings are inconsistent."
            )

        if mask is not None:
            flat_from_image = dwi_data[..., from_index][binary_mask]
            flat_neighbor_image = dwi_data[..., neighbor_index][binary_mask]
            flat_contrast_image = dwi_data[..., contrast_index][binary_mask]
        else:
            flat_from_image = dwi_data[..., from_index].flatten()
            flat_neighbor_image = dwi_data[..., neighbor_index].flatten()
            flat_contrast_image = dwi_data[..., contrast_index].flatten()

        neighbor_correlation = np.corrcoef(
            flat_from_image,
            flat_neighbor_image,
        )[0, 1]

        contrast_correlation = np.corrcoef(
            flat_from_image,
            flat_contrast_image,
        )[0, 1]

        neighbor_correlations.append(neighbor_correlation)
        contrast_correlations.append(contrast_correlation)

    return np.mean(neighbor_correlations) / np.mean(contrast_correlations)
