import numpy as np
from dipy.core.geometry import cart_distance


def neighboring_dwi_correlation(dwi_data, gtab, mask=None):
    r"""Calculate the Neighboring DWI Correlation (NDC) from dMRI data.

    Using a mask is highly recommended, otherwise the FOV will influence the
    correlations. According to [Yeh2019], an NDC less than 0.4 indicates a
    low quality image.

    Parameters
    ----------
    dwi_data : 4D ndarray
        dwi data on which to calculate NDC
    gtab : dipy.core.gradients.GradientTable
        Gradient table.
    mask : 3D ndarray
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

    # Only correlate the b>0 images
    dwi_mask = np.logical_not(gtab.b0s_mask)
    dwi_indices = np.flatnonzero(dwi_mask)

    # Load the data into vectors
    if mask is not None:
        vectorized_images = [
            dwi_data[..., idx][mask > 0] for idx in dwi_indices]
    else:
        vectorized_images = [
            dwi_data[..., idx].flatten() for idx in dwi_indices]

    # Get a pseudo-qspace value for b>0s
    qvecs = (np.sqrt(gtab.bvals)[:, np.newaxis] * gtab.bvecs)[dwi_mask]

    neighbor_correlations = []
    for qcoord_index, qvec in enumerate(qvecs):

        # Calculate distance in q-space, accounting for symmetry
        pos_dist = cart_distance(qvec[np.newaxis, :], qvecs)
        neg_dist = cart_distance(qvec[np.newaxis, :], -qvecs)
        distances = np.min(np.column_stack([pos_dist, neg_dist]), axis=1)

        # Be sure we don't select the image as its own neighbor
        distances[qcoord_index] = np.inf

        neighbor_index = np.argmin(distances)
        neighbor_correlations.append(
            np.corrcoef(vectorized_images[qcoord_index],
                        vectorized_images[neighbor_index])[0, 1])

    return np.mean(neighbor_correlations)
