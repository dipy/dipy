import numpy as np
from scipy.ndimage import binary_opening, label


def hist_mask(mean_volume, reference_volume=None, m=0.2, M=0.9,
              cc=True, opening=2, exclude_zeros=False):
    """
    Compute a mask file from dMRI or other EPI data. Useful for brain 
    extraction or any foreground extraction in 3D volumes.

    Compute and write the mask of an image based on the grey level
    
    Parameters
    ----------
    mean_volume : 3D ndarray
        mean EPI image, used to compute the threshold for the mask.
    reference_volume: 3D ndarray, optional
        reference volume used to compute the mask. If none is give, the
        mean volume is used.
    m : float, optional
        lower fraction of the histogram to be discarded.
    M: float, optional
        upper fraction of the histogram to be discarded.
    cc: boolean, optional
        if cc is True, only the largest connect component is kept.
    opening: int, optional
        if opening is larger than 0, an morphological opening is performed,
        to keep only large structures. This step is useful to remove parts of
        the skull that might have been included.
    exclude_zeros: boolean, optional
        Consider zeros as missing values for the computation of the
        threshold. This option is useful if the images have been
        resliced with a large padding of zeros.

    Returns
    -------
    mask : 3D boolean ndarray
        The brain mask

    Example
    -------
    >>> from scipy.ndimage import generate_binary_structure, binary_dilation
    >>> from dipy.segment.mask import hist_mask
    >>> vol = np.zeros((30, 30, 30))
    >>> vol[15, 15, 15] = 1
    >>> struct = generate_binary_structure(3, 1)
    >>> voln = binary_dilation(vol, structure=struct, iterations=4).astype('f4')
    >>> voln = 5 * voln + np.random.random(voln.shape)
    >>> mask = hist_mask(voln, m=0.9, M=.99)

    Notes
    -----
    This is based on an heuristic proposed by T.Nichols:

    Find the least dense point of the histogram, between fractions
    m and M of the total image histogram. In case of failure, it 
    is usually advisable to increase m.

    """
    if reference_volume is None:
        reference_volume = mean_volume
    sorted_input = np.sort(mean_volume.reshape(-1))
    if exclude_zeros:
        sorted_input = sorted_input[sorted_input != 0]
    limiteinf = np.floor(m * len(sorted_input))
    limitesup = np.floor(M * len(sorted_input))

    delta = sorted_input[limiteinf + 1:limitesup + 1] \
        - sorted_input[limiteinf:limitesup]
    ia = delta.argmax()
    threshold = 0.5 * (sorted_input[ia + limiteinf]
                       + sorted_input[ia + limiteinf + 1])

    mask = (reference_volume >= threshold)

    if cc:
        mask = largest_cc(mask)

    if opening > 0:
        mask = binary_opening(mask.astype(np.int),
                              iterations=opening)
    return mask.astype(bool)


def largest_cc(mask):
    """ Return the largest connected component of a 3D mask array.

    Parameters
    -----------
    mask: 3D boolean array
          3D array indicating a mask.

    Returns
    --------
    mask: 3D boolean array
          3D array indicating a mask, with only one connected component.
    """
    # We use asarray to be able to work with masked arrays.
    mask = np.asarray(mask)
    labels, label_nb = label(mask)
    if not label_nb:
        raise ValueError('No non-zero values: no connected components')
    if label_nb == 1:
        return mask.astype(np.bool)
    label_count = np.bincount(labels.ravel())
    # discard 0 the 0 label
    label_count[0] = 0
    return labels == label_count.argmax()
