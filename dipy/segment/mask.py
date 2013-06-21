import numpy as np
import copy
from scipy.ndimage import binary_opening, label
from scipy.ndimage.filters import median_filter

def hist_mask(mean_volume, reference_volume=None, m=0.2, M=0.9,
              cc=True, opening=2, exclude_zeros=False):
    """
    Compute a mask file from dMRI or other Echo Planar imaging (EPI) data.
    Useful for brain extraction or any foreground extraction in 3D volumes.

    Compute and write the mask of an image based on the grey level
    Parameters
    ----------
    mean_volume : 3D ndarray
        mean EPI image, used to compute the threshold for the mask.
    reference_volume: 3D ndarray, optional
        reference volume used to compute the mask. If None is give, the
        mean volume is used.
    m : float, optional
        lower fraction of the histogram to be discarded.
    M: float, optional
        upper fraction of the histogram to be discarded.
    cc: bool, optional
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

    #delta = np.diff(np.percentile(sorted_input, [m, M]))
    #the line above is the same as 
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
    # discard the 0 label
    label_count[0] = 0
    return labels == label_count.argmax()


# Very simple brain extraction tool (BET) method for DWI data
# Inspired by MRtrix
# mrconvert dwi.nii -coord 3 0 - | threshold - - | median3D - - | median3D - mask.nii 
# MRtrix uses default mean_radius=3 and numpass=2
# However, from my tests on 1.5T and 3T data from GE, Philips, Siemens, the most
# robust choice is median_radis=4, numpass=4
def dwi_bet_filter(input, median_radius=4, numpass=4, autocrop=True):
    # The original data will be needed for final crop / mask
    vol = input.copy()

    # Use only first 3D slice
    if len(vol.shape) > 3:
        vol = vol[:,:,:,0]

    # Make a mask using multi pass median filter and histogram thresholding.
    mask = _multi_median(vol, median_radius, numpass)
    thresh = _otsu(mask)
    _threshold2(mask, thresh)

    # Auto crop the volumes using the mask as input for bounding box computing.
    if autocrop:
        mins, maxs = _bounding_box(mask)
        mask = _crop(mask, mins, maxs)
        input = _crop(input, mins, maxs)

    # Apply the cropped mask to the cropped original volume.
    _applymask(input, mask)

    return input, mask

# Applies a median filter with median_radius numpass times on input.
def _multi_median(input, median_radius, numpass):
    outvol = np.zeros_like(input, dtype=input.dtype)
    
    # Array representing the size of the median window in each dimension.
    medarr = np.ones_like(input.shape) * ((median_radius * 2) +1)

    # Multi pass
    for i in range(0, numpass):
        median_filter(input, medarr, output=input)

    return input

# histogram thresholding
def _otsu(image, nbins=256):
    hist, bin_centers = np.histogram(image, nbins)
    hist = hist.astype(float)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers[1:]) / weight1
    mean2 = (np.cumsum((hist * bin_centers[1:])[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:])**2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold

# Recursively applies N dimensionnal mask to a M dimensionnal volume
# for N <= M.
def _applymask(vol, mask):
    if len(mask.shape) > len(vol.shape):
        raise Exception('applymask: The mask\'s dimmensionnality is bigger than the input\'s')

    elif len(mask.shape) > len(vol.shape):
        lastdimelen = vol.shape[len(vol.shape)-1]
        for i in range(0,lastdimlen):
            _applymask(vol[..., i], mask)
    else:
        outliers = np.where(mask == 0)
        outliers = np.array(outliers)
        outliers = tuple(outliers)
        vol[outliers] = 0

# -
# These methods were copied or derived from the pypavi library.
# This was done to make this module as close as possible to being 
# standalone(Numpy and scipy are still needed though).
# PAVI : http://pavi.dinf.usherb.ca/
# -
def _threshold2(vol, thresh):
    maxvalue = _maxvalue(vol.dtype)
    for x in np.nditer(vol, flags=['external_loop','buffered'],
                       op_flags=['readwrite'], order='F'):

        x[np.where(x >= thresh)] = maxvalue
        x[np.where(x < thresh)] = 0

def _maxvalue(datatype):
    if datatype.kind in 'iu':
        return np.iinfo(datatype.type).max
    else:
        return np.finfo(datatype.type).max

def _bounding_box(vol):
    pts = np.array(np.where(vol != 0)).T
    if len(pts) == 0:
        print 'WARNING: Not data found in volume to bound. Returning empty bounding box.'
        return [0,0,0], [0,0,0]

    maxs = copy.copy(pts[0])
    mins = copy.copy(pts[0])
    numdims = len(pts[0])

    for pt in pts:
        for curdim in range(0, numdims):
            if pt[curdim] > maxs[curdim]:
                maxs[curdim] = copy.copy(pt[curdim])

            if pt[curdim] < mins[curdim]:
                mins[curdim] = copy.copy(pt[curdim])

    npmaxs = np.array(maxs)
    npmins = np.array(mins)
    return npmins, npmaxs

def _crop(vol, mins, maxs):
    return vol[mins[0]:maxs[0],mins[1]:maxs[1],mins[2]:maxs[2]]
