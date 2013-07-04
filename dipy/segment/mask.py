import numpy as np
import copy
from scipy.ndimage import binary_opening, label
from scipy.ndimage.filters import median_filter

def dwi_bet(input_volume, median_radius=4, numpass=4, autocrop=True):
    """
    Simple brain extraction tool (BET) method for DWI data. It uses a median filter 
    smoothing of the input_volume and an automatic histogram Otsu thresholding technique. 

    It mimics the MRtrix bet from the documentation.
    (mrconvert dwi.nii -coord 3 0 - | threshold - - | median3D - - | median3D - mask.nii)
    MRtrix uses default mean_radius=3 and numpass=2
    
    However, from tests on multiple 1.5T and 3T data from GE, Philips, Siemens, the most
    robust choice is median_radis=4, numpass=4
    Parameters
    ----------
    input_volume : 3D or 4D ndarray
        3D ndarray if b=0 volume is provide, 4D ndarray if whole DWI data is provided
    median_radius : float
        Radius of the applied median filter (default 4)
    numpass: int
        Number of pass of the median filter (default 4)
    autocrop: bool, optional
        if True, the masked input_volume will also be cropped using the bounding box
        defined by the masked data. Should be on if DWI is upsampled to 1x1x1 resolution.

    Returns
    -------
    input_volume : 3D or 4D ndarray
        Masked input_volume
    mask : 3D ndarray
        The binary brain mask
    """

    # The original data will be needed for final crop / mask
    vol = input_volume.copy()

    # Use only first 3D slice. Should have an automatic way to detect b=0 images
    # average them and compute mask vol from it.
    if len(vol.shape) > 3:
        vol = vol[:,:,:,0]

    # Make a mask using a multiple pass median filter and histogram thresholding.
    mask = multi_median(vol, median_radius, numpass)
    thresh = otsu(mask)
    threshold2(mask, thresh)
    
    # Auto crop the volumes using the mask as input_volume for bounding box computing.
    if autocrop:
        mins, maxs = bounding_box(mask)
        mask = crop(mask, mins, maxs)
        input_volume = crop(input_volume, mins, maxs)

    # Apply the cropped mask to the cropped original volume.
    applymask(input_volume, mask)

    return input_volume, mask


def multi_median(input, median_radius, numpass):
    """
    Applies a median filter with median_radius numpass times on input.
    
    """
    outvol = np.zeros_like(input, dtype=input.dtype)
    
    # Array representing the size of the median window in each dimension.
    medarr = np.ones_like(input.shape) * ((median_radius * 2) +1)

    # Multi pass
    for i in range(0, numpass):
        median_filter(input, medarr, output=input)

    return input

def otsu(image, nbins=256):
    """
    Automatic histogram thresholding
    """

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


def applymask(vol, mask):
    """
    Recursively applies N dimensionnal mask to a M dimensionnal volume
    for N <= M.
    """
    if len(mask.shape) > len(vol.shape):
        raise Exception('applymask: The mask\'s dimmensionnality is bigger than the input\'s')

    elif len(mask.shape) > len(vol.shape):
        lastdimelen = vol.shape[len(vol.shape)-1]
        for i in range(0,lastdimlen):
            applymask(vol[..., i], mask)
    else:
        outliers = np.where(mask == 0)
        outliers = np.array(outliers)
        outliers = tuple(outliers)
        vol[outliers] = 0


def threshold2(vol, thresh):
    """
    Simple binary thresholding of vol bigger or equal to thresh
    """
    maxval = maxvalue(vol.dtype)
    for x in np.nditer(vol, flags=['external_loop','buffered'],
                       op_flags=['readwrite'], order='F'):

        x[np.where(x >= thresh)] = maxval
        x[np.where(x < thresh)] = 0

def maxvalue(datatype):
    if datatype.kind in 'iu':
        return np.iinfo(datatype.type).max
    else:
        return np.finfo(datatype.type).max

def bounding_box(vol):
    """
    Compute the bounding box of nonzero intensity voxels of vol
    """
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

def crop(vol, mins, maxs):
    return vol[mins[0]:maxs[0],mins[1]:maxs[1],mins[2]:maxs[2]]
