from __future__ import division, print_function, absolute_import
import copy
import numpy as np
from scipy.ndimage import binary_opening, label
from scipy.ndimage.filters import median_filter
from scipy.stats import threshold

def multi_median(input, median_radius, numpass):
    """
    Applies multiple times scikit-image's median filter on input data.

    Parameters
    ----------
    input : ndarray
        The input volume to apply filter on.
    median_radius : int
        Radius (in voxels) of the applied median filter 
    numpass: int
        Number of pass of the median filter
    Returns
    -------
    input : ndarray
        Filtered input volume.
    """
    outvol = np.zeros_like(input, dtype=input.dtype)
    
    # Array representing the size of the median window in each dimension.
    medarr = np.ones_like(input.shape) * ((median_radius * 2) + 1)
    
    # Multi pass
    for i in range(0, numpass):
        median_filter(input, medarr, output=input)

    return input


def otsu(image, nbins=256):
    """
    Return threshold value based on Otsu's method.
    Copied from scikit-image to remove dependency.

    Parameters
    ----------
    image : array
        Input image.
    nbins : int
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.

    Returns
    -------
    threshold : float
        Threshold value.
    """
    hist, bin_centers = np.histogram(image, nbins)
    hist = hist.astype(np.float)

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
    Mask vol with mask.

    Parameters
    ----------
    vol : ndarray
        Volume to apply mask on.
    mask : ndarray
        Binary mask.
    """
    mask = mask.reshape(mask.shape + (vol.ndim - mask.ndim) * (1,))
    return vol * mask

def binary_threshold(vol, thresh):
    """
    Simple binary thresholding.

    Parameters
    ----------
    vol : ndarray
        Volume to apply threshold on.
    thresh : float
        Thresholding value.

    Returns
    -------
        Binary ndarray.
    """
    return np.where(vol > thresh, True, False)

def bounding_box(vol):
    """
    Compute the bounding box of nonzero intensity voxels in the volume.

    Parameters
    ----------
        vol : ndarray
            Volume to compute bounding box on.

    Returns
    -------
    npmins : array
        Array containg minimum index of each dimension
    npmaxs : array
        Array containg maximum index of each dimension
    """

    pts = np.array(np.where(vol != 0)).T
    if len(pts) == 0:
        print('WARNING: Not data found in volume to bound. Returning empty bounding box.')
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
    """
    Crops the input volume.

    Parameters
    ----------
    vol : 3D ndarray
        Volume to crop.
    mins : array
        Array containg minimum index of each dimension.
    maxs : array
        Array containg maximum index of each dimension.

    Returns
    -------
        vol : 3D ndarray
            The cropped volume.
    """
    return vol[tuple(slice(i, j+1) for i, j in zip(mins, maxs))]

def medotsu(input_volume, median_radius=4, numpass=4, autocrop=False, b0Slices=None):
    """
    Simple brain extraction tool method for images from DWI data. It uses a
    median filter smoothing of the input_volumes b0Slices and an automatic
    histogram Otsu thresholding technique, hence the name medotsu.

    It mimics the MRtrix bet from the documentation.
    (mrconvert dwi.nii -coord 3 0 - | threshold - - | median3D - - | median3D - mask.nii)
    MRtrix uses default mean_radius=3 and numpass=2

    However, from tests on multiple 1.5T and 3T data from
    GE, Philips, Siemens, the most robust choice is median_radius=4, numpass=4

    Parameters
    ----------
    input_volume : ndarray
        ndarray of the brain volume
    median_radius : int
        Radius (in voxels) of the applied median filter(default 4)
    numpass: int
        Number of pass of the median filter (default 4)
    autocrop: bool, optional
        if True, the masked input_volume will also be cropped using the bounding
        box defined by the masked data. Should be on if DWI is upsampled to 1x1x1
        resolution. (default False)
    b0Slices : array
        1D array representing indexes of the volume where b=0

    Returns
    -------
    maskedvolume : ndarray
        Masked input_volume
    mask : 3D ndarray
        The binary brain mask
    """

    if len(input_volume.shape) == 4:
        if b0Slices <> None:
            b0vol = np.mean(input_volume[..., tuple(b0Slices)], axis=3)
        else:
            b0vol = input_volume[..., 0].copy()
    else:
        b0vol = input_volume.copy()

    # Make a mask using a multiple pass median filter and histogram thresholding.
    mask = multi_median(b0vol, median_radius, numpass)
    thresh = otsu(mask)
    mask = binary_threshold(mask, thresh)

    # Auto crop the volumes using the mask as input_volume for bounding box computing.
    if autocrop:
        mins, maxs = bounding_box(mask)
        mask = crop(mask, mins, maxs)
        croppedvolume = crop(input_volume, mins, maxs)
        maskedvolume = applymask(croppedvolume, mask)
    else:
        maskedvolume = applymask(input_volume, mask)

    return maskedvolume, mask
