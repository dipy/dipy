from __future__ import division, print_function, absolute_import

from warnings import warn

import numpy as np

from scipy.ndimage.filters import median_filter
try:
    from skimage.filter import threshold_otsu as otsu
except:
    from .threshold import otsu


def multi_median(input, median_radius, numpass):
    """
    Applies multiple times scikit-image's median filter on input data.

    Parameters
    ----------
    input : ndarray
        The input volume to apply filter on.
    median_radius : int
        Radius (in voxels) of the applied median filter 
    numpass : int
        Number of pass of the median filter

    Returns
    -------
    input : ndarray
        Filtered input volume.
    """
    outvol = np.zeros_like(input)
    # Array representing the size of the median window in each dimension.
    medarr = np.ones_like(input.shape) * ((median_radius * 2) + 1)
    # Multi pass
    for i in range(0, numpass):
        median_filter(input, medarr, output=input)
    return input


def applymask(vol, mask):
    """ Mask vol with mask.

    Parameters
    ----------
    vol : ndarray
        Array with $V$ dimensions
    mask : ndarray
        Binary mask.  Has $M$ dimensions where $M <= V$. When $M < V$, we append
        $V - M$ dimensions with axis length 1 to `mask` so that `mask` will
        broadcast against `vol`.  In the typical case `vol` can be 4D, `mask`
        can be 3D, and we append a 1 to the mask shape which (via numpy
        broadcasting) has the effect of appling the 3D mask to each 3D slice in
        `vol` (``vol[..., 0]`` to ``vol[..., -1``).

    Returns
    -------
    masked_vol : ndarray
        `vol` multiplied by `mask` where `mask` may have been extended to match
        extra dimensions in `vol`
    """
    mask = mask.reshape(mask.shape + (vol.ndim - mask.ndim) * (1,))
    return vol * mask


def bounding_box(vol):
    """ Compute the bounding box of nonzero intensity voxels in the volume.

    Parameters
    ----------
    vol : ndarray
        Volume to compute bounding box on.

    Returns
    -------
    npmins : list
        Array containg minimum index of each dimension
    npmaxs : list
        Array containg maximum index of each dimension
    """
    # Find bounds on first dimension
    temp = vol
    for i in range(vol.ndim - 1):
        temp = temp.any(-1)
    mins = [temp.argmax()]
    maxs = [len(temp) - temp[::-1].argmax()]
    # Check that vol is not all 0
    if mins[0] == 0 and temp[0] == 0:
        warn('No data found in volume to bound. Returning empty bounding box.')
        return [0] * vol.ndim, [0] * vol.ndim
    # Find bounds on remaining dimensions
    if vol.ndim > 1:
        a, b = bounding_box(vol.any(0))
        mins.extend(a)
        maxs.extend(b)
    return mins, maxs


def crop(vol, mins, maxs):
    """ Crops the input volume.

    Parameters
    ----------
    vol : ndarray
        Volume to crop.
    mins : array
        Array containg minimum index of each dimension.
    maxs : array
        Array containg maximum index of each dimension.

    Returns
    -------
    vol : ndarray
        The cropped volume.
    """
    return vol[tuple(slice(i, j) for i, j in zip(mins, maxs))]


def median_otsu(input_volume, median_radius=4, numpass=4,
                autocrop=False, b0Slices=None):
    """ Simple brain extraction tool method for images from DWI data

    It uses a median filter smoothing of the input_volumes `b0Slices` and an
    automatic histogram Otsu thresholding technique, hence the name
    *median_otsu*.

    It mimics the ``MRtrix`` bet from the documentation::

        mrconvert dwi.nii -coord 3 0 - | threshold - - | median3D - - | \
                median3D - mask.nii

    ``MRtrix`` uses default ``mean_radius=3`` and ``numpass=2``

    However, from tests on multiple 1.5T and 3T data from GE, Philips, Siemens,
    the most robust choice is ``median_radius=4``, ``numpass=4``

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
    b0Slices : None or array, optional
        1D array representing indices of ``axis=3`` of a 4D `input_volume` where
        the acquisition b value == 0. None (the default) corresponds to ``(0,)``
        (assumes first volume in 4D array is b == 0)

    Returns
    -------
    maskedvolume : ndarray
        Masked input_volume
    mask : 3D ndarray
        The binary brain mask
    """
    if len(input_volume.shape) == 4:
        if b0Slices is not None:
            b0vol = np.mean(input_volume[..., tuple(b0Slices)], axis=3)
        else:
            b0vol = input_volume[..., 0].copy()
    else:
        b0vol = input_volume.copy()
    # Make a mask using a multiple pass median filter and histogram thresholding.
    mask = multi_median(b0vol, median_radius, numpass)
    thresh = otsu(mask)
    mask = mask > thresh
    # Auto crop the volumes using the mask as input_volume for bounding box computing.
    if autocrop:
        mins, maxs = bounding_box(mask)
        mask = crop(mask, mins, maxs)
        croppedvolume = crop(input_volume, mins, maxs)
        maskedvolume = applymask(croppedvolume, mask)
    else:
        maskedvolume = applymask(input_volume, mask)
    return maskedvolume, mask
