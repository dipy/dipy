from warnings import warn

import numpy as np

from dipy.reconst.dti import fractional_anisotropy, color_fa

from scipy.ndimage import median_filter
try:
    from skimage.filters import threshold_otsu as otsu
except Exception:
    from dipy.segment.threshold import otsu

from scipy.ndimage import binary_dilation, generate_binary_structure


def multi_median(data, median_radius, numpass):
    """ Applies median filter multiple times on input data.

    Parameters
    ----------
    data : ndarray
        The input volume to apply filter on.
    median_radius : int
        Radius (in voxels) of the applied median filter
    numpass: int
        Number of pass of the median filter

    Returns
    -------
    data : ndarray
        Filtered input volume.
    """
    # Array representing the size of the median window in each dimension.
    medarr = np.ones_like(data.shape) * ((median_radius * 2) + 1)

    if numpass > 1:
        # ensure the input array is not modified
        data = data.copy()

    # Multi pass
    output = np.empty_like(data)
    for i in range(0, numpass):
        median_filter(data, medarr, output=output)
        data, output = output, data
    return data


def applymask(vol, mask):
    """ Mask vol with mask.

    Parameters
    ----------
    vol : ndarray
        Array with $V$ dimensions
    mask : ndarray
        Binary mask.  Has $M$ dimensions where $M <= V$. When $M < V$, we
        append $V - M$ dimensions with axis length 1 to `mask` so that `mask`
        will broadcast against `vol`.  In the typical case `vol` can be 4D,
        `mask` can be 3D, and we append a 1 to the mask shape which (via numpy
        broadcasting) has the effect of applying the 3D mask to each 3D slice in
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
    """Compute the bounding box of nonzero intensity voxels in the volume.

    Parameters
    ----------
    vol : ndarray
        Volume to compute bounding box on.

    Returns
    -------
    npmins : list
        Array containing minimum index of each dimension
    npmaxs : list
        Array containing maximum index of each dimension
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
    """Crops the input volume.

    Parameters
    ----------
    vol : ndarray
        Volume to crop.
    mins : array
        Array containing minimum index of each dimension.
    maxs : array
        Array containing maximum index of each dimension.

    Returns
    -------
    vol : ndarray
        The cropped volume.
    """
    return vol[tuple(slice(i, j) for i, j in zip(mins, maxs))]


def median_otsu(input_volume, vol_idx=None, median_radius=4, numpass=4,
                autocrop=False, dilate=None):
    """Simple brain extraction tool method for images from DWI data.

    It uses a median filter smoothing of the input_volumes `vol_idx` and an
    automatic histogram Otsu thresholding technique, hence the name
    *median_otsu*.

    This function is inspired from Mrtrix's bet which has default values
    ``median_radius=3``, ``numpass=2``. However, from tests on multiple 1.5T
    and 3T data     from GE, Philips, Siemens, the most robust choice is
    ``median_radius=4``, ``numpass=4``.

    Parameters
    ----------
    input_volume : ndarray
        3D or 4D array of the brain volume.
    vol_idx : None or array, optional.
        1D array representing indices of ``axis=3`` of a 4D `input_volume`.
        None is only an acceptable input if ``input_volume`` is 3D.
    median_radius : int
        Radius (in voxels) of the applied median filter (default: 4).
    numpass: int
        Number of pass of the median filter (default: 4).
    autocrop: bool, optional
        if True, the masked input_volume will also be cropped using the
        bounding box defined by the masked data. Should be on if DWI is
        upsampled to 1x1x1 resolution. (default: False).

    dilate : None or int, optional
        number of iterations for binary dilation

    Returns
    -------
    maskedvolume : ndarray
        Masked input_volume
    mask : 3D ndarray
        The binary brain mask

    Notes
    -----
    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    """
    if len(input_volume.shape) == 4:
        if vol_idx is not None:
            b0vol = np.mean(input_volume[..., tuple(vol_idx)], axis=3)
        else:
            raise ValueError("For 4D images, must provide vol_idx input")
    else:
        b0vol = input_volume
    # Make a mask using a multiple pass median filter and histogram
    # thresholding.
    mask = multi_median(b0vol, median_radius, numpass)
    thresh = otsu(mask)
    mask = mask > thresh

    if dilate is not None:
        cross = generate_binary_structure(3, 1)
        mask = binary_dilation(mask, cross, iterations=dilate)

    # Auto crop the volumes using the mask as input_volume for bounding box
    # computing.
    if autocrop:
        mins, maxs = bounding_box(mask)
        mask = crop(mask, mins, maxs)
        croppedvolume = crop(input_volume, mins, maxs)
        maskedvolume = applymask(croppedvolume, mask)
    else:
        maskedvolume = applymask(input_volume, mask)
    return maskedvolume, mask


def segment_from_cfa(tensor_fit, roi, threshold, return_cfa=False):
    """
    Segment the cfa inside roi using the values from threshold as bounds.

    Parameters
    ----------
    tensor_fit : TensorFit object
        TensorFit object

    roi : ndarray
        A binary mask, which contains the bounding box for the segmentation.

    threshold : array-like
        An iterable that defines the min and max values to use for the
        thresholding.
        The values are specified as (R_min, R_max, G_min, G_max, B_min, B_max)

    return_cfa : bool, optional
        If True, the cfa is also returned.

    Returns
    -------
    mask : ndarray
        Binary mask of the segmentation.

    cfa : ndarray, optional
        Array with shape = (..., 3), where ... is the shape of tensor_fit.
        The color fractional anisotropy, ordered as a nd array with the last
        dimension of size 3 for the R, G and B channels.
    """

    FA = fractional_anisotropy(tensor_fit.evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)  # Clamp the FA to remove degenerate tensors

    cfa = color_fa(FA, tensor_fit.evecs)
    roi = np.asarray(roi, dtype=bool)

    include = ((cfa >= threshold[0::2]) &
               (cfa <= threshold[1::2]) &
               roi[..., None])
    mask = np.all(include, axis=-1)

    if return_cfa:
        return mask, cfa

    return mask


def clean_cc_mask(mask):
    """
    Cleans a segmentation of the corpus callosum so no random pixels
    are included.

    Parameters
    ----------
    mask : ndarray
        Binary mask of the coarse segmentation.

    Returns
    -------
    new_cc_mask : ndarray
        Binary mask of the cleaned segmentation.
    """

    from scipy.ndimage import label

    new_cc_mask = np.zeros(mask.shape)

    # Flood fill algorithm to find contiguous regions.
    labels, numL = label(mask)

    volumes = [len(labels[np.where(labels == l_idx+1)])
               for l_idx in np.arange(numL)]
    biggest_vol = np.arange(numL)[np.where(volumes == np.max(volumes))] + 1
    new_cc_mask[np.where(labels == biggest_vol)] = 1

    return new_cc_mask
