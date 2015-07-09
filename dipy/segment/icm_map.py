from __future__ import division, print_function, absolute_import

import numpy as np
from dipy.core.ndindex import ndindex
from dipy.segment.energy_mrf import total_energy
from dipy.denoise.denspeed import add_padding_reflection


def icm(mu, var, masked_img, seg_img, classes, beta):
    r"""Use ICM to segment T1 image with MRF
    Parameters
    -----------

    mu : 1x3 ndarray - mean of each tissue
    var : 1x3 ndarray - variance of each tissue
    masked_img : 3D ndarray - masked T1 structural image
    seg_img : 3D ndarray - initial segmentation provided as an input
    classes : integer - number of tissue classes
    beta : float - the weight of the neighborhood

    Returns
    --------

    segmented : 3D ndarray - segmentation of the input T1 structural image
    totalE : 1x3 ndarrray - total energy per tissue class

    """

    totalE = np.zeros(classes)
    L = range(1, classes + 1)

    segmented = np.zeros(masked_img.shape)
    totalenergy = np.zeros(masked_img.shape)
    shape = masked_img.shape[:3]
    masked_img = masked_img.copy(order='C')
    masked_img_pad = add_padding_reflection(masked_img, 1)
    seg_img = seg_img.copy(order='C')
    seg_img_pad = add_padding_reflection(seg_img, 1)

    for idx in ndindex(shape):
        if not masked_img[idx]:
            continue
        for l in range(0, classes):

            totalE[l] = total_energy(masked_img_pad, seg_img_pad,
                                     mu, var, idx, l, beta)

        segmented[idx] = L[np.argmin(totalE)]
        totalenergy[idx] = np.argmin(totalE)

    return segmented, totalenergy
