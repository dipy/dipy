from __future__ import division, print_function, absolute_import

# Use ICM to segment T1 image with MRF
import numpy as np
from dipy.core.ndindex import ndindex
from dipy.segment.energy_mrf import total_energy
from dipy.denoise.denspeed import add_padding_reflection


def icm(mu, var, masked_img, seg_img, classes, beta):

    totalE = np.zeros(classes)
    L = range(1, classes + 1)

    segmented = np.zeros(masked_img.shape)
    shape = masked_img.shape[:3]
    masked_img = masked_img.copy(order='C')
    masked_img_pad = add_padding_reflection(masked_img, 1)
    seg_img = masked_img.copy(order='C')
    seg_img_pad = add_padding_reflection(seg_img, 1)

    for idx in ndindex(shape):
        if not masked_img[idx]:
            continue
        for l in range(0, classes):

            totalE[l] = total_energy(masked_img_pad, seg_img_pad,
                                     mu, var, idx, l, beta)

        segmented[idx] = L[np.argmin(totalE)]

    return segmented
