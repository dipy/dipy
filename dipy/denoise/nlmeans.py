from __future__ import division, print_function

import numpy as np
from dipy.core.ndindex import ndindex


def get_block(arr, center, radius):

    cx, cy, cz = center

    return arr[cx - radius : cx + radius + 1,
               cy - radius : cy + radius + 1,
               cz - radius : cz + radius + 1]


def nlmeans(image, patch_size=3, block_size=11, sigma=None, rician=True):

    if sigma is None:
        sigma = 5 # Call piesno instead

    out = np.zeros_like(image)

    for idx in ndindex(image.shape):

        neighborhood = get_block(image, idx, block_size)
        x = get_block(neighborhood, idx, patch_size)
        w = np.zeros_like(neighborhood)

        for pos in ndindex(neighborhood.shape):

            y = get_block(neighborhood, pos, patch_size)
            w = np.exp(-np.mean((x - y)**2)/sigma)
            w /= np.sum(w)

        out[idx] = np.sum(w * x**2)

    if rician:
        out -= 2 * sigma**2
        out[out < 0] = 0

    return np.sqrt(out)
