from __future__ import division, print_function

import numpy as np
from dipy.core.ndindex import ndindex


def get_block(arr, center, radius):

    cx, cy, cz = center

    return arr[cx - radius : cx + radius + 1,
               cy - radius : cy + radius + 1,
               cz - radius : cz + radius + 1]


def padding(A, radius):
    """
    Pad A with radius-1 elements on each side in a reflecting manner.
    """

    shape = np.array(A.shape)
    padded = np.zeros(shape + 2*(radius - 1))

    padded[:radius-1, :radius-1, :radius-1] = A[:radius-1:-1, :radius-1:-1, :radius-1:-1]
    padded[-1:radius-shape[0]:-1, -1:radius-shape[1]:-1, -1:radius-shape[2]:-1] = A[-1:radius:-1, -1:radius:-1, -1:radius:-1]

    return padded


def nlmeans(image, patch_size=3, block_size=11, sigma=None, ncoils=1):

    if sigma is None:
        sigma = 5 # Call piesno instead

    # do padding so we can cut exactly the number of required blocks
    print(image.shape)
    image = padding(image, block_size)
    print(image.shape)
    out = np.zeros_like(image)

    for idx in ndindex(image.shape):

        neighborhood = get_block(image, idx, block_size)
        x = get_block(neighborhood, idx, patch_size)
        w = np.zeros_like(neighborhood)

        print(x.shape, w.shape, neighborhood.shape)

        for pos in ndindex(neighborhood.shape):

            y = get_block(neighborhood, pos, patch_size)
            w = np.exp(-np.sqrt(np.mean((x - y)**2))/sigma)
            w /= np.sum(w)

            out[idx] = np.sum(w * x**2)

    if ncoils:
        out -= 2*ncoils * sigma**2
        out[out < 0] = 0

    # unpad out
    out = out[radius:out.shape[0]-radius, radius:out.shape[1]-radius, radius:out.shape[2]-radius]

    return np.sqrt(out)
