from __future__ import division, print_function

import numpy as np
from dipy.core.ndindex import ndindex


def get_block(arr, center, radius):

    cx, cy, cz = center

   # print(cx - radius - 1, cx + radius, center, radius)

    return arr[cx - radius - 1: cx + radius,
               cy - radius - 1: cy + radius,
               cz - radius - 1: cz + radius]


def padding(A, radius):
    """
    Pad A with radius-1 elements on each side in a reflecting manner.
    """

    shape = np.array(A.shape)
    padded = np.zeros(shape + 2*(radius - 1))

    padded[:radius-1, :radius-1, :radius-1] = A[:radius-1, :radius-1, :radius-1:][::-1]
    padded[-1:-radius:-1, -1:-radius:-1, -1:-radius:-1] = A[-1:-radius:-1, -1:-radius:-1, -1:-radius:-1][::-1]

    padded[:radius-1, :radius-1, :radius-1] = A[:radius-1, :radius-1, :radius-1:][::-1]
    padded[-1:-radius:-1, -1:-radius:-1, -1:-radius:-1] = A[-1:-radius:-1, -1:-radius:-1, -1:-radius:-1][::-1]

    padded[:radius-1, :radius-1, :radius-1] = A[:radius-1, :radius-1, :radius-1:][::-1]
    padded[-1:-radius:-1, -1:-radius:-1, -1:-radius:-1] = A[-1:-radius:-1, -1:-radius:-1, -1:-radius:-1][::-1]

    padded[:radius-1, :radius-1, :radius-1] = A[:radius-1, :radius-1, :radius-1:][::-1]
    padded[-1:-radius:-1, -1:-radius:-1, -1:-radius:-1] = A[-1:-radius:-1, -1:-radius:-1, -1:-radius:-1][::-1]

    return padded


def nlmeans(image, patch_size=3, block_size=11, sigma=None, ncoils=1):

    if sigma is None:
        sigma = 5 # Call piesno instead

    # do padding so we can cut exactly the number of required blocks
    #print(image.shape)
    #image = padding(image, block_size)
    #print(image.shape)

    shape = image.shape
    padding = (block_size-1, block_size-1) * image.ndim
    image = np.pad(image, zip(padding[::2], padding[1::2]), mode='reflect')
    out = np.zeros_like(image)

    #print("le print")
    #print(out.shape)
    #print("fin")

    for idx in ndindex(image.shape):

        idx = np.array(idx) + block_size - 1
        neighborhood = get_block(image, idx, (block_size-1)/2)
        print(image.shape,idx,block_size)
        x = get_block(neighborhood, idx, (patch_size-1)/2)
        w = np.zeros_like(neighborhood)

        #print("2e print")
        print(x.shape, w.shape, neighborhood.shape)

        for pos in ndindex(np.array(neighborhood.shape) - patch_size):

            pos = np.array(pos) + patch_size - 1
            y = get_block(neighborhood, pos, (patch_size-1)/2)
            #print(x.shape,y.shape,pos,idx)
            w = np.exp(-np.sqrt(np.mean((x - y)**2))/sigma)
            w /= np.sum(w)

            out[idx] = np.sum(w * x**2)

    if ncoils:
        out -= 2*ncoils * sigma**2
        out[out < 0] = 0

    # unpad out
    out = out[radius:shape[0]-radius, radius:shape[1]-radius, radius:shape[2]-radius]

    return np.sqrt(out)
