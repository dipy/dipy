from __future__ import division, print_function

import numpy as np
from dipy.core.ndindex import ndindex

def nlmeans(input, patch_size=3, block_size=11, sigma=None, rician=True):

    if sigma is None:
        sigma = 5 # Call piesno instead

    def get_block(input, pos, patch_size):
        return input[pos-patchsize]



    def _im2col_3d(A, size, overlap, order):

        assert len(A.shape) == len(size), "number of dimensions mismatch!"

        size = np.array(size)
        overlap = np.array(overlap)
        dim = ((A.shape - size) / (size - overlap)) + 1

        R = np.zeros((np.prod(dim), np.prod(size)), dtype=A.dtype, order=order)
        k = 0

        for a in range(0, A.shape[0] - overlap[0], size[0] - overlap[0]):
            for b in range(0, A.shape[1] - overlap[1], size[1] - overlap[1]):
                for c in range(0, A.shape[2] - overlap[2], size[2] - overlap[2]):
                    R[k, :] = A[a:a + size[0],
                                b:b + size[1],
                                c:c + size[2]].ravel()
                    k += 1

        return R


    def _im2col_4d(A, size, overlap, order):

        assert len(A.shape) == len(size), "number of dimensions mismatch!"

        size = np.array(size)
        overlap = np.array(overlap)
        dim = ((A.shape - size) / (size - overlap)) + 1

        R = np.zeros((np.prod(dim), np.prod(size)), dtype=A.dtype, order=order)
        k = 0

        for a in range(0, A.shape[0]-overlap[0], size[0]-overlap[0]):
            for b in range(0, A.shape[1]-overlap[1], size[1]-overlap[1]):
                for c in range(0, A.shape[2]-overlap[2], size[2]-overlap[2]):
                    for d in range(0, A.shape[3]-overlap[3], size[3]-overlap[3]):
                        R[k, :] = A[a:a + size[0],
                                    b:b + size[1],
                                    c:c + size[2],
                                    d:d + size[3]].ravel()
                        k += 1

        return R


    def im2col_nd(A,  block_shape, overlap, order='F'):
        """
        Returns a 2d array of shape flat(block_shape) by A.shape/block_shape made
        from blocks of a nd array.
        """

        block_shape = np.array(block_shape)
        overlap = np.array(overlap)

        if (overlap.any() < 0) or ((block_shape < overlap).any()):
            raise ValueError('Invalid overlap value, it must lie between 0 \
                             \nand min(block_size)-1', overlap, block_shape)
        print(A.shape, block_shape, overlap)
        A = padding(A, block_shape, overlap)

        if len(block_shape) == 3:
            return _im2col_3d(A, block_shape, overlap, order=order)

        elif len(block_shape) == 4:
            return _im2col_4d(A, block_shape, overlap, order=order)

        raise ValueError("invalid type of window")



    for idx in ndindex(input.shape):
        block = get_block(input, idx, patch_size)