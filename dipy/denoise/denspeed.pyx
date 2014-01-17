from __future__ import division

import numpy as np
cimport numpy as cnp
cimport cython

from libc.math cimport sqrt, exp


def nlmeans_3d(arr, patch_size=3, block_size=11, sigma=None, rician=True):

    arr = np.ascontiguousarray(arr)

    if arr.sum() == 0:
        raise ValueError('Wrong parameter, arr is 0 everywhere')

    arr = add_border(arr, block_size)

    print(arr.flags)
    print(arr.shape)

    arrnlm = _nlmeans_3d(arr, patch_size, block_size, sigma, rician)

    return remove_border(arrnlm, block_size)


@cython.wraparound(False)
@cython.boundscheck(False)
def _nlmeans_3d(double [:, :, ::1] arr, patch_size=3, block_size=11, sigma=None, rician=True):

    cdef:
        cnp.npy_intp i, j, k, I, J, K
        double [:, :, ::1] out = np.zeros_like(arr)
        double [::1] W = np.zeros(block_size ** 3)
        double summ = 0
        double sigm = 0
        cnp.npy_intp P = (patch_size - 1) / 2
        cnp.npy_intp B = (block_size - 1) / 2
        cnp.npy_intp BS = block_size

    if sigma is None:
        sigm = 5 # call piesno
    else:
        sigm = sigma

    I = arr.shape[0]
    J = arr.shape[1]
    K = arr.shape[2]

    #moving the block
    with nogil:
        for i in range(BS - 1, I - BS + 1):
            for j in range(BS - 1, J - BS + 1):
                for k in range(BS - 1, K - BS + 1):

                    out[i, j, k] = process_block(arr, W, i, j, k, B, P, sigm)

    new = np.asarray(out)

    if rician:
        new -= 2 * sigm ** 2
        new[new < 0] = 0

    return np.sqrt(new)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double process_block(double [:, :, ::1] arr, double [::1] W,
                          cnp.npy_intp i, cnp.npy_intp j, cnp.npy_intp k,
                          cnp.npy_intp B, cnp.npy_intp P, double sigma) nogil:

    cdef:
        cnp.npy_intp m, n, o, M, N, O, a, b, c, cnt, step
        double patch_vol_size
        double summ, x, d, w, sumw, sum_out

    cnt = 0
    patch_vol_size = (P + P + 1) * (P + P + 1) * (P + P + 1)
    sumw = 0
    step = P + P + 1

    # calculate weights between the central patch and the moving patch in block
    # moving the patch
    for m from i - B <= m < i + B + 1 by step:
        for n from j - B <= n < j + B + 1 by step:
            for o from k - B <= o < k + B + 1 by step:

                summ = 0

                # calculate square distance
                for a in range(-P, P + 1):
                    for b in range(-P, P + 1):
                        for c in range(-P, P + 1):

                            x = arr[i + a, j + b, k + c]
                            d = x - arr[m + a, n + b, o + c]
                            d = d * d

                            summ += d

                w = exp(-(sqrt(summ / patch_vol_size)) / sigma)
                sumw += w
                W[cnt] = w
                cnt += 1

    cnt = 0

    sum_out = 0

    # calculate normalized weights and sums of the weights with the positions
    # of the patches
    for m from i - B <= m < i + B + 1 by step:
        for n from j - B <= n < j + B + 1 by step:
            for o from k - B <= o < k + B + 1 by step:

                if sumw > 0:
                    w = W[cnt] / sumw
                else:
                    w = 0

                x = arr[m, n, o]

                sum_out += w * x * x

                cnt += 1

    return sum_out


@cython.wraparound(False)
@cython.boundscheck(False)
def add_border(arr, block_size=11):

    padding = (block_size - 1) / 2
    #padding = (block_size-1) * arr.ndim-1

    #print(padding, arr.ndim)#, zip(padding[::2], padding[1::2]))
    #arr = np.pad(arr, zip(padding[::2], padding[1::2]), mode='reflect')
    print(arr.shape)
    arr = np.pad(arr, (padding, padding,), mode='reflect')
    print(arr.shape)
    return arr


@cython.wraparound(False)
@cython.boundscheck(False)
def remove_border(arr, block_size=11):

    shape = arr.shape
    padding = (block_size - 1) / 2
    return arr[padding:shape[0] - padding,
               padding:shape[1] - padding,
               padding:shape[2] - padding]
