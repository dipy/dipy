from __future__ import division

import numpy as np
cimport numpy as cnp
cimport cython

from libc.math cimport sqrt, exp
from libc.stdlib cimport malloc, free


def nlmeans_3d(arr, patch_radius=1, block_radius=5, sigma=None, rician=True):

    arr = np.ascontiguousarray(arr, dtype='f8')

    arr = add_border(arr, block_radius)

    arrnlm = _nlmeans_3d(arr, patch_radius, block_radius, sigma, rician)

    return remove_border(arrnlm, block_radius)


@cython.wraparound(False)
@cython.boundscheck(False)
def _nlmeans_3d(double [:, :, ::1] arr, patch_radius=1, block_radius=5,
                sigma=None, rician=True):

    cdef:
        cnp.npy_intp i, j, k, I, J, K
        double [:, :, ::1] out = np.zeros_like(arr)
        #double [::1] W = np.zeros((2 * block_radius + 1) ** 3)
        double summ = 0
        double sigm = 0
        cnp.npy_intp P = patch_radius
        cnp.npy_intp B = block_radius

    if sigma is None:
        sigm = 5 # call piesno
    else:
        sigm = sigma

    I = arr.shape[0]
    J = arr.shape[1]
    K = arr.shape[2]

    #move the block
    with nogil:
        for i in range(B, I - B + 1):
            for j in range(B , J - B + 1):
                for k in range(B, K - B + 1):

                    out[i, j, k] = process_block(arr, i, j, k, B, P, sigm)

    new = np.asarray(out)

    if rician:
        new -= 2 * sigm ** 2
        new[new < 0] = 0

    return np.sqrt(new)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double process_block(double [:, :, ::1] arr,
                          cnp.npy_intp i, cnp.npy_intp j, cnp.npy_intp k,
                          cnp.npy_intp B, cnp.npy_intp P, double sigma) nogil:

    cdef:
        cnp.npy_intp m, n, o, M, N, O, a, b, c, cnt, step
        double patch_vol_size
        double summ, d, w, sumw, sum_out, x
        double * W
        double denom
        cnp.npy_intp BS = B * 2 + 1

    cnt = 0
    sumw = 0
    patch_vol_size = (P + P + 1) * (P + P + 1) * (P + P + 1)
    denom = sigma * sigma

    W = <double *> malloc(BS * BS * BS * sizeof(double))

    # calculate weights between the central patch and the moving patch in block
    # (m, n, o) coordinates are the center of the moving patch
    # (i, j, k) coordinates are the center of the static patch
    # (a, b, c) run incide both patches
    for m in range(i - B + P, i + B - P):
        for n in range(j - B + P, j + B - P):
            for o in range(k - B + P, k + B - P):

                summ = 0

                # calculate square distance
                for a in range(- P, P + 1):
                    for b in range(- P, P + 1):
                        for c in range(- P, P + 1):

                            # this line takes most of the time! mem access
                            d = arr[i + a, j + b, k + c] - arr[m + a, n + b, o + c]
                            summ += d * d

                w = exp(-(summ / patch_vol_size) / denom)
                sumw += w
                W[cnt] = w
                cnt += 1

    cnt = 0

    sum_out = 0

    # calculate normalized weights and sums of the weights with the positions
    # of the patches

    for m in range(i - B + P, i + B - P):
        for n in range(j - B + P, j + B - P):
            for o in range(k - B + P, k + B - P):

                if sumw > 0:
                    w = W[cnt] / sumw
                else:
                    w = 0

                x = arr[m, n, o]

                sum_out += w * x * x

                cnt += 1

    free(W)

    return sum_out


def add_border(arr, padding):
    arr = np.pad(arr, (padding, padding,), mode='reflect')
    return arr


def remove_border(arr, padding):
    shape = arr.shape
    return arr[padding:shape[0] - padding,
               padding:shape[1] - padding,
               padding:shape[2] - padding]
