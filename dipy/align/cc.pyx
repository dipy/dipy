import numpy as np
cimport cython
from fused_types cimport floating

cdef enum:
    SI = 0
    SI2 = 1
    SJ = 2
    SJ2 = 3
    SIJ = 4
    CNT = 5


cdef inline int _int_max(int a, int b):
    return a if a >= b else b


cdef inline int _int_min(int a, int b):
    return a if a <= b else b


def precompute_cc_factors_3d(floating[:, :, :] static, floating[:, :, :] moving,
                             int radius):
    cdef int side = 2 * radius + 1
    cdef int ns = static.shape[0]
    cdef int nr = static.shape[1]
    cdef int nc = static.shape[2]
    cdef int s, r, c, k, i, j, t, q, qq, firstc, lastc, firstr, lastr
    cdef double Imean, Jmean
    cdef floating[:, :, :, :] factors = np.ndarray((ns, nr, nc, 5), dtype=np.asarray(static).dtype)
    cdef double[:, :] lines = np.zeros((6, side), dtype=np.float64)
    cdef double[:] sums = np.zeros((6,), dtype=np.float64)
    for r in range(nr):
        firstr = _int_max(0, r - radius)
        lastr = _int_min(nr - 1, r + radius)
        for c in range(nc):
            firstc = _int_max(0, c - radius)
            lastc = _int_min(nc - 1, c + radius)
            # compute factors for line [:,r,c]
            sums[...] = 0
            # Compute all slices and set the sums on the fly
            # compute each slice [k, i={r-radius..r+radius}, j={c-radius,
            # c+radius}]
            for k in range(ns):
                q = k % side
                for t in range(6):
                    sums[t] -= lines[t, q]
                    lines[t, q] = 0
                for i in range(firstr, lastr + 1):
                    for j in range(firstc, lastc + 1):
                        lines[SI, q] += static[k, i, j]
                        lines[SI2, q] += static[k, i, j] * static[k, i, j]
                        lines[SJ, q] += moving[k, i, j]
                        lines[SJ2, q] += moving[k, i, j] * moving[k, i, j]
                        lines[SIJ, q] += static[k, i, j] * moving[k, i, j]
                        lines[CNT, q] += 1
                sums[...] = 0
                for t in range(6):
                    for qq in range(side):
                        sums[t] += lines[t, qq]
                if(k >= radius):
                    # s is the voxel that is affected by the cube with slices
                    # [s-radius..s+radius, :, :]
                    s = k - radius
                    Imean = sums[SI] / sums[CNT]
                    Jmean = sums[SJ] / sums[CNT]
                    factors[s, r, c, 0] = static[s, r, c] - Imean
                    factors[s, r, c, 1] = moving[s, r, c] - Jmean
                    factors[s, r, c, 2] = sums[SIJ] - Jmean * sums[SI] - \
                        Imean * sums[SJ] + sums[CNT] * Jmean * Imean
                    factors[s, r, c, 3] = sums[SI2] - Imean * sums[SI] - \
                        Imean * sums[SI] + sums[CNT] * Imean * Imean
                    factors[s, r, c, 4] = sums[SJ2] - Jmean * sums[SJ] - \
                        Jmean * sums[SJ] + sums[CNT] * Jmean * Jmean
            # Finally set the values at the end of the line
            for s in range(ns - radius, ns):
                # this would be the last slice to be processed for voxel
                # [s,r,c], if it existed
                k = s + radius
                q = k % side
                for t in range(6):
                    sums[t] -= lines[t, q]
                Imean = sums[SI] / sums[CNT]
                Jmean = sums[SJ] / sums[CNT]
                factors[s, r, c, 0] = static[s, r, c] - Imean
                factors[s, r, c, 1] = moving[s, r, c] - Jmean
                factors[s, r, c, 2] = sums[SIJ] - Jmean * sums[SI] - \
                    Imean * sums[SJ] + sums[CNT] * Jmean * Imean
                factors[s, r, c, 3] = sums[SI2] - Imean * sums[SI] - \
                    Imean * sums[SI] + sums[CNT] * Imean * Imean
                factors[s, r, c, 4] = sums[SJ2] - Jmean * sums[SJ] - \
                    Jmean * sums[SJ] + sums[CNT] * Jmean * Jmean
    return factors


def compute_cc_forward_step_3d(floating[:, :, :, :] grad_static,
                               floating[:, :, :, :] gradMoving,
                               floating[:, :, :, :] factors):
    cdef int ns = grad_static.shape[0]
    cdef int nr = grad_static.shape[1]
    cdef int nc = grad_static.shape[2]
    cdef double energy = 0
    cdef double Ii, Ji, sfm, sff, smm, localCorrelation, temp
    cdef floating[:, :, :, :] out = np.zeros((ns, nr, nc, 3), dtype=np.asarray(grad_static).dtype)
    for s in range(ns):
        for r in range(nr):
            for c in range(nc):
                Ii = factors[s, r, c, 0]
                Ji = factors[s, r, c, 1]
                sfm = factors[s, r, c, 2]
                sff = factors[s, r, c, 3]
                smm = factors[s, r, c, 4]
                if(sff == 0.0 or smm == 0.0):
                    continue
                localCorrelation = 0
                if(sff * smm > 1e-5):
                    localCorrelation = sfm * sfm / (sff * smm)
                if(localCorrelation < 1):  # avoid bad values...
                    energy -= localCorrelation
                temp = 2.0 * sfm / (sff * smm) * (Ji - sfm / sff * Ii)
                for qq in range(3):
                    out[s, r, c, qq] -= temp * grad_static[s, r, c, qq]
    return out, energy


def compute_cc_backward_step_3d(floating[:, :, :, :] grad_static,
                                floating[:, :, :, :] gradMoving,
                                floating[:, :, :, :] factors):
    cdef int ns = grad_static.shape[0]
    cdef int nr = grad_static.shape[1]
    cdef int nc = grad_static.shape[2]
    cdef double energy = 0
    cdef double Ii, Ji, sfm, sff, smm, localCorrelation, temp
    cdef floating[:, :, :, :] out = np.zeros((ns, nr, nc, 3), dtype=np.asarray(grad_static).dtype)
    for s in range(ns):
        for r in range(nr):
            for c in range(nc):
                Ii = factors[s, r, c, 0]
                Ji = factors[s, r, c, 1]
                sfm = factors[s, r, c, 2]
                sff = factors[s, r, c, 3]
                smm = factors[s, r, c, 4]
                if(sff == 0.0 or smm == 0.0):
                    continue
                localCorrelation = 0
                if(sff * smm > 1e-5):
                    localCorrelation = sfm * sfm / (sff * smm)
                if(localCorrelation < 1):  # avoid bad values...
                    energy -= localCorrelation
                temp = 2.0 * sfm / (sff * smm) * (Ii - sfm / smm * Ji)
                for qq in range(3):
                    out[s, r, c, qq] -= temp * gradMoving[s, r, c, qq]
    return out, energy
