""" Utility functions used by the Cross Correlation (CC) metric """

import numpy as np
from dipy.align.fused_types cimport floating
cimport cython
cimport numpy as cnp


cdef inline int _int_max(int a, int b) noexcept nogil:
    r"""
    Returns the maximum of a and b
    """
    return a if a >= b else b


cdef inline int _int_min(int a, int b) noexcept nogil:
    r"""
    Returns the minimum of a and b
    """
    return a if a <= b else b


cdef enum:
    SI = 0
    SI2 = 1
    SJ = 2
    SJ2 = 3
    SIJ = 4
    CNT = 5


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int _wrap(int x, int m) noexcept nogil:
    r""" Auxiliary function to `wrap` an array around its low-end side.
    Negative indices are mapped to last coordinates so that no extra memory
    is required to account for local rectangular windows that exceed the
    array's low-end boundary.

    Parameters
    ----------
    x : int
        the array position to be wrapped
    m : int
        array length
    """
    if x < 0:
        return x + m
    return x


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void _update_factors(double[:, :, :, :] factors,
                                 floating[:, :, :] moving,
                                 floating[:, :, :] static,
                                 cnp.npy_intp ss, cnp.npy_intp rr, cnp.npy_intp cc,
                                 cnp.npy_intp s, cnp.npy_intp r, cnp.npy_intp c, int operation)noexcept nogil:
    r"""Updates the precomputed CC factors of a rectangular window

    Updates the precomputed CC factors of the rectangular window centered
    at (`ss`, `rr`, `cc`) by adding the factors corresponding to voxel
    (`s`, `r`, `c`) of input images `moving` and `static`.

    Parameters
    ----------
    factors : array, shape (S, R, C, 5)
        array containing the current precomputed factors to be updated
    moving : array, shape (S, R, C)
        the moving volume (notice that both images must already be in a common
        reference domain, in particular, they must have the same shape)
    static : array, shape (S, R, C)
        the static volume, which also defines the reference registration domain
    ss : int
        first coordinate of the rectangular window to be updated
    rr : int
        second coordinate of the rectangular window to be updated
    cc : int
        third coordinate of the rectangular window to be updated
    s: int
        first coordinate of the voxel the local window should be updated with
    r: int
        second coordinate of the voxel the local window should be updated with
    c: int
        third coordinate of the voxel the local window should be updated with
    operation : int, either -1, 0 or 1
        indicates whether the factors of voxel (`s`, `r`, `c`) should be
        added to (`operation`=1), subtracted from (`operation`=-1), or set as
        (`operation`=0) the current factors for the rectangular window centered
        at (`ss`, `rr`, `cc`).

    """
    cdef:
        double sval
        double mval
    if s >= moving.shape[0] or r >= moving.shape[1] or c >= moving.shape[2]:
        if operation == 0:
            factors[ss, rr, cc, SI] = 0
            factors[ss, rr, cc, SI2] = 0
            factors[ss, rr, cc, SJ] = 0
            factors[ss, rr, cc, SJ2] = 0
            factors[ss, rr, cc, SIJ] = 0
    else:
        sval = static[s, r, c]
        mval = moving[s, r, c]
        if operation == 0:
            factors[ss, rr, cc, SI] = sval
            factors[ss, rr, cc, SI2] = sval*sval
            factors[ss, rr, cc, SJ] = mval
            factors[ss, rr, cc, SJ2] = mval*mval
            factors[ss, rr, cc, SIJ] = sval*mval
        elif operation == -1:
            factors[ss, rr, cc, SI] -= sval
            factors[ss, rr, cc, SI2] -= sval*sval
            factors[ss, rr, cc, SJ] -= mval
            factors[ss, rr, cc, SJ2] -= mval*mval
            factors[ss, rr, cc, SIJ] -= sval*mval
        elif operation == 1:
            factors[ss, rr, cc, SI] += sval
            factors[ss, rr, cc, SI2] += sval*sval
            factors[ss, rr, cc, SJ] += mval
            factors[ss, rr, cc, SJ2] += mval*mval
            factors[ss, rr, cc, SIJ] += sval*mval


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def precompute_cc_factors_3d(floating[:, :, :] static,
                             floating[:, :, :] moving,
                             cnp.npy_intp radius, num_threads=None):
    """Precomputations to quickly compute the gradient of the CC Metric.

    Pre-computes the separate terms of the cross correlation metric and image
    norms at each voxel considering a neighborhood of the given radius to
    efficiently compute the gradient of the metric with respect to the
    deformation field :footcite:p:`Ocegueda2016`, :footcite:p:`Avants2008`,
    :footcite:p:`Avants2009`.

    Parameters
    ----------
    static : array, shape (S, R, C)
        the static volume, which also defines the reference registration domain
    moving : array, shape (S, R, C)
        the moving volume (notice that both images must already be in a common
        reference domain, i.e. the same S, R, C)
    radius : the radius of the neighborhood (cube of (2 * radius + 1)^3 voxels)

    Returns
    -------
    factors : array, shape (S, R, C, 5)
        the precomputed cross correlation terms::

            - factors[:,:,:,0] : static minus its mean value along the neighborhood
            - factors[:,:,:,1] : moving minus its mean value along the neighborhood
            - factors[:,:,:,2] : sum of the pointwise products of static and moving
              along the neighborhood
            - factors[:,:,:,3] : sum of sq. values of static along the neighborhood
            - factors[:,:,:,4] : sum of sq. values of moving along the neighborhood

    References
    ----------
    .. footbibliography::
    """
    cdef:
        cnp.npy_intp ns = static.shape[0]
        cnp.npy_intp nr = static.shape[1]
        cnp.npy_intp nc = static.shape[2]
        cnp.npy_intp side = 2 * radius + 1
        cnp.npy_intp firstc, lastc, firstr, lastr, firsts, lasts
        cnp.npy_intp s, r, c, it, sides, sider, sidec
        double cnt
        cnp.npy_intp ssss, sss, ss, rr, cc, prev_ss, prev_rr, prev_cc
        double Imean, Jmean, IJprods, Isq, Jsq
        double[:, :, :, :] temp = np.zeros((2, nr, nc, 5), dtype=np.float64)
        floating[:, :, :, :] factors = np.zeros((ns, nr, nc, 5),
                                                dtype=np.asarray(static).dtype)

    with nogil:
        sss = 1
        for s in range(ns+radius):
            ss = _wrap(s - radius, ns)
            sss = 1 - sss
            firsts = _int_max(0, ss - radius)
            lasts = _int_min(ns - 1, ss + radius)
            sides = (lasts - firsts + 1)
            for r in range(nr+radius):
                rr = _wrap(r - radius, nr)
                firstr = _int_max(0, rr - radius)
                lastr = _int_min(nr - 1, rr + radius)
                sider = (lastr - firstr + 1)
                for c in range(nc+radius):
                    cc = _wrap(c - radius, nc)
                    # New corner
                    _update_factors(temp, moving, static,
                                    sss, rr, cc, s, r, c, 0)

                    # Add signed sub-volumes
                    if s > 0:
                        prev_ss = 1 - sss
                        for it in range(5):
                            temp[sss, rr, cc, it] += temp[prev_ss, rr, cc, it]
                        if r > 0:
                            prev_rr = _wrap(rr-1, nr)
                            for it in range(5):
                                temp[sss, rr, cc, it] -= \
                                    temp[prev_ss, prev_rr, cc, it]
                            if c > 0:
                                prev_cc = _wrap(cc-1, nc)
                                for it in range(5):
                                    temp[sss, rr, cc, it] += \
                                        temp[prev_ss, prev_rr, prev_cc, it]
                        if c > 0:
                            prev_cc = _wrap(cc-1, nc)
                            for it in range(5):
                                temp[sss, rr, cc, it] -= \
                                    temp[prev_ss, rr, prev_cc, it]
                    if r > 0:
                        prev_rr = _wrap(rr-1, nr)
                        for it in range(5):
                            temp[sss, rr, cc, it] += \
                                temp[sss, prev_rr, cc, it]
                        if c > 0:
                            prev_cc = _wrap(cc-1, nc)
                            for it in range(5):
                                temp[sss, rr, cc, it] -= \
                                    temp[sss, prev_rr, prev_cc, it]
                    if c > 0:
                        prev_cc = _wrap(cc-1, nc)
                        for it in range(5):
                            temp[sss, rr, cc, it] += temp[sss, rr, prev_cc, it]

                    # Add signed corners
                    if s >= side:
                        _update_factors(temp, moving, static,
                                        sss, rr, cc, s-side, r, c, -1)
                        if r >= side:
                            _update_factors(temp, moving, static,
                                            sss, rr, cc, s-side, r-side, c, 1)
                            if c >= side:
                                _update_factors(temp, moving, static, sss, rr,
                                                cc, s-side, r-side, c-side, -1)
                        if c >= side:
                            _update_factors(temp, moving, static,
                                            sss, rr, cc, s-side, r, c-side, 1)
                    if r >= side:
                        _update_factors(temp, moving, static,
                                        sss, rr, cc, s, r-side, c, -1)
                        if c >= side:
                            _update_factors(temp, moving, static,
                                            sss, rr, cc, s, r-side, c-side, 1)

                    if c >= side:
                        _update_factors(temp, moving, static,
                                        sss, rr, cc, s, r, c-side, -1)
                    # Compute final factors
                    if s >= radius and r >= radius and c >= radius:
                        firstc = _int_max(0, cc - radius)
                        lastc = _int_min(nc - 1, cc + radius)
                        sidec = (lastc - firstc + 1)
                        cnt = sides*sider*sidec
                        Imean = temp[sss, rr, cc, SI] / cnt
                        Jmean = temp[sss, rr, cc, SJ] / cnt
                        IJprods = (temp[sss, rr, cc, SIJ] -
                                   Jmean * temp[sss, rr, cc, SI] -
                                   Imean * temp[sss, rr, cc, SJ] +
                                   cnt * Jmean * Imean)
                        Isq = (temp[sss, rr, cc, SI2] -
                               Imean * temp[sss, rr, cc, SI] -
                               Imean * temp[sss, rr, cc, SI] +
                               cnt * Imean * Imean)
                        Jsq = (temp[sss, rr, cc, SJ2] -
                               Jmean * temp[sss, rr, cc, SJ] -
                               Jmean * temp[sss, rr, cc, SJ] +
                               cnt * Jmean * Jmean)
                        factors[ss, rr, cc, 0] = static[ss, rr, cc] - Imean
                        factors[ss, rr, cc, 1] = moving[ss, rr, cc] - Jmean
                        factors[ss, rr, cc, 2] = IJprods
                        factors[ss, rr, cc, 3] = Isq
                        factors[ss, rr, cc, 4] = Jsq
    return factors


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def precompute_cc_factors_3d_test(floating[:, :, :] static,
                                  floating[:, :, :] moving, int radius):
    """Precomputations to quickly compute the gradient of the CC Metric.

    This version of precompute_cc_factors_3d is for testing purposes, it
    directly computes the local cross-correlation factors without any
    optimization, so it is less error-prone than the accelerated version.
    """
    cdef:
        cnp.npy_intp ns = static.shape[0]
        cnp.npy_intp nr = static.shape[1]
        cnp.npy_intp nc = static.shape[2]
        cnp.npy_intp s, r, c, k, i, j, t
        cnp.npy_intp firstc, lastc, firstr, lastr, firsts, lasts
        double Imean, Jmean
        floating[:, :, :, :] factors = np.zeros((ns, nr, nc, 5),
                                                dtype=np.asarray(static).dtype)
        double[:] sums = np.zeros((6,), dtype=np.float64)

    with nogil:
        for s in range(ns):
            firsts = _int_max(0, s - radius)
            lasts = _int_min(ns - 1, s + radius)
            for r in range(nr):
                firstr = _int_max(0, r - radius)
                lastr = _int_min(nr - 1, r + radius)
                for c in range(nc):
                    firstc = _int_max(0, c - radius)
                    lastc = _int_min(nc - 1, c + radius)
                    for t in range(6):
                        sums[t] = 0
                    for k in range(firsts, 1 + lasts):
                        for i in range(firstr, 1 + lastr):
                            for j in range(firstc, 1 + lastc):
                                sums[SI] += static[k, i, j]
                                sums[SI2] += static[k, i, j]**2
                                sums[SJ] += moving[k, i, j]
                                sums[SJ2] += moving[k, i, j]**2
                                sums[SIJ] += static[k, i, j]*moving[k, i, j]
                                sums[CNT] += 1
                    Imean = sums[SI] / sums[CNT]
                    Jmean = sums[SJ] / sums[CNT]
                    factors[s, r, c, 0] = static[s, r, c] - Imean
                    factors[s, r, c, 1] = moving[s, r, c] - Jmean
                    factors[s, r, c, 2] = (sums[SIJ] - Jmean * sums[SI] -
                                           Imean * sums[SJ] +
                                           sums[CNT] * Jmean * Imean)
                    factors[s, r, c, 3] = (sums[SI2] - Imean * sums[SI] -
                                           Imean * sums[SI] +
                                           sums[CNT] * Imean * Imean)
                    factors[s, r, c, 4] = (sums[SJ2] - Jmean * sums[SJ] -
                                           Jmean * sums[SJ] +
                                           sums[CNT] * Jmean * Jmean)
    return np.asarray(factors)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_cc_forward_step_3d(floating[:, :, :, :] grad_static,
                               floating[:, :, :, :] factors,
                               cnp.npy_intp radius):
    """Gradient of the CC Metric w.r.t. the forward transformation.

    Computes the gradient of the Cross Correlation metric for symmetric
    registration (SyN) :footcite:p:`Avants2008` w.r.t. the displacement
    associated to the moving volume ('forward' step) as in
    :footcite:t:`Avants2009`.

    Parameters
    ----------
    grad_static : array, shape (S, R, C, 3)
        the gradient of the static volume
    factors : array, shape (S, R, C, 5)
        the precomputed cross correlation terms obtained via
        precompute_cc_factors_3d
    radius : int
        the radius of the neighborhood used for the CC metric when
        computing the factors. The returned vector field will be
        zero along a boundary of width radius voxels.

    Returns
    -------
    out : array, shape (S, R, C, 3)
        the gradient of the cross correlation metric with respect to the
        displacement associated to the moving volume
    energy : the cross correlation energy (data term) at this iteration

    References
    ----------
    .. footbibliography::
    """
    cdef:
        cnp.npy_intp ns = grad_static.shape[0]
        cnp.npy_intp nr = grad_static.shape[1]
        cnp.npy_intp nc = grad_static.shape[2]
        double energy = 0
        cnp.npy_intp s, r, c
        double Ii, Ji, sfm, sff, smm, localCorrelation, temp
        floating[:, :, :, :] out =\
            np.zeros((ns, nr, nc, 3), dtype=np.asarray(grad_static).dtype)
    with nogil:
        for s in range(radius, ns-radius):
            for r in range(radius, nr-radius):
                for c in range(radius, nc-radius):
                    Ii = factors[s, r, c, 0]
                    Ji = factors[s, r, c, 1]
                    sfm = factors[s, r, c, 2]
                    sff = factors[s, r, c, 3]
                    smm = factors[s, r, c, 4]
                    if sff == 0.0 or smm == 0.0:
                        continue
                    localCorrelation = 0
                    if sff * smm > 1e-5:
                        localCorrelation = sfm * sfm / (sff * smm)
                    if localCorrelation < 1:  # avoid bad values...
                        energy -= localCorrelation
                    temp = 2.0 * sfm / (sff * smm) * (Ji - sfm / sff * Ii)
                    out[s, r, c, 0] -= temp * grad_static[s, r, c, 0]
                    out[s, r, c, 1] -= temp * grad_static[s, r, c, 1]
                    out[s, r, c, 2] -= temp * grad_static[s, r, c, 2]
    return np.asarray(out), energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_cc_backward_step_3d(floating[:, :, :, :] grad_moving,
                                floating[:, :, :, :] factors,
                                cnp.npy_intp radius):
    """Gradient of the CC Metric w.r.t. the backward transformation.

    Computes the gradient of the Cross Correlation metric for symmetric
    registration (SyN) :footcite:p:`Avants2008`. w.r.t. the displacement
    associated to the static volume ('backward' step) as in
    :footcite:t:`Avants2009`.

    Parameters
    ----------
    grad_moving : array, shape (S, R, C, 3)
        the gradient of the moving volume
    factors : array, shape (S, R, C, 5)
        the precomputed cross correlation terms obtained via
        precompute_cc_factors_3d
    radius : int
        the radius of the neighborhood used for the CC metric when
        computing the factors. The returned vector field will be
        zero along a boundary of width radius voxels.

    Returns
    -------
    out : array, shape (S, R, C, 3)
        the gradient of the cross correlation metric with respect to the
        displacement associated to the static volume
    energy : the cross correlation energy (data term) at this iteration

    References
    ----------
    .. footbibliography::
    """
    ftype = np.asarray(grad_moving).dtype
    cdef:
        cnp.npy_intp ns = grad_moving.shape[0]
        cnp.npy_intp nr = grad_moving.shape[1]
        cnp.npy_intp nc = grad_moving.shape[2]
        cnp.npy_intp s, r, c
        double energy = 0
        double Ii, Ji, sfm, sff, smm, localCorrelation, temp
        floating[:, :, :, :] out = np.zeros((ns, nr, nc, 3), dtype=ftype)

    with nogil:

        for s in range(radius, ns-radius):
            for r in range(radius, nr-radius):
                for c in range(radius, nc-radius):
                    Ii = factors[s, r, c, 0]
                    Ji = factors[s, r, c, 1]
                    sfm = factors[s, r, c, 2]
                    sff = factors[s, r, c, 3]
                    smm = factors[s, r, c, 4]
                    if sff == 0.0 or smm == 0.0:
                        continue
                    localCorrelation = 0
                    if sff * smm > 1e-5:
                        localCorrelation = sfm * sfm / (sff * smm)
                    if localCorrelation < 1:  # avoid bad values...
                        energy -= localCorrelation
                    temp = 2.0 * sfm / (sff * smm) * (Ii - sfm / smm * Ji)
                    out[s, r, c, 0] -= temp * grad_moving[s, r, c, 0]
                    out[s, r, c, 1] -= temp * grad_moving[s, r, c, 1]
                    out[s, r, c, 2] -= temp * grad_moving[s, r, c, 2]
    return np.asarray(out), energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def precompute_cc_factors_2d(floating[:, :] static, floating[:, :] moving,
                             cnp.npy_intp radius):
    """Precomputations to quickly compute the gradient of the CC Metric.

    Pre-computes the separate terms of the cross correlation metric
    :footcite:p:`Avants2008` and image norms at each voxel considering a
    neighborhood of the given radius to efficiently compute the gradient of the
    metric with respect to the deformation field :footcite:p:`Avants2009`.

    Parameters
    ----------
    static : array, shape (R, C)
        the static volume, which also defines the reference registration domain
    moving : array, shape (R, C)
        the moving volume (notice that both images must already be in a common
        reference domain, i.e. the same R, C)
    radius : the radius of the neighborhood(square of (2*radius + 1)^2 voxels)

    Returns
    -------
    factors : array, shape (R, C, 5)
        the precomputed cross correlation terms::

            - factors[:,:,0] : static minus its mean value along the neighborhood
            - factors[:,:,1] : moving minus its mean value along the neighborhood
            - factors[:,:,2] : sum of the pointwise products of static and moving
              along the neighborhood
            - factors[:,:,3] : sum of sq. values of static along the neighborhood
            - factors[:,:,4] : sum of sq. values of moving along the neighborhood

    References
    ----------
    .. footbibliography::
    """
    ftype = np.asarray(static).dtype
    cdef:
        cnp.npy_intp side = 2 * radius + 1
        cnp.npy_intp nr = static.shape[0]
        cnp.npy_intp nc = static.shape[1]
        cnp.npy_intp r, c, i, j, t, q, qq, firstc, lastc
        double Imean, Jmean
        floating[:, :, :] factors = np.zeros((nr, nc, 5), dtype=ftype)
        double[:, :] lines = np.zeros((6, side), dtype=np.float64)
        double[:] sums = np.zeros((6,), dtype=np.float64)

    with nogil:

        for c in range(nc):
            firstc = _int_max(0, c - radius)
            lastc = _int_min(nc - 1, c + radius)
            # compute factors for row [:,c]
            for t in range(6):
                for q in range(side):
                    lines[t, q] = 0
            # Compute all rows and set the sums on the fly
            # compute row [i, j = {c-radius, c + radius}]
            for i in range(nr):
                q = i % side
                for t in range(6):
                    lines[t, q] = 0
                for j in range(firstc, lastc + 1):
                    lines[SI, q] += static[i, j]
                    lines[SI2, q] += static[i, j] * static[i, j]
                    lines[SJ, q] += moving[i, j]
                    lines[SJ2, q] += moving[i, j] * moving[i, j]
                    lines[SIJ, q] += static[i, j] * moving[i, j]
                    lines[CNT, q] += 1

                for t in range(6):
                    sums[t] = 0
                    for qq in range(side):
                        sums[t] += lines[t, qq]
                if i >= radius:
                    # r is the pixel that is affected by the cube with slices
                    # [r - radius.. r + radius, :]
                    r = i - radius
                    Imean = sums[SI] / sums[CNT]
                    Jmean = sums[SJ] / sums[CNT]
                    factors[r, c, 0] = static[r, c] - Imean
                    factors[r, c, 1] = moving[r, c] - Jmean
                    factors[r, c, 2] = (sums[SIJ] - Jmean * sums[SI] -
                                        Imean * sums[SJ] +
                                        sums[CNT] * Jmean * Imean)
                    factors[r, c, 3] = (sums[SI2] - Imean * sums[SI] -
                                        Imean * sums[SI] +
                                        sums[CNT] * Imean * Imean)
                    factors[r, c, 4] = (sums[SJ2] - Jmean * sums[SJ] -
                                        Jmean * sums[SJ] +
                                        sums[CNT] * Jmean * Jmean)
            # Finally set the values at the end of the line
            for r in range(nr - radius, nr):
                # this would be the last slice to be processed for pixel
                # [r, c], if it existed
                i = r + radius
                q = i % side
                for t in range(6):
                    sums[t] -= lines[t, q]
                Imean = sums[SI] / sums[CNT]
                Jmean = sums[SJ] / sums[CNT]
                factors[r, c, 0] = static[r, c] - Imean
                factors[r, c, 1] = moving[r, c] - Jmean
                factors[r, c, 2] = (sums[SIJ] - Jmean * sums[SI] -
                                    Imean * sums[SJ] +
                                    sums[CNT] * Jmean * Imean)
                factors[r, c, 3] = (sums[SI2] - Imean * sums[SI] -
                                    Imean * sums[SI] +
                                    sums[CNT] * Imean * Imean)
                factors[r, c, 4] = (sums[SJ2] - Jmean * sums[SJ] -
                                    Jmean * sums[SJ] +
                                    sums[CNT] * Jmean * Jmean)
    return np.asarray(factors)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def precompute_cc_factors_2d_test(floating[:, :] static, floating[:, :] moving,
                                  cnp.npy_intp radius):
    """Precomputations to quickly compute the gradient of the CC Metric.

    This version of precompute_cc_factors_2d is for testing purposes, it
    directly computes the local cross-correlation without any optimization.
    """
    ftype = np.asarray(static).dtype
    cdef:
        cnp.npy_intp nr = static.shape[0]
        cnp.npy_intp nc = static.shape[1]
        cnp.npy_intp r, c, i, j, t, firstr, lastr, firstc, lastc
        double Imean, Jmean
        floating[:, :, :] factors = np.zeros((nr, nc, 5), dtype=ftype)
        double[:] sums = np.zeros((6,), dtype=np.float64)

    with nogil:

        for r in range(nr):
            firstr = _int_max(0, r - radius)
            lastr = _int_min(nr - 1, r + radius)
            for c in range(nc):
                firstc = _int_max(0, c - radius)
                lastc = _int_min(nc - 1, c + radius)
                for t in range(6):
                    sums[t] = 0
                for i in range(firstr, 1 + lastr):
                    for j in range(firstc, 1+lastc):
                        sums[SI] += static[i, j]
                        sums[SI2] += static[i, j]**2
                        sums[SJ] += moving[i, j]
                        sums[SJ2] += moving[i, j]**2
                        sums[SIJ] += static[i, j]*moving[i, j]
                        sums[CNT] += 1
                Imean = sums[SI] / sums[CNT]
                Jmean = sums[SJ] / sums[CNT]
                factors[r, c, 0] = static[r, c] - Imean
                factors[r, c, 1] = moving[r, c] - Jmean
                factors[r, c, 2] = (sums[SIJ] - Jmean * sums[SI] -
                                    Imean * sums[SJ] +
                                    sums[CNT] * Jmean * Imean)
                factors[r, c, 3] = (sums[SI2] - Imean * sums[SI] -
                                    Imean * sums[SI] +
                                    sums[CNT] * Imean * Imean)
                factors[r, c, 4] = (sums[SJ2] - Jmean * sums[SJ] -
                                    Jmean * sums[SJ] +
                                    sums[CNT] * Jmean * Jmean)
    return np.asarray(factors)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_cc_forward_step_2d(floating[:, :, :] grad_static,
                               floating[:, :, :] factors,
                               cnp.npy_intp radius):
    """Gradient of the CC Metric w.r.t. the forward transformation.

    Computes the gradient of the Cross Correlation metric for symmetric
    registration (SyN) :footcite:p:`Avants2008` w.r.t. the displacement
    associated to the moving image ('backward' step) as in
    :footcite:t:`Avants2009`.

    Parameters
    ----------
    grad_static : array, shape (R, C, 2)
        the gradient of the static image
    factors : array, shape (R, C, 5)
        the precomputed cross correlation terms obtained via
        precompute_cc_factors_2d

    Returns
    -------
    out : array, shape (R, C, 2)
        the gradient of the cross correlation metric with respect to the
        displacement associated to the moving image
    energy : the cross correlation energy (data term) at this iteration

    Notes
    -----
    Currently, the gradient of the static image is not being used, but some
    authors suggest that symmetrizing the gradient by including both, the
    moving and static gradients may improve the registration quality. We are
    leaving this parameter as a placeholder for future investigation

    References
    ----------
    .. footbibliography::
    """
    cdef:
        cnp.npy_intp nr = grad_static.shape[0]
        cnp.npy_intp nc = grad_static.shape[1]
        double energy = 0
        cnp.npy_intp r, c
        double Ii, Ji, sfm, sff, smm, localCorrelation, temp
        floating[:, :, :] out = np.zeros((nr, nc, 2),
                                         dtype=np.asarray(grad_static).dtype)
    with nogil:

        for r in range(radius, nr-radius):
            for c in range(radius, nc-radius):
                Ii = factors[r, c, 0]
                Ji = factors[r, c, 1]
                sfm = factors[r, c, 2]
                sff = factors[r, c, 3]
                smm = factors[r, c, 4]
                if sff == 0.0 or smm == 0.0:
                    continue
                localCorrelation = 0
                if sff * smm > 1e-5:
                    localCorrelation = sfm * sfm / (sff * smm)
                if localCorrelation < 1:  # avoid bad values...
                    energy -= localCorrelation
                temp = 2.0 * sfm / (sff * smm) * (Ji - sfm / sff * Ii)
                out[r, c, 0] -= temp * grad_static[r, c, 0]
                out[r, c, 1] -= temp * grad_static[r, c, 1]
    return np.asarray(out), energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_cc_backward_step_2d(floating[:, :, :] grad_moving,
                                floating[:, :, :] factors,
                                cnp.npy_intp radius):
    """Gradient of the CC Metric w.r.t. the backward transformation.

    Computes the gradient of the Cross Correlation metric for symmetric
    registration (SyN) :footcite:p:`Avants2008` w.r.t. the displacement
    associated to the static image ('forward' step) as in
    :footcite:t:`Avants2009`.

    Parameters
    ----------
    grad_moving : array, shape (R, C, 2)
        the gradient of the moving image
    factors : array, shape (R, C, 5)
        the precomputed cross correlation terms obtained via
        precompute_cc_factors_2d

    Returns
    -------
    out : array, shape (R, C, 2)
        the gradient of the cross correlation metric with respect to the
        displacement associated to the static image
    energy : the cross correlation energy (data term) at this iteration

    References
    ----------
    .. footbibliography::
    """
    ftype = np.asarray(grad_moving).dtype
    cdef:
        cnp.npy_intp nr = grad_moving.shape[0]
        cnp.npy_intp nc = grad_moving.shape[1]
        cnp.npy_intp r, c
        double energy = 0
        double Ii, Ji, sfm, sff, smm, localCorrelation, temp
        floating[:, :, :] out = np.zeros((nr, nc, 2), dtype=ftype)

    with nogil:

        for r in range(radius, nr-radius):
            for c in range(radius, nc-radius):
                Ii = factors[r, c, 0]
                Ji = factors[r, c, 1]
                sfm = factors[r, c, 2]
                sff = factors[r, c, 3]
                smm = factors[r, c, 4]
                if sff == 0.0 or smm == 0.0:
                    continue
                localCorrelation = 0
                if sff * smm > 1e-5:
                    localCorrelation = sfm * sfm / (sff * smm)
                if localCorrelation < 1:  # avoid bad values...
                    energy -= localCorrelation
                temp = 2.0 * sfm / (sff * smm) * (Ii - sfm / smm * Ji)
                out[r, c, 0] -= temp * grad_moving[r, c, 0]
                out[r, c, 1] -= temp * grad_moving[r, c, 1]
    return np.asarray(out), energy
