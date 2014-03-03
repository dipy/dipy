import numpy as np
cimport cython
from fused_types cimport floating, number

cdef extern from "math.h":
    double sqrt(double x) nogil
    double floor(double x) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _apply_affine_3d_x0(double x0, double x1, double x2,
                                       floating[:, :] aff) nogil:
    r"""
    Returns the first component of the product of the affine matrix aff by
    (x0, x1, x2)
    """
    return aff[0, 0] * x0 + aff[0, 1] * x1 + aff[0, 2] * x2 + aff[0, 3]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _apply_affine_3d_x1(double x0, double x1, double x2,
                                       floating[:, :] aff) nogil:
    r"""
    Returns the first component of the product of the affine matrix aff by
    (x0, x1, x2)
    """
    return aff[1, 0] * x0 + aff[1, 1] * x1 + aff[1, 2] * x2 + aff[1, 3]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _apply_affine_3d_x2(double x0, double x1, double x2,
                                       floating[:, :] aff) nogil:
    r"""
    Returns the first component of the product of the affine matrix aff by
    (x0, x1, x2)
    """
    return aff[2, 0] * x0 + aff[2, 1] * x1 + aff[2, 2] * x2 + aff[2, 3]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _apply_affine_2d_x0(double x0, double x1,
                                       floating[:, :] aff) nogil:
    r"""
    Returns the first component of the product of the aff matrix aff by
    (x0, x1, x2)
    """
    return aff[0, 0] * x0 + aff[0, 1] * x1 + aff[0, 2]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _apply_affine_2d_x1(double x0, double x1,
                                       floating[:, :] aff) nogil:
    r"""
    Returns the first component of the product of the affine matrix aff by
    (x0, x1, x2)
    """
    return aff[1, 0] * x0 + aff[1, 1] * x1 + aff[1, 2]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _compose_vector_fields(floating[:, :, :] d1, floating[:, :, :] d2,
                                 floating[:, :, :] comp, floating[:] stats):
    r"""
    Computes the composition of the two 2-D displacemements d1 and d2 defined by
    comp[r, c] = d2(d1[r, c]) for each (r,c) in the domain of d1. The evaluation
    of d2 at non-lattice points is computed using trilinear interpolation. The
    result is stored in comp.

    Parameters
    ----------
    d1 : array, shape (R, C, 2)
        first 2-D displacement field to be applied. R, C are the number of rows
        and columns of the displacement field d1, respectively.
    d2 : array, shape (R', C', 2)
        second displacement field to be applied. R', C' are the number of rows
        and columns of the displacement field d2, respectively.
    comp : array, shape (R, C, 2), same dimension as d1
        on output, this array will contain the composition of the two fields
    stats : array, shape (3,)
        on output, this array will contain three statistics of the vector norms
        of the composition (maximum, mean, standard_deviation)

    Notes
    -----
    If d1[r,c] lies outside the domain of d2, then comp[r,c] will contain a zero
    vector.
    """
    cdef:
        int nr1 = d1.shape[0]
        int nc1 = d1.shape[1]
        int nr2 = d2.shape[0]
        int nc2 = d2.shape[1]
        int cnt = 0
        floating maxNorm = 0
        floating meanNorm = 0
        floating stdNorm = 0
        floating nn
        int i, j, ii, jj
        floating dii, djj, alpha, beta, calpha, cbeta
    comp[...] = 0
    for i in range(nr1):
        for j in range(nc1):
            dii = i + d1[i, j, 0]
            djj = j + d1[i, j, 1]
            if((dii < 0) or (nr2 - 1 < dii) or (djj < 0) or (nc2 - 1 < djj)):
                continue
            ii = int(dii)
            jj = int(djj)
            if((ii < 0) or (nr2 <= ii) or (jj < 0) or (nc2 <= jj)):
                continue
            calpha = dii - ii
            cbeta = djj - jj
            alpha = 1 - calpha
            beta = 1 - cbeta
            comp[i, j, 0] = d1[i, j, 0]
            comp[i, j, 1] = d1[i, j, 1]
            # top-left
            comp[i, j, 0] += alpha * beta * d2[ii, jj, 0]
            comp[i, j, 1] += alpha * beta * d2[ii, jj, 1]
            # top-right
            jj += 1
            if(jj < nc2):
                comp[i, j, 0] += alpha * cbeta * d2[ii, jj, 0]
                comp[i, j, 1] += alpha * cbeta * d2[ii, jj, 1]
            # bottom-right
            ii += 1
            if((ii >= 0) and (jj >= 0) and (ii < nr2)and (jj < nc2)):
                comp[i, j, 0] += calpha * cbeta * d2[ii, jj, 0]
                comp[i, j, 1] += calpha * cbeta * d2[ii, jj, 1]
            # bottom-left
            jj -= 1
            if((ii >= 0) and (jj >= 0) and (ii < nr2) and (jj < nc2)):
                comp[i, j, 0] += calpha * beta * d2[ii, jj, 0]
                comp[i, j, 1] += calpha * beta * d2[ii, jj, 1]
            # consider only displacements that land inside the image
            if((0 <= dii <= nr2 - 1) and (0 <= djj <= nc2 - 1)):
                nn = comp[i, j, 0] ** 2 + comp[i, j, 1] ** 2
                if(maxNorm < nn):
                    maxNorm = nn
                meanNorm += nn
                stdNorm += nn * nn
                cnt += 1
    meanNorm /= cnt
    stats[0] = sqrt(maxNorm)
    stats[1] = sqrt(meanNorm)
    stats[2] = sqrt(stdNorm / cnt - meanNorm * meanNorm)


def compose_vector_fields(floating[:, :, :] d1, floating[:, :, :] d2):
    r"""
    Computes the composition of the two 2-D displacemements d1 and d2 defined by
    comp[r, c] = d2(d1[r, c]) for each (r,c) in the domain of d1. The evaluation
    of d2 at non-lattice points is computed using trilinear interpolation.

    Parameters
    ----------
    d1 : array, shape (R, C, 2)
        first displacement field to be applied. R, C are the number of rows
        and columns of the displacement field, respectively.
    d2 : array, shape (R', C', 2)
        second displacement field to be applied. R', C' are the number of rows
        and columns of the displacement field, respectively.

    Returns
    -------
    comp : array, shape (R, C, 2), same dimension as d1
        on output, this array will contain the composition of the two fields
    stats : array, shape (3,)
        on output, this array will contain three statistics of the vector norms
        of the composition (maximum, mean, standard_deviation)
    """
    cdef:
        floating[:, :, :] comp = np.zeros_like(d1)
        floating[:] stats = np.zeros(shape=(3,),
                                     dtype=cython.typeof(d1[0, 0, 0]))
    _compose_vector_fields(d1, d2, comp, stats)
    return comp, stats


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _compose_vector_fields_3d(floating[:, :, :, :] d1,
                                    floating[:, :, :, :] d2,
                                    floating[:, :, :, :] comp,
                                    floating[:] stats):
    r"""
    Computes the composition of the two 3-D displacemements d1 and d2 defined by
    comp[s, r, c] = d2(d1[s, r, c]) for each (s,r,c) in the domain of d1.
    The evaluation of d2 at non-lattice points is computed using trilinear
    interpolation. The result is stored in comp.

    Parameters
    ----------
    d1 : array, shape (S, R, C, 3)
        first 3-D displacement field to be applied. S, R, C are the number of
        slices, rows and columns of the displacement field d1, respectively.
    d2 : array, shape (S', R', C', 3)
        second displacement field to be applied. S', R', C' are the number of
        slices, rows and columns of the displacement field d2, respectively.
    comp : array, shape (S, R, C, 3), same dimension as d1
        on output, this array will contain the composition of the two fields
    stats : array, shape (3,)
        on output, this array will contain three statistics of the vector norms
        of the composition (maximum, mean, standard_deviation)

    Notes
    -----
    If d1[s,r,c] lies outside the domain of d2, then comp[s,r,c] will contain a
    zero vector.
    """
    cdef:
        int ns1 = d1.shape[0]
        int nr1 = d1.shape[1]
        int nc1 = d1.shape[2]
        int ns2 = d2.shape[0]
        int nr2 = d2.shape[1]
        int nc2 = d2.shape[2]
        int cnt = 0
        int k, i, j, kk, ii, jj
        floating maxNorm = 0
        floating meanNorm = 0
        floating stdNorm = 0
        floating dkk, dii, djj
        floating alpha, beta, gamma, calpha, cbeta, cgamma, nn

    comp[...] = 0
    for k in range(ns1):
        for i in range(nr1):
            for j in range(nc1):
                dkk = k + d1[k, i, j, 0]
                dii = i + d1[k, i, j, 1]
                djj = j + d1[k, i, j, 2]
                if((dii < 0) or (djj < 0) or (dkk < 0) or (dii > nr2 - 1) or (djj > nc2 - 1) or (dkk > ns2 - 1)):
                    continue
                #---top-left
                kk = int(dkk)
                ii = int(dii)
                jj = int(djj)
                if((ii < 0) or (jj < 0) or (kk < 0) or (ii >= nr2) or (jj >= nc2) or (kk >= ns2)):
                    continue
                cgamma = dkk - kk
                # by definition these factors are nonnegative
                calpha = dii - ii
                cbeta = djj - jj
                alpha = 1 - calpha
                beta = 1 - cbeta
                gamma = 1 - cgamma
                comp[k, i, j, 0] = d1[k, i, j, 0]
                comp[k, i, j, 1] = d1[k, i, j, 1]
                comp[k, i, j, 2] = d1[k, i, j, 2]
                comp[k, i, j, 0] += alpha * beta * gamma * d2[kk, ii, jj, 0]
                comp[k, i, j, 1] += alpha * beta * gamma * d2[kk, ii, jj, 1]
                comp[k, i, j, 2] += alpha * beta * gamma * d2[kk, ii, jj, 2]
                #---top-right
                jj += 1
                if(jj < nc2):
                    comp[k, i, j, 0] += alpha * \
                        cbeta * gamma * d2[kk, ii, jj, 0]
                    comp[k, i, j, 1] += alpha * \
                        cbeta * gamma * d2[kk, ii, jj, 1]
                    comp[k, i, j, 2] += alpha * \
                        cbeta * gamma * d2[kk, ii, jj, 2]
                #---bottom-right
                ii += 1
                if((ii >= 0) and (jj >= 0) and (ii < nr2) and (jj < nc2)):
                    comp[k, i, j, 0] += calpha * \
                        cbeta * gamma * d2[kk, ii, jj, 0]
                    comp[k, i, j, 1] += calpha * \
                        cbeta * gamma * d2[kk, ii, jj, 1]
                    comp[k, i, j, 2] += calpha * \
                        cbeta * gamma * d2[kk, ii, jj, 2]
                #---bottom-left
                jj -= 1
                if((ii >= 0) and (jj >= 0) and (ii < nr2) and (jj < nc2)):
                    comp[k, i, j, 0] += calpha * \
                        beta * gamma * d2[kk, ii, jj, 0]
                    comp[k, i, j, 1] += calpha * \
                        beta * gamma * d2[kk, ii, jj, 1]
                    comp[k, i, j, 2] += calpha * \
                        beta * gamma * d2[kk, ii, jj, 2]
                kk += 1
                if(kk < ns2):
                    ii -= 1
                    comp[k, i, j, 0] += alpha * beta * \
                        cgamma * d2[kk, ii, jj, 0]
                    comp[k, i, j, 1] += alpha * beta * \
                        cgamma * d2[kk, ii, jj, 1]
                    comp[k, i, j, 2] += alpha * beta * \
                        cgamma * d2[kk, ii, jj, 2]
                    jj += 1
                    if(jj < nc2):
                        comp[k, i, j, 0] += alpha * \
                            cbeta * cgamma * d2[kk, ii, jj, 0]
                        comp[k, i, j, 1] += alpha * \
                            cbeta * cgamma * d2[kk, ii, jj, 1]
                        comp[k, i, j, 2] += alpha * \
                            cbeta * cgamma * d2[kk, ii, jj, 2]
                    #---bottom-right
                    ii += 1
                    if((ii >= 0) and (jj >= 0) and (ii < nr2) and (jj < nc2)):
                        comp[k, i, j, 0] += calpha * \
                            cbeta * cgamma * d2[kk, ii, jj, 0]
                        comp[k, i, j, 1] += calpha * \
                            cbeta * cgamma * d2[kk, ii, jj, 1]
                        comp[k, i, j, 2] += calpha * \
                            cbeta * cgamma * d2[kk, ii, jj, 2]
                    #---bottom-left
                    jj -= 1
                    if((ii >= 0) and (jj >= 0) and (ii < nr2) and (jj < nc2)):
                        comp[k, i, j, 0] += calpha * \
                            beta * cgamma * d2[kk, ii, jj, 0]
                        comp[k, i, j, 1] += calpha * \
                            beta * cgamma * d2[kk, ii, jj, 1]
                        comp[k, i, j, 2] += calpha * \
                            beta * cgamma * d2[kk, ii, jj, 2]
                if((0 <= dkk <= ns2 - 1) and (0 <= dii <= nr2 - 1) and (0 <= djj <= nc2 - 1)):
                    nn = comp[k, i, j, 0] ** 2 + \
                        comp[k, i, j, 1] ** 2 + comp[k, i, j, 2] ** 2
                    if(maxNorm < nn):
                        maxNorm = nn
                    meanNorm += nn
                    stdNorm += nn * nn
                    cnt += 1
    meanNorm /= cnt
    stats[0] = sqrt(maxNorm)
    stats[1] = sqrt(meanNorm)
    stats[2] = sqrt(stdNorm / cnt - meanNorm * meanNorm)


def compose_vector_fields_3d(floating[:, :, :, :] d1, floating[:, :, :, :] d2):
    r"""
    Computes the composition of the two 3-D displacemements d1 and d2 defined by
    comp[s, r, c] = d2(d1[s, r, c]) for each (s,r,c) in the domain of d1. The
    evaluation of d2 at non-lattice points is computed using trilinear
    interpolation.

    Parameters
    ----------
    d1 : array, shape (S, R, C, 3)
        first 3-D displacement field to be applied. S, R, C are the number of
        slices, rows and columns of the displacement field, respectively.
    d2 : array, shape (S', R', C', 3)
        second displacement field to be applied. S', R', C' are the number of
        slices, rows and columns of the displacement field, respectively.

    Returns
    -------
    comp : array, shape (S, R, C, 3), same dimension as d1
        on output, this array will contain the composition of the two fields
    stats : array, shape (3,)
        on output, this array will contain three statistics of the vector norms
        of the composition (maximum, mean, standard_deviation)
    """
    cdef:
        floating[:, :, :, :] comp = np.zeros_like(d1)
        floating[:] stats = np.zeros(shape=(3,), dtype=cython.typeof(d1[0, 0, 0, 0]))
    _compose_vector_fields_3d(d1, d2, comp, stats)
    return comp, stats


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def invert_vector_field_fixed_point(floating[:, :, :] d, int[:] inv_shape,
                                    int max_iter, double tolerance,
                                    floating[:, :, :] start=None):
    r"""
    Computes the inverse of the given 2-D displacement field d using the
    fixed-point algorithm.

    Parameters
    ----------
    d : array, shape (R, C, 2)
        the 2-D displacement field to be inverted
    inv_shape : array, shape (2,)
        the expected shape of the inverse displacement field.
    max_iter : int
        maximum number of iterations to be performed
    tolerance : float
        maximum tolerated inversion error
    start : array, shape (R', C')
        an aproximation to the inverse displacemnet field (if no aproximation
        is available, None can be provided and the start displacement fieldwill
        be zero)

    Returns
    -------
    p : array, shape inv_shape+(2,) or (R, C, 2) if inv_shape is None
        the inverse displacement field

    Notes
    -----
    The 'inversion error' at iteration t is defined as the mean norm of the
    displacement vectors of the input displacement field composed with the
    inverse at iteration t. If inv_shape is None, the shape of the resulting
    inverse will be the same as the input displacement field.
    """
    cdef:
        int nr1 = d.shape[0]
        int nc1 = d.shape[1]
        int nr2, nc2, iter_count, current
        floating difmag, mag
        floating epsilon = 0.25
        floating error = 1 + tolerance
    if inv_shape != None:
        nr2, nc2 = inv_shape[0], inv_shape[1]
    else:
        nr2, nc2 = nr1, nc1
    cdef:
        floating[:] stats = np.zeros(shape=(2,), dtype=cython.typeof(d[0, 0, 0]))
        floating[:] substats = np.empty(shape=(3,), dtype=cython.typeof(d[0, 0, 0]))
        floating[:, :, :] p = np.zeros(shape=(nr2, nc2, 2), dtype=cython.typeof(d[0, 0, 0]))
        floating[:, :, :] q = np.zeros(shape=(nr2, nc2, 2), dtype=cython.typeof(d[0, 0, 0]))
    if start != None:
        p[...] = start

    iter_count = 0
    while (iter_count < max_iter) and (tolerance < error):
        p, q = q, p
        _compose_vector_fields(q, d, p, substats)
        difmag = 0
        error = 0
        for i in range(nr2):
            for j in range(nc2):
                mag = sqrt(p[i, j, 0] ** 2 + p[i, j, 1] ** 2)
                p[i, j, 0] = q[i, j, 0] - epsilon * p[i, j, 0]
                p[i, j, 1] = q[i, j, 1] - epsilon * p[i, j, 1]
                error += mag
                if(difmag < mag):
                    difmag = mag
        error /= (nr2 * nc2)
        iter_count += 1
    stats[0] = substats[1]
    stats[1] = iter_count
    return p


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def invert_vector_field_fixed_point_3d(floating[:, :, :, :] d,
                                       int[:] inv_shape,
                                       int max_iter, double tolerance,
                                       floating[:, :, :, :] start=None):
    r"""
    Computes the inverse of the given 3-D displacement field d using the
    fixed-point algorithm.

    Parameters
    ----------
    d : array, shape (S, R, C, 3)
        the 3-D displacement field to be inverted
    inv_shape : array, shape (3,)
        the expected shape of the inverse displacement field.
    max_iter : int
        maximum number of iterations to be performed
    tolerance : float
        maximum tolerated inversion error
    start : array, shape (R', C')
        an aproximation to the inverse displacemnet field (if no aproximation
        is available, None can be provided and the start displacement fieldwill
        be zero)

    Returns
    -------
    p : array, shape inv_shape+(2,) or (S, R, C, 2) if inv_shape is None
        the inverse displacement field

    Notes
    -----
    The 'inversion error' at iteration t is defined as the mean norm of the
    displacement vectors of the input displacement field composed with the
    inverse at iteration t. If inv_shape is None, the shape of the resulting
    inverse will be the same as the input displacement field.
    """
    cdef:
        int ns1 = d.shape[0]
        int nr1 = d.shape[1]
        int nc1 = d.shape[2]
        int ns2, nr2, nc2
        int k, i, j, iter_count
    if inv_shape != None:
        ns2, nr2, nc2 = inv_shape[0], inv_shape[1], inv_shape[2]
    else:
        ns2, nr2, nc2 = ns1, nr1, nc1
    cdef:
        floating[:] stats = np.empty(shape=(2,), dtype=cython.typeof(d[0, 0, 0, 0]))
        floating[:] substats = np.empty(shape=(3,), dtype=cython.typeof(d[0, 0, 0, 0]))
        floating[:, :, :, :] p = np.zeros((ns2, nr2, nc2, 3), dtype=cython.typeof(d[0, 0, 0, 0]))
        floating[:, :, :, :] q = np.zeros((ns2, nr2, nc2, 3), dtype=cython.typeof(d[0, 0, 0, 0]))
        floating error = 1 + tolerance
        floating epsilon = 0.5
        floating mag, difmag
    if start != None:
        p[...] = start
    iter_count = 0
    while (iter_count < max_iter) and (tolerance < error):
        p, q = q, p
        _compose_vector_fields_3d(q, d, p, substats)
        difmag = 0
        error = 0
        for k in range(ns2):
            for i in range(nr2):
                for j in range(nc2):
                    mag = sqrt(p[k, i, j, 0] ** 2 + p[k, i, j, 1]
                               ** 2 + p[k, i, j, 2] ** 2)
                    p[k, i, j, 0] = q[k, i, j, 0] - epsilon * p[k, i, j, 0]
                    p[k, i, j, 1] = q[k, i, j, 1] - epsilon * p[k, i, j, 1]
                    p[k, i, j, 2] = q[k, i, j, 2] - epsilon * p[k, i, j, 2]
                    error += mag
                    if(difmag < mag):
                        difmag = mag
        error /= (ns2 * nr2 * nc2)
        iter_count += 1
    stats[0] = error
    stats[1] = iter_count
    return p


@cython.boundscheck(False)
@cython.wraparound(False)
def prepend_affine_to_displacement_field_2d(floating[:, :, :] d,
                                            floating[:, :] affine):
    r"""
    Modifies the given 2-D displacement field by applying the given affine
    transformation. The resulting transformation T is of the from
    T(x) = d(A*x), where A is the affine transformation.

    Parameters
    ----------
    d : array, shape (R, C, 2)
        the input 2-D displacement field with R rows and C columns
    affine : array, shape (2, 3)
        the matrix representation of the affine transformation to be applied
    """
    if affine == None:
        return
    cdef:
        int nrows = d.shape[0]
        int ncols = d.shape[1]
        int i, j
    for i in range(nrows):
        for j in range(ncols):
            d[i, j, 0] += _apply_affine_2d_x0(i, j, affine) - i
            d[i, j, 1] += _apply_affine_2d_x1(i, j, affine) - j


@cython.boundscheck(False)
@cython.wraparound(False)
def prepend_affine_to_displacement_field_3d(floating[:, :, :, :] d,
                                            floating[:, :] affine):
    r"""
    Modifies thegiven 3-D displacement field by applying the given affine
    transformation. The resulting transformation T is of the from
    T(x) = d(A*x), where A is the affine transformation.

    Parameters
    ----------
    d : array, shape (S, R, C, 3)
        the input 3-D displacement field with S slices, R rows and C columns
    affine : array, shape (3, 4)
        the matrix representation of the affine transformation to be applied
    """
    if affine == None:
        return
    cdef:
        int nslices = d.shape[0]
        int nrows = d.shape[1]
        int ncols = d.shape[2]
        int i, j, k
    for k in range(nslices):
        for i in range(nrows):
            for j in range(ncols):
                d[k, i, j, 0] += _apply_affine_3d_x0(k, i, j, affine) - k
                d[k, i, j, 1] += _apply_affine_3d_x1(k, i, j, affine) - i
                d[k, i, j, 2] += _apply_affine_3d_x2(k, i, j, affine) - j


@cython.boundscheck(False)
@cython.wraparound(False)
def append_affine_to_displacement_field_2d(floating[:, :, :] d,
                                           floating[:, :] affine):
    r"""
    Modifies the given 2-D displacement field by applying the given affine
    transformation. The resulting transformation T is of the from
    T(x) = A*d(x), where A is the affine transformation.

    Parameters
    ----------
    d : array, shape (R, C, 2)
        the input 2-D displacement field with R rows and C columns
    affine : array, shape (2, 3)
        the matrix representation of the affine transformation to be applied
    """
    if affine == None:
        return
    cdef:
        int nrows = d.shape[0]
        int ncols = d.shape[1]
        floating dii, djj
        int i, j
    for i in range(nrows):
        for j in range(ncols):
            dii = d[i, j, 0] + i
            djj = d[i, j, 1] + j
            d[i, j, 0] = _apply_affine_2d_x0(dii, djj, affine) - i
            d[i, j, 1] = _apply_affine_2d_x1(dii, djj, affine) - j


@cython.boundscheck(False)
@cython.wraparound(False)
def append_affine_to_displacement_field_3d(floating[:, :, :, :] d,
                                           floating[:, :] affine):
    r"""
    Modifies thegiven 3-D displacement field by applying the given affine
    transformation. The resulting transformation T is of the from
    T(x) = A*d(x), where A is the affine transformation.

    Parameters
    ----------
    d : array, shape (S, R, C, 3)
        the input 3-D displacement field with S slices, R rows and C columns
    affine : array, shape (3, 4)
        the matrix representation of the affine transformation to be applied
    """
    if affine == None:
        return
    cdef:
        int nslices = d.shape[0]
        int nrows = d.shape[1]
        int ncols = d.shape[2]
        floating dkk, dii, djj
        int i, j, k
    for k in range(nslices):
        for i in range(nrows):
            for j in range(ncols):
                dkk = d[k, i, j, 0] + k
                dii = d[k, i, j, 1] + i
                djj = d[k, i, j, 2] + j
                d[k, i, j, 0] = _apply_affine_3d_x0(dkk, dii, djj, affine) - k
                d[k, i, j, 1] = _apply_affine_3d_x1(dkk, dii, djj, affine) - i
                d[k, i, j, 2] = _apply_affine_3d_x2(dkk, dii, djj, affine) - j


@cython.boundscheck(False)
@cython.wraparound(False)
def upsample_displacement_field(floating[:, :, :] field, int[:] target_shape):
    r"""
    Upsamples de input 2-D displacement field by a factor of 2. The target shape
    (the shape of the resulting upsampled displacement field) must be specified
    to ensure the resulting field has the required dimensions (the input field
    might be the result of subsampling a larger array with odd or even
    dimensions, which cannot be determined from the input dimensions alone).

    Parameters
    ----------
    field : array, shape (R, C, 2)
        the 2-D displacement field to be upsampled
    target_shape : array, shape (2,)
        the intended shape of the resulting upsampled field

    Returns
    -------
    up : array, shape target_shape + (2,)
        the upsampled displacement field
    """
    cdef:
        int nr = field.shape[0]
        int nc = field.shape[1]
        int nrows = target_shape[0]
        int ncols = target_shape[1]
        int i, j, ii, jj
        floating dii, djj
        floating alpha, beta, calpha, cbeta
        floating[:, :, :] up = np.zeros(shape=(nrows, ncols, 2), dtype=cython.typeof(field[0, 0, 0]))
    for i in range(nrows):
        for j in range(ncols):
            dii = 0.5 * i
            djj = 0.5 * j
            # no one is affected
            if((dii < 0) or (djj < 0) or (dii > nr - 1) or (djj > nc - 1)):
                continue
            ii = int(dii)
            jj = int(djj)
            # no one is affected
            if((ii < 0) or (jj < 0) or (ii >= nr) or (jj >= nc)):
                continue
            calpha = dii - ii  # by definition these factors are nonnegative
            cbeta = djj - jj
            alpha = 1 - calpha
            beta = 1 - cbeta
            # top-left
            up[i, j, 0] += alpha * beta * field[ii, jj, 0]
            up[i, j, 1] += alpha * beta * field[ii, jj, 1]
            # top-right
            jj += 1
            if(jj < nc):
                up[i, j, 0] += alpha * cbeta * field[ii, jj, 0]
                up[i, j, 1] += alpha * cbeta * field[ii, jj, 1]
            # bottom-right
            ii += 1
            if((ii >= 0) and (jj >= 0) and (ii < nr) and (jj < nc)):
                up[i, j, 0] += calpha * cbeta * field[ii, jj, 0]
                up[i, j, 1] += calpha * cbeta * field[ii, jj, 1]
            # bottom-left
            jj -= 1
            if((ii >= 0) and (jj >= 0) and (ii < nr) and (jj < nc)):
                up[i, j, 0] += calpha * beta * field[ii, jj, 0]
                up[i, j, 1] += calpha * beta * field[ii, jj, 1]
    return up


@cython.boundscheck(False)
@cython.wraparound(False)
def upsample_displacement_field_3d(floating[:, :, :, :] field,
                                   int[:] target_shape):
    r"""
    Upsamples de input 3-D displacement field by a factor of 2. The target shape
    (the shape of the resulting upsampled displacement field) must be specified
    to ensure the resulting field has the required dimensions (the input field
    might be the result of subsampling a larger array with odd or even
    dimensions, which cannot be determined from the input dimensions alone).

    Parameters
    ----------
    field : array, shape (S, R, C, 3)
        the 3-D displacement field to be upsampled
    target_shape : array, shape (3,)
        the intended shape of the resulting upsampled field

    Returns
    -------
    up : array, shape target_shape + (3,)
        the upsampled displacement field
    """
    cdef:
        int nslices = field.shape[0]
        int nrows = field.shape[1]
        int ncols = field.shape[2]
        int ns = target_shape[0]
        int nr = target_shape[1]
        int nc = target_shape[2]
        int i, j, k, ii, jj, kk
        floating dkk, dii, djj
        floating alpha, beta, gamma, calpha, cbeta, cgamma
        floating[:, :, :, :] up = np.zeros(shape=(ns, nr, nc, 3), dtype=cython.typeof(field[0, 0, 0, 0]))
    for k in range(ns):
        for i in range(nr):
            for j in range(nc):
                dkk = 0.5 * k
                dii = 0.5 * i
                djj = 0.5 * j
                # no one is affected
                if((dkk < 0) or (dii < 0) or (djj < 0) or (dii > nrows - 1) or (djj > ncols - 1) or (dkk > nslices - 1)):
                    continue
                kk = int(dkk)
                ii = int(dii)
                jj = int(djj)
                # no one is affected
                if((kk < 0) or (ii < 0) or (jj < 0) or (ii >= nrows) or (jj >= ncols) or (kk >= nslices)):
                    continue
                cgamma = dkk - kk
                # by definition these factors are nonnegative
                calpha = dii - ii
                cbeta = djj - jj
                alpha = 1 - calpha
                beta = 1 - cbeta
                gamma = 1 - cgamma
                # top-left
                up[k, i, j, 0] += alpha * beta * gamma * field[kk, ii, jj, 0]
                up[k, i, j, 1] += alpha * beta * gamma * field[kk, ii, jj, 1]
                up[k, i, j, 2] += alpha * beta * gamma * field[kk, ii, jj, 2]
                # top-right
                jj += 1
                if(jj < ncols):
                    up[k, i, j, 0] += alpha * cbeta * \
                        gamma * field[kk, ii, jj, 0]
                    up[k, i, j, 1] += alpha * cbeta * \
                        gamma * field[kk, ii, jj, 1]
                    up[k, i, j, 2] += alpha * cbeta * \
                        gamma * field[kk, ii, jj, 2]
                # bottom-right
                ii += 1
                if((ii >= 0)and(jj >= 0)and(ii < nrows)and(jj < ncols)):
                    up[k, i, j, 0] += calpha * cbeta * \
                        gamma * field[kk, ii, jj, 0]
                    up[k, i, j, 1] += calpha * cbeta * \
                        gamma * field[kk, ii, jj, 1]
                    up[k, i, j, 2] += calpha * cbeta * \
                        gamma * field[kk, ii, jj, 2]
                # bottom-left
                jj -= 1
                if((ii >= 0)and(jj >= 0)and(ii < nrows)and(jj < ncols)):
                    up[k, i, j, 0] += calpha * beta * \
                        gamma * field[kk, ii, jj, 0]
                    up[k, i, j, 1] += calpha * beta * \
                        gamma * field[kk, ii, jj, 1]
                    up[k, i, j, 2] += calpha * beta * \
                        gamma * field[kk, ii, jj, 2]
                kk += 1
                if(kk < nslices):
                    ii -= 1
                    up[k, i, j, 0] += alpha * beta * \
                        cgamma * field[kk, ii, jj, 0]
                    up[k, i, j, 1] += alpha * beta * \
                        cgamma * field[kk, ii, jj, 1]
                    up[k, i, j, 2] += alpha * beta * \
                        cgamma * field[kk, ii, jj, 2]
                    jj += 1
                    if(jj < ncols):
                        up[k, i, j, 0] += alpha * cbeta * \
                            cgamma * field[kk, ii, jj, 0]
                        up[k, i, j, 1] += alpha * cbeta * \
                            cgamma * field[kk, ii, jj, 1]
                        up[k, i, j, 2] += alpha * cbeta * \
                            cgamma * field[kk, ii, jj, 2]
                    # bottom-right
                    ii += 1
                    if((ii >= 0)and(jj >= 0)and(ii < nrows)and(jj < ncols)):
                        up[k, i, j, 0] += calpha * cbeta * \
                            cgamma * field[kk, ii, jj, 0]
                        up[k, i, j, 1] += calpha * cbeta * \
                            cgamma * field[kk, ii, jj, 1]
                        up[k, i, j, 2] += calpha * cbeta * \
                            cgamma * field[kk, ii, jj, 2]
                    # bottom-left
                    jj -= 1
                    if((ii >= 0)and(jj >= 0)and(ii < nrows)and(jj < ncols)):
                        up[k, i, j, 0] += calpha * beta * \
                            cgamma * field[kk, ii, jj, 0]
                        up[k, i, j, 1] += calpha * beta * \
                            cgamma * field[kk, ii, jj, 1]
                        up[k, i, j, 2] += calpha * beta * \
                            cgamma * field[kk, ii, jj, 2]
    return up


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def accumulate_upsample_displacement_field3D(floating[:, :, :, :] field,
                                             floating[:, :, :, :] up):
    cdef int nslices = field.shape[0]
    cdef int nrows = field.shape[1]
    cdef int ncols = field.shape[2]
    cdef int ns = up.shape[0]
    cdef int nr = up.shape[1]
    cdef int nc = up.shape[2]
    cdef int i, j, k, ii, jj, kk
    cdef floating dkk, dii, djj
    cdef floating alpha, beta, gamma, calpha, cbeta, cgamma
    for k in range(ns):
        for i in range(nr):
            for j in range(nc):
                dkk = 0.5 * k
                dii = 0.5 * i
                djj = 0.5 * j
                # no one is affected
                if((dkk < 0) or (dii < 0) or (djj < 0) or (dii > nrows - 1) or (djj > ncols - 1) or (dkk > nslices - 1)):
                    continue
                kk = int(dkk)
                ii = int(dii)
                jj = int(djj)
                # no one is affected
                if((kk < 0) or (ii < 0) or (jj < 0) or (ii >= nrows) or (jj >= ncols) or (kk >= nslices)):
                    continue
                cgamma = dkk - kk
                # by definition these factors are nonnegative
                calpha = dii - ii
                cbeta = djj - jj
                alpha = 1 - calpha
                beta = 1 - cbeta
                gamma = 1 - cgamma
                #---top-left
                up[k, i, j, 0] += alpha * beta * gamma * field[kk, ii, jj, 0]
                up[k, i, j, 1] += alpha * beta * gamma * field[kk, ii, jj, 1]
                up[k, i, j, 2] += alpha * beta * gamma * field[kk, ii, jj, 2]
                #---top-right
                jj += 1
                if(jj < ncols):
                    up[k, i, j, 0] += alpha * cbeta * \
                        gamma * field[kk, ii, jj, 0]
                    up[k, i, j, 1] += alpha * cbeta * \
                        gamma * field[kk, ii, jj, 1]
                    up[k, i, j, 2] += alpha * cbeta * \
                        gamma * field[kk, ii, jj, 2]
                #---bottom-right
                ii += 1
                if((ii >= 0)and(jj >= 0)and(ii < nrows)and(jj < ncols)):
                    up[k, i, j, 0] += calpha * cbeta * \
                        gamma * field[kk, ii, jj, 0]
                    up[k, i, j, 1] += calpha * cbeta * \
                        gamma * field[kk, ii, jj, 1]
                    up[k, i, j, 2] += calpha * cbeta * \
                        gamma * field[kk, ii, jj, 2]
                #---bottom-left
                jj -= 1
                if((ii >= 0)and(jj >= 0)and(ii < nrows)and(jj < ncols)):
                    up[k, i, j, 0] += calpha * beta * \
                        gamma * field[kk, ii, jj, 0]
                    up[k, i, j, 1] += calpha * beta * \
                        gamma * field[kk, ii, jj, 1]
                    up[k, i, j, 2] += calpha * beta * \
                        gamma * field[kk, ii, jj, 2]
                kk += 1
                if(kk < nslices):
                    ii -= 1
                    up[k, i, j, 0] += alpha * beta * \
                        cgamma * field[kk, ii, jj, 0]
                    up[k, i, j, 1] += alpha * beta * \
                        cgamma * field[kk, ii, jj, 1]
                    up[k, i, j, 2] += alpha * beta * \
                        cgamma * field[kk, ii, jj, 2]
                    jj += 1
                    if(jj < ncols):
                        up[k, i, j, 0] += alpha * cbeta * \
                            cgamma * field[kk, ii, jj, 0]
                        up[k, i, j, 1] += alpha * cbeta * \
                            cgamma * field[kk, ii, jj, 1]
                        up[k, i, j, 2] += alpha * cbeta * \
                            cgamma * field[kk, ii, jj, 2]
                    #---bottom-right
                    ii += 1
                    if((ii >= 0)and(jj >= 0)and(ii < nrows)and(jj < ncols)):
                        up[k, i, j, 0] += calpha * cbeta * \
                            cgamma * field[kk, ii, jj, 0]
                        up[k, i, j, 1] += calpha * cbeta * \
                            cgamma * field[kk, ii, jj, 1]
                        up[k, i, j, 2] += calpha * cbeta * \
                            cgamma * field[kk, ii, jj, 2]
                    #---bottom-left
                    jj -= 1
                    if((ii >= 0)and(jj >= 0)and(ii < nrows)and(jj < ncols)):
                        up[k, i, j, 0] += calpha * beta * \
                            cgamma * field[kk, ii, jj, 0]
                        up[k, i, j, 1] += calpha * beta * \
                            cgamma * field[kk, ii, jj, 1]
                        up[k, i, j, 2] += calpha * beta * \
                            cgamma * field[kk, ii, jj, 2]
    return up


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def downsample_scalar_field3D(floating[:, :, :] field):
    cdef int ns = field.shape[0]
    cdef int nr = field.shape[1]
    cdef int nc = field.shape[2]
    cdef int nns = (ns + 1) // 2
    cdef int nnr = (nr + 1) // 2
    cdef int nnc = (nc + 1) // 2
    cdef int i, j, k, ii, jj, kk
    cdef floating[:, :, :] down = np.zeros((nns, nnr, nnc), dtype=cython.typeof(field[0, 0, 0]))
    cdef int[:, :, :] cnt = np.zeros((nns, nnr, nnc), dtype=np.int32)
    for k in range(ns):
        for i in range(nr):
            for j in range(nc):
                kk = k // 2
                ii = i // 2
                jj = j // 2
                down[kk, ii, jj] += field[k, i, j]
                cnt[kk, ii, jj] += 1
    for k in range(nns):
        for i in range(nnr):
            for j in range(nnc):
                if cnt[k, i, j] > 0:
                    down[k, i, j] /= cnt[k, i, j]
    return down


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def downsample_displacement_field3D(floating[:, :, :, :] field):
    cdef int ns = field.shape[0]
    cdef int nr = field.shape[1]
    cdef int nc = field.shape[2]
    cdef int nns = (ns + 1) // 2
    cdef int nnr = (nr + 1) // 2
    cdef int nnc = (nc + 1) // 2
    cdef int i, j, k, ii, jj, kk
    cdef floating[:, :, :, :] down = np.zeros((nns, nnr, nnc, 3), dtype=cython.typeof(field[0, 0, 0, 0]))
    cdef int[:, :, :] cnt = np.zeros((nns, nnr, nnc), dtype=np.int32)
    for k in range(ns):
        for i in range(nr):
            for j in range(nc):
                kk = k // 2
                ii = i // 2
                jj = j // 2
                down[kk, ii, jj, 0] += field[k, i, j, 0]
                down[kk, ii, jj, 1] += field[k, i, j, 1]
                down[kk, ii, jj, 2] += field[k, i, j, 2]
                cnt[kk, ii, jj] += 1
    for k in range(nns):
        for i in range(nnr):
            for j in range(nnc):
                if cnt[k, i, j] > 0:
                    down[k, i, j, 0] /= cnt[k, i, j]
                    down[k, i, j, 1] /= cnt[k, i, j]
                    down[k, i, j, 2] /= cnt[k, i, j]
    return down


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def downsample_scalar_field2D(floating[:, :] field):
    cdef int nr = field.shape[0]
    cdef int nc = field.shape[1]
    cdef int nnr = (nr + 1) // 2
    cdef int nnc = (nc + 1) // 2
    cdef int i, j, ii, jj
    cdef floating[:, :] down = np.zeros(shape=(nnr, nnc), dtype=cython.typeof(field[0, 0]))
    cdef int[:, :] cnt = np.zeros(shape=(nnr, nnc), dtype=np.int32)
    for i in range(nr):
        for j in range(nc):
            ii = i // 2
            jj = j // 2
            down[ii, jj] += field[i, j]
            cnt[ii, jj] += 1
    for i in range(nnr):
        for j in range(nnc):
            if cnt[i, j] > 0:
                down[i, j] /= cnt[i, j]
    return down


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def downsample_displacement_field2D(floating[:, :, :] field):
    cdef int nr = field.shape[0]
    cdef int nc = field.shape[1]
    cdef int nnr = (nr + 1) // 2
    cdef int nnc = (nc + 1) // 2
    cdef int i, j, ii, jj
    cdef floating[:, :, :] down = np.zeros((nnr, nnc, 2), dtype=cython.typeof(field[0, 0, 0]))
    cdef int[:, :] cnt = np.zeros((nnr, nnc), dtype=np.int32)
    for i in range(nr):
        for j in range(nc):
            ii = i // 2
            jj = j // 2
            down[ii, jj, 0] += field[i, j, 0]
            down[ii, jj, 1] += field[i, j, 1]
            cnt[ii, jj] += 1
    for i in range(nnr):
        for j in range(nnc):
            if cnt[i, j] > 0:
                down[i, j, 0] /= cnt[i, j]
                down[i, j, 1] /= cnt[i, j]
    return down


@cython.boundscheck(False)
@cython.wraparound(False)
def get_displacement_range(floating[:, :, :, :] d, floating[:, :] affine):
    cdef int nslices = d.shape[0]
    cdef int nrows = d.shape[1]
    cdef int ncols = d.shape[2]
    cdef int i, j, k
    cdef floating dkk, dii, djj
    cdef floating[:] minVal = np.ndarray((3,), dtype=cython.typeof(d[0, 0, 0, 0]))
    cdef floating[:] maxVal = np.ndarray((3,), dtype=cython.typeof(d[0, 0, 0, 0]))
    minVal[...] = d[0, 0, 0, :]
    maxVal[...] = minVal[...]
    for k in range(nslices):
        for i in range(nrows):
            for j in range(ncols):
                if(affine != None):
                    dkk = _apply_affine_3d_x0(k, i, j, affine) + d[k, i, j, 0]
                    dii = _apply_affine_3d_x1(k, i, j, affine) + d[k, i, j, 1]
                    djj = _apply_affine_3d_x2(k, i, j, affine) + d[k, i, j, 2]
                else:
                    dkk = k + d[k, i, j, 0]
                    dii = i + d[k, i, j, 1]
                    djj = j + d[k, i, j, 2]
                if(dkk > maxVal[0]):
                    maxVal[0] = dkk
                if(dii > maxVal[1]):
                    maxVal[1] = dii
                if(djj > maxVal[2]):
                    maxVal[2] = djj
    return minVal, maxVal


@cython.boundscheck(False)
@cython.wraparound(False)
def warp_volume(floating[:, :, :] volume, floating[:, :, :, :] d1,
                floating[:, :] affinePre=None, floating[:, :] affinePost=None):
    cdef int nslices = volume.shape[0]
    cdef int nrows = volume.shape[1]
    cdef int ncols = volume.shape[2]
    cdef int nsVol = volume.shape[0]
    cdef int nrVol = volume.shape[1]
    cdef int ncVol = volume.shape[2]
    cdef int i, j, k, ii, jj, kk
    cdef floating dkk, dii, djj, tmp0, tmp1
    cdef floating alpha, beta, gamma, calpha, cbeta, cgamma
    if d1 != None:
        nslices = d1.shape[0]
        nrows = d1.shape[1]
        ncols = d1.shape[2]
    cdef floating[:, :, :] warped = np.zeros(shape=(nslices, nrows, ncols), dtype=cython.typeof(volume[0, 0, 0]))
    for k in range(nslices):
        for i in range(nrows):
            for j in range(ncols):
                if(affinePre != None):
                    dkk = _apply_affine_3d_x0(
                        k, i, j, affinePre) + d1[k, i, j, 0]
                    dii = _apply_affine_3d_x1(
                        k, i, j, affinePre) + d1[k, i, j, 1]
                    djj = _apply_affine_3d_x2(
                        k, i, j, affinePre) + d1[k, i, j, 2]
                else:
                    dkk = k + d1[k, i, j, 0]
                    dii = i + d1[k, i, j, 1]
                    djj = j + d1[k, i, j, 2]
                if(affinePost != None):
                    tmp0 = _apply_affine_3d_x0(dkk, dii, djj, affinePost)
                    tmp1 = _apply_affine_3d_x1(dkk, dii, djj, affinePost)
                    djj = _apply_affine_3d_x2(dkk, dii, djj, affinePost)
                    dii = tmp1
                    dkk = tmp0
                # no one is affected
                if((dii < 0) or (djj < 0) or (dkk < 0) or (dii > nrVol - 1) or (djj > ncVol - 1) or (dkk > nsVol - 1)):
                    continue
                # find the top left index and the interpolation coefficients
                kk = int(dkk)
                ii = int(dii)
                jj = int(djj)
                # no one is affected
                if((ii < 0) or (jj < 0) or (kk < 0) or (ii >= nrVol) or (jj >= ncVol) or (kk >= nsVol)):
                    continue
                cgamma = dkk - kk
                # by definition these factors are nonnegative
                calpha = dii - ii
                cbeta = djj - jj
                alpha = 1 - calpha
                beta = 1 - cbeta
                gamma = 1 - cgamma
                #---top-left
                warped[k, i, j] = alpha * beta * gamma * volume[kk, ii, jj]
                #---top-right
                jj += 1
                if(jj < ncVol):
                    warped[k, i, j] += alpha * cbeta * \
                        gamma * volume[kk, ii, jj]
                #---bottom-right
                ii += 1
                if((ii >= 0) and (jj >= 0) and (ii < nrVol) and (jj < ncVol)):
                    warped[k, i, j] += calpha * \
                        cbeta * gamma * volume[kk, ii, jj]
                #---bottom-left
                jj -= 1
                if((ii >= 0) and (jj >= 0) and (ii < nrVol) and (jj < ncVol)):
                    warped[k, i, j] += calpha * beta * \
                        gamma * volume[kk, ii, jj]
                kk += 1
                if(kk < nsVol):
                    ii -= 1
                    warped[k, i, j] += alpha * beta * \
                        cgamma * volume[kk, ii, jj]
                    jj += 1
                    if(jj < ncVol):
                        warped[k, i, j] += alpha * cbeta * \
                            cgamma * volume[kk, ii, jj]
                    #---bottom-right
                    ii += 1
                    if((ii >= 0) and (jj >= 0) and (ii < nrVol) and (jj < ncVol)):
                        warped[k, i, j] += calpha * cbeta * \
                            cgamma * volume[kk, ii, jj]
                    #---bottom-left
                    jj -= 1
                    if((ii >= 0) and (jj >= 0) and (ii < nrVol) and (jj < ncVol)):
                        warped[k, i, j] += calpha * beta * \
                            cgamma * volume[kk, ii, jj]
    return warped


@cython.boundscheck(False)
@cython.wraparound(False)
def warp_volume_affine(floating[:, :, :] volume, int[:]refShape,
                       floating[:, :] affine):
    cdef int nslices = refShape[0]
    cdef int nrows = refShape[1]
    cdef int ncols = refShape[2]
    cdef int nsVol = volume.shape[0]
    cdef int nrVol = volume.shape[1]
    cdef int ncVol = volume.shape[2]
    cdef int i, j, k, ii, jj, kk
    cdef floating dkk, dii, djj, tmp0, tmp1
    cdef floating alpha, beta, gamma, calpha, cbeta, cgamma
    cdef floating[:, :, :] warped = np.zeros(shape=(nslices, nrows, ncols), dtype=cython.typeof(volume[0, 0, 0]))
    for k in range(nslices):
        for i in range(nrows):
            for j in range(ncols):
                if(affine != None):
                    dkk = _apply_affine_3d_x0(k, i, j, affine)
                    dii = _apply_affine_3d_x1(k, i, j, affine)
                    djj = _apply_affine_3d_x2(k, i, j, affine)
                else:
                    dkk = k
                    dii = i
                    djj = j
                # no one is affected
                if((dii < 0) or (djj < 0) or (dkk < 0) or (dii > nrVol - 1) or (djj > ncVol - 1) or (dkk > nsVol - 1)):
                    continue
                # find the top left index and the interpolation coefficients
                kk = int(dkk)
                ii = int(dii)
                jj = int(djj)
                # no one is affected
                if((ii < 0) or (jj < 0) or (kk < 0) or (ii >= nrVol) or (jj >= ncVol) or (kk >= nsVol)):
                    continue
                cgamma = dkk - kk
                # by definition these factors are nonnegative
                calpha = dii - ii
                cbeta = djj - jj
                alpha = 1 - calpha
                beta = 1 - cbeta
                gamma = 1 - cgamma
                #---top-left
                warped[k, i, j] = alpha * beta * gamma * volume[kk, ii, jj]
                #---top-right
                jj += 1
                if(jj < ncVol):
                    warped[k, i, j] += alpha * cbeta * \
                        gamma * volume[kk, ii, jj]
                #---bottom-right
                ii += 1
                if((ii >= 0) and (jj >= 0) and (ii < nrVol) and (jj < ncVol)):
                    warped[k, i, j] += calpha * \
                        cbeta * gamma * volume[kk, ii, jj]
                #---bottom-left
                jj -= 1
                if((ii >= 0) and (jj >= 0) and (ii < nrVol) and (jj < ncVol)):
                    warped[k, i, j] += calpha * beta * \
                        gamma * volume[kk, ii, jj]
                kk += 1
                if(kk < nsVol):
                    ii -= 1
                    warped[k, i, j] += alpha * beta * \
                        cgamma * volume[kk, ii, jj]
                    jj += 1
                    if(jj < ncVol):
                        warped[k, i, j] += alpha * cbeta * \
                            cgamma * volume[kk, ii, jj]
                    #---bottom-right
                    ii += 1
                    if((ii >= 0) and (jj >= 0) and (ii < nrVol) and (jj < ncVol)):
                        warped[k, i, j] += calpha * cbeta * \
                            cgamma * volume[kk, ii, jj]
                    #---bottom-left
                    jj -= 1
                    if((ii >= 0) and (jj >= 0) and (ii < nrVol) and (jj < ncVol)):
                        warped[k, i, j] += calpha * beta * \
                            cgamma * volume[kk, ii, jj]
    return warped


@cython.boundscheck(False)
@cython.wraparound(False)
def warp_volume_nn(number[:, :, :] volume, floating[:, :, :, :] displacement,
                   floating[:, :] affinePre=None,
                   floating[:, :] affinePost=None):
    cdef int nslices = displacement.shape[0]
    cdef int nrows = displacement.shape[1]
    cdef int ncols = displacement.shape[2]
    cdef int nsVol = volume.shape[0]
    cdef int nrVol = volume.shape[1]
    cdef int ncVol = volume.shape[2]
    cdef floating dkk, dii, djj, tmp0, tmp1
    cdef floating alpha, beta, gamma, calpha, cbeta, cgamma
    cdef int k, i, j, kk, ii, jj
    cdef number[:, :, :] warped = np.zeros((nslices, nrows, ncols), dtype=np.asarray(volume).dtype)
    for k in range(nslices):
        for i in range(nrows):
            for j in range(ncols):
                if(affinePre != None):
                    dkk = _apply_affine_3d_x0(
                        k, i, j, affinePre) + displacement[k, i, j, 0]
                    dii = _apply_affine_3d_x1(
                        k, i, j, affinePre) + displacement[k, i, j, 1]
                    djj = _apply_affine_3d_x2(
                        k, i, j, affinePre) + displacement[k, i, j, 2]
                else:
                    dkk = k + displacement[k, i, j, 0]
                    dii = i + displacement[k, i, j, 1]
                    djj = j + displacement[k, i, j, 2]
                if(affinePost != None):
                    tmp0 = _apply_affine_3d_x0(dkk, dii, djj, affinePost)
                    tmp1 = _apply_affine_3d_x1(dkk, dii, djj, affinePost)
                    djj = _apply_affine_3d_x2(dkk, dii, djj, affinePost)
                    dii = tmp1
                    dkk = tmp0
                # no one is affected
                if((dii < 0) or (djj < 0) or (dkk < 0) or (dii > nrVol - 1) or (djj > ncVol - 1) or (dkk > nsVol - 1)):
                    continue
                # find the top left index and the interpolation coefficients
                kk = int(dkk)
                ii = int(dii)
                jj = int(djj)
                # no one is affected
                if((ii < 0) or (jj < 0) or (kk < 0) or (ii >= nrVol) or (jj >= ncVol) or (kk >= nsVol)):
                    continue
                cgamma = dkk - kk
                # by definition these factors are nonnegative
                calpha = dii - ii
                cbeta = djj - jj
                alpha = 1 - calpha
                beta = 1 - cbeta
                gamma = 1 - cgamma
                if(gamma < cgamma):
                    kk += 1
                if(alpha < calpha):
                    ii += 1
                if(beta < cbeta):
                    jj += 1
                # no one is affected
                if((ii < 0) or (jj < 0) or (kk < 0) or (ii >= nrVol) or (jj >= ncVol) or (kk >= nsVol)):
                    continue
                else:
                    warped[k, i, j] = volume[kk, ii, jj]
    return warped


@cython.boundscheck(False)
@cython.wraparound(False)
def warp_volume_affine_nn(number[:, :, :] volume, int[:] refShape,
                          floating[:, :] affine=None):
    cdef int nslices = refShape[0]
    cdef int nrows = refShape[1]
    cdef int ncols = refShape[2]
    cdef int nsVol = volume.shape[0]
    cdef int nrVol = volume.shape[1]
    cdef int ncVol = volume.shape[2]
    cdef floating dkk, dii, djj, tmp0, tmp1
    cdef floating alpha, beta, gamma, calpha, cbeta, cgamma
    cdef int k, i, j, kk, ii, jj
    cdef number[:, :, :] warped = np.zeros((nslices, nrows, ncols), dtype=np.asarray(volume).dtype)
    for k in range(nslices):
        for i in range(nrows):
            for j in range(ncols):
                if(affine != None):
                    dkk = _apply_affine_3d_x0(k, i, j, affine)
                    dii = _apply_affine_3d_x1(k, i, j, affine)
                    djj = _apply_affine_3d_x2(k, i, j, affine)
                else:
                    dkk = k
                    dii = i
                    djj = j
                # no one is affected
                if((dii < 0) or (djj < 0) or (dkk < 0) or (dii > nrVol - 1) or (djj > ncVol - 1) or (dkk > nsVol - 1)):
                    continue
                # find the top left index and the interpolation coefficients
                kk = int(dkk)
                ii = int(dii)
                jj = int(djj)
                # no one is affected
                if((ii < 0) or (jj < 0) or (kk < 0) or (ii >= nrVol) or (jj >= ncVol) or (kk >= nsVol)):
                    continue
                cgamma = dkk - kk
                # by definition these factors are nonnegative
                calpha = dii - ii
                cbeta = djj - jj
                alpha = 1 - calpha
                beta = 1 - cbeta
                gamma = 1 - cgamma
                if(gamma < cgamma):
                    kk += 1
                if(alpha < calpha):
                    ii += 1
                if(beta < cbeta):
                    jj += 1
                # no one is affected
                if((ii < 0) or (jj < 0) or (kk < 0) or (ii >= nrVol) or (jj >= ncVol) or (kk >= nsVol)):
                    continue
                else:
                    warped[k, i, j] = volume[kk, ii, jj]
    return warped


@cython.boundscheck(False)
@cython.wraparound(False)
def warp_image(floating[:, :] image, floating[:, :, :] d1,
               floating[:, :] affinePre=None, floating[:, :] affinePost=None):
    cdef int nrows = image.shape[0]
    cdef int ncols = image.shape[1]
    cdef int nrVol = image.shape[0]
    cdef int ncVol = image.shape[1]
    cdef int i, j, ii, jj
    cdef floating dii, djj, tmp0
    cdef floating alpha, beta, calpha, cbeta
    if d1 != None:
        nrows = d1.shape[0]
        ncols = d1.shape[1]
    cdef floating[:, :] warped = np.zeros(shape=(nrows, ncols), dtype=cython.typeof(image[0, 0]))
    for i in range(nrows):
        for j in range(ncols):
            if(affinePre != None):
                dii = _apply_affine_2d_x0(i, j, affinePre) + d1[i, j, 0]
                djj = _apply_affine_2d_x1(i, j, affinePre) + d1[i, j, 1]
            else:
                dii = i + d1[i, j, 0]
                djj = j + d1[i, j, 1]
            if(affinePost != None):
                tmp0 = _apply_affine_2d_x0(dii, djj, affinePost)
                djj = _apply_affine_2d_x1(dii, djj, affinePost)
                dii = tmp0
            # no one is affected
            if((dii < 0) or (djj < 0) or (dii > nrVol - 1) or (djj > ncVol - 1)):
                continue
            # find the top left index and the interpolation coefficients
            ii = int(dii)
            jj = int(djj)
            # no one is affected
            if((ii < 0) or (jj < 0) or (ii >= nrVol) or (jj >= ncVol)):
                continue
            calpha = dii - ii  # by definition these factors are nonnegative
            cbeta = djj - jj
            alpha = 1 - calpha
            beta = 1 - cbeta
            #---top-left
            warped[i, j] = alpha * beta * image[ii, jj]
            #---top-right
            jj += 1
            if(jj < ncVol):
                warped[i, j] += alpha * cbeta * image[ii, jj]
            #---bottom-right
            ii += 1
            if((ii >= 0) and (jj >= 0) and (ii < nrVol) and (jj < ncVol)):
                warped[i, j] += calpha * cbeta * image[ii, jj]
            #---bottom-left
            jj -= 1
            if((ii >= 0) and (jj >= 0) and (ii < nrVol) and (jj < ncVol)):
                warped[i, j] += calpha * beta * image[ii, jj]
    return warped


@cython.boundscheck(False)
@cython.wraparound(False)
def warp_image_affine(floating[:, :] image, int[:] refShape,
                      floating[:, :] affine=None):
    cdef int nrows = refShape[0]
    cdef int ncols = refShape[1]
    cdef int nrVol = image.shape[0]
    cdef int ncVol = image.shape[1]
    cdef int i, j, ii, jj
    cdef floating dii, djj, tmp0
    cdef floating alpha, beta, calpha, cbeta
    cdef floating[:, :] warped = np.zeros(shape=(nrows, ncols), dtype=cython.typeof(image[0, 0]))
    for i in range(nrows):
        for j in range(ncols):
            if(affine != None):
                dii = _apply_affine_2d_x0(i, j, affine)
                djj = _apply_affine_2d_x1(i, j, affine)
            else:
                dii = i
                djj = j
            # no one is affected
            if((dii < 0) or (djj < 0) or (dii > nrVol - 1) or (djj > ncVol - 1)):
                continue
            # find the top left index and the interpolation coefficients
            ii = int(dii)
            jj = int(djj)
            # no one is affected
            if((ii < 0) or (jj < 0) or (ii >= nrVol) or (jj >= ncVol)):
                continue
            calpha = dii - ii  # by definition these factors are nonnegative
            cbeta = djj - jj
            alpha = 1 - calpha
            beta = 1 - cbeta
            #---top-left
            warped[i, j] = alpha * beta * image[ii, jj]
            #---top-right
            jj += 1
            if(jj < ncVol):
                warped[i, j] += alpha * cbeta * image[ii, jj]
            #---bottom-right
            ii += 1
            if((ii >= 0) and (jj >= 0) and (ii < nrVol) and (jj < ncVol)):
                warped[i, j] += calpha * cbeta * image[ii, jj]
            #---bottom-left
            jj -= 1
            if((ii >= 0) and (jj >= 0) and (ii < nrVol) and (jj < ncVol)):
                warped[i, j] += calpha * beta * image[ii, jj]
    return warped


@cython.boundscheck(False)
@cython.wraparound(False)
def warp_image_nn(number[:, :] image, floating[:, :, :] displacement,
                  floating[:, :] affinePre=None,
                  floating[:, :] affinePost=None):
    cdef int nrows = image.shape[0]
    cdef int ncols = image.shape[1]
    cdef int nrVol = image.shape[0]
    cdef int ncVol = image.shape[1]
    cdef floating dii, djj, tmp0
    cdef floating alpha, beta, calpha, cbeta
    cdef int i, j, ii, jj
    if displacement != None:
        nrows = displacement.shape[0]
        ncols = displacement.shape[1]
    cdef number[:, :] warped = np.zeros((nrows, ncols), dtype=np.asarray(image).dtype)
    for i in range(nrows):
        for j in range(ncols):
            if(affinePre != None):
                dii = _apply_affine_2d_x0(i, j, affinePre) + \
                    displacement[i, j, 0]
                djj = _apply_affine_2d_x1(i, j, affinePre) + \
                    displacement[i, j, 1]
            else:
                dii = i + displacement[i, j, 0]
                djj = j + displacement[i, j, 1]
            if(affinePost != None):
                tmp0 = _apply_affine_2d_x0(dii, djj, affinePost)
                djj = _apply_affine_2d_x1(dii, djj, affinePost)
                dii = tmp0
            # no one is affected
            if((dii < 0) or (djj < 0) or (dii > nrVol - 1) or (djj > ncVol - 1)):
                continue
            # find the top left index and the interpolation coefficients
            ii = int(dii)
            jj = int(djj)
            # no one is affected
            if((ii < 0) or (jj < 0) or (ii >= nrVol) or (jj >= ncVol)):
                continue
            calpha = dii - ii  # by definition these factors are nonnegative
            cbeta = djj - jj
            alpha = 1 - calpha
            beta = 1 - cbeta
            if(alpha < calpha):
                ii += 1
            if(beta < cbeta):
                jj += 1
            # no one is affected
            if((ii < 0) or (jj < 0) or (ii >= nrVol) or (jj >= ncVol)):
                continue
            else:
                warped[i, j] = image[ii, jj]
    return warped


@cython.boundscheck(False)
@cython.wraparound(False)
def warp_image_affine_nn(number[:, :] image, int[:] refShape,
                         floating[:, :] affine=None):
    cdef int nrows = refShape[0]
    cdef int ncols = refShape[1]
    cdef int nrVol = image.shape[0]
    cdef int ncVol = image.shape[1]
    cdef floating dii, djj, tmp0
    cdef floating alpha, beta, calpha, cbeta
    cdef int i, j, ii, jj
    cdef number[:, :] warped = np.zeros((nrows, ncols), dtype=np.asarray(image).dtype)
    for i in range(nrows):
        for j in range(ncols):
            if(affine != None):
                dii = _apply_affine_2d_x0(i, j, affine)
                djj = _apply_affine_2d_x1(i, j, affine)
            else:
                dii = i
                djj = j
            # no one is affected
            if((dii < 0) or (djj < 0) or (dii > nrVol - 1) or (djj > ncVol - 1)):
                continue
            # find the top left index and the interpolation coefficients
            ii = int(dii)
            jj = int(djj)
            # no one is affected
            if((ii < 0) or (jj < 0) or (ii >= nrVol) or (jj >= ncVol)):
                continue
            calpha = dii - ii  # by definition these factors are nonnegative
            cbeta = djj - jj
            alpha = 1 - calpha
            beta = 1 - cbeta
            if(alpha < calpha):
                ii += 1
            if(beta < cbeta):
                jj += 1
            # no one is affected
            if((ii < 0) or (jj < 0) or (ii >= nrVol) or (jj >= ncVol)):
                continue
            else:
                warped[i, j] = image[ii, jj]
    return warped
