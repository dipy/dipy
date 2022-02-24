#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as cnp

from dipy.align.fused_types cimport floating, number
from dipy.core.interpolation cimport (_interpolate_scalar_2d,
                                      _interpolate_scalar_3d,
                                      _interpolate_vector_2d,
                                      _interpolate_vector_3d,
                                      _interpolate_scalar_nn_2d,
                                      _interpolate_scalar_nn_3d)


cdef extern from "dpy_math.h" nogil:
    double floor(double)
    double sqrt(double)
    double cos(double)
    double atan2(double, double)


def is_valid_affine(double[:, :] M, int dim):
    if M is None:
        return True
    if M.shape[0] < dim or M.shape[1] < dim + 1:
        return False
    if not np.all(np.isfinite(M)):
        return False
    return True


cdef void _compose_vector_fields_2d(floating[:, :, :] d1, floating[:, :, :] d2,
                                    double[:, :] premult_index,
                                    double[:, :] premult_disp,
                                    double time_scaling,
                                    floating[:, :, :] comp,
                                    double[:] stats) nogil:
    r"""Computes the composition of two 2D displacement fields

    Computes the composition of the two 2-D displacements d1 and d2. The
    evaluation of d2 at non-lattice points is computed using tri-linear
    interpolation. The actual composition is computed as:

    comp[i] = d1[i] + t * d2[ A * i + B * d1[i] ]

    where t = time_scaling, A = premult_index and B=premult_disp and i denotes
    the voxel coordinates of a voxel in d1's grid. Using this parameters it is
    possible to compose vector fields with arbitrary discretization: let R and
    S be the voxel-to-space transformation associated to d1 and d2,
    respectively then the composition at a voxel with coordinates i in d1's
    grid is given by:

    comp[i] = d1[i] + R*i + d2[Sinv*(R*i + d1[i])] - R*i

    (the R*i terms cancel each other) where Sinv = S^{-1}
    we can then define A = Sinv * R and B = Sinv to compute the composition
    using this function.

    Parameters
    ----------
    d1 : array, shape (R, C, 2)
        first displacement field to be applied. R, C are the number of rows
        and columns of the displacement field, respectively.
    d2 : array, shape (R', C', 2)
        second displacement field to be applied. R', C' are the number of rows
        and columns of the displacement field, respectively.
    premult_index : array, shape (3, 3)
        the matrix A in the explanation above
    premult_disp : array, shape (3, 3)
        the matrix B in the explanation above
    time_scaling : float
        this corresponds to the time scaling 't' in the above explanation
    comp : array, shape (R, C, 2), same dimension as d1
        on output, this array will contain the composition of the two fields
    stats : array, shape (3,)
        on output, this array will contain three statistics of the vector norms
        of the composition (maximum, mean, standard_deviation)

    Returns
    -------
    comp : array, shape (R, C, 2), same dimension as d1
        on output, this array will contain the composition of the two fields
    stats : array, shape (3,)
        on output, this array will contain three statistics of the vector norms
        of the composition (maximum, mean, standard_deviation)

    Notes
    -----
    If d1[r,c] lies outside the domain of d2, then comp[r,c] will contain a
    zero vector.

    Warning: it is possible to use the same array reference for d1 and comp to
    effectively update d1 to the composition of d1 and d2 because previously
    updated values from d1 are no longer used (this is done to save memory and
    time). However, using the same array for d2 and comp may not be the
    intended operation (see comment below).

    """
    cdef:
        cnp.npy_intp nr1 = d1.shape[0]
        cnp.npy_intp nc1 = d1.shape[1]
        cnp.npy_intp nr2 = d2.shape[0]
        cnp.npy_intp nc2 = d2.shape[1]
        int inside, cnt = 0
        double maxNorm = 0
        double meanNorm = 0
        double stdNorm = 0
        double nn
        cnp.npy_intp i, j
        double di, dj, dii, djj, diii, djjj

    for i in range(nr1):
        for j in range(nc1):

            # This is the only place we access d1[i, j]
            dii = d1[i, j, 0]
            djj = d1[i, j, 1]

            if premult_disp is None:
                di = dii
                dj = djj
            else:
                di = _apply_affine_2d_x0(dii, djj, 0, premult_disp)
                dj = _apply_affine_2d_x1(dii, djj, 0, premult_disp)

            if premult_index is None:
                diii = i
                djjj = j
            else:
                diii = _apply_affine_2d_x0(i, j, 1, premult_index)
                djjj = _apply_affine_2d_x1(i, j, 1, premult_index)

            diii += di
            djjj += dj

            # If d1 and comp are the same array, this will correctly update
            # d1[i,j], which will never be accessed again
            # If d2 and comp are the same array, then (diii, djjj) may be
            # in the neighborhood of a previously updated vector from d2,
            # which may be problematic
            inside = _interpolate_vector_2d[floating](d2, diii, djjj,
                                                      &comp[i, j, 0])

            if inside == 1:
                comp[i, j, 0] = time_scaling * comp[i, j, 0] + dii
                comp[i, j, 1] = time_scaling * comp[i, j, 1] + djj
                nn = comp[i, j, 0] ** 2 + comp[i, j, 1] ** 2
                meanNorm += nn
                stdNorm += nn * nn
                cnt += 1
                if(maxNorm < nn):
                    maxNorm = nn
            else:
                comp[i, j, 0] = 0
                comp[i, j, 1] = 0

    meanNorm /= cnt
    stats[0] = sqrt(maxNorm)
    stats[1] = sqrt(meanNorm)
    stats[2] = sqrt(stdNorm / cnt - meanNorm * meanNorm)


def compose_vector_fields_2d(floating[:, :, :] d1, floating[:, :, :] d2,
                             double[:, :] premult_index,
                             double[:, :] premult_disp,
                             double time_scaling,
                             floating[:, :, :] comp):
    r"""Computes the composition of two 2D displacement fields

    Computes the composition of the two 2-D displacements d1 and d2. The
    evaluation of d2 at non-lattice points is computed using tri-linear
    interpolation. The actual composition is computed as:

    comp[i] = d1[i] + t * d2[ A * i + B * d1[i] ]

    where t = time_scaling, A = premult_index and B=premult_disp and i denotes
    the voxel coordinates of a voxel in d1's grid. Using this parameters it is
    possible to compose vector fields with arbitrary discretizations: let R and
    S be the voxel-to-space transformation associated to d1 and d2,
    respectively then the composition at a voxel with coordinates i in d1's
    grid is given by:

    comp[i] = d1[i] + R*i + d2[Sinv*(R*i + d1[i])] - R*i

    (the R*i terms cancel each other) where Sinv = S^{-1}
    we can then define A = Sinv * R and B = Sinv to compute the composition
    using this function.

    Parameters
    ----------
    d1 : array, shape (R, C, 2)
        first displacement field to be applied. R, C are the number of rows
        and columns of the displacement field, respectively.
    d2 : array, shape (R', C', 2)
        second displacement field to be applied. R', C' are the number of rows
        and columns of the displacement field, respectively.
    premult_index : array, shape (3, 3)
        the matrix A in the explanation above
    premult_disp : array, shape (3, 3)
        the matrix B in the explanation above
    time_scaling : float
        this corresponds to the time scaling 't' in the above explanation
    comp : array, shape (R, C, 2)
        the buffer to write the composition to. If None, the buffer is created
        internally

    Returns
    -------
    comp : array, shape (R, C, 2), same dimension as d1
        on output, this array will contain the composition of the two fields
    stats : array, shape (3,)
        on output, this array will contain three statistics of the vector norms
        of the composition (maximum, mean, standard_deviation)
    """
    cdef:
        double[:] stats = np.zeros(shape=(3,), dtype=np.float64)

    if comp is None:
        comp = np.zeros_like(d1)

    if not is_valid_affine(premult_index, 2):
        raise ValueError("Invalid index multiplication matrix")
    if not is_valid_affine(premult_disp, 2):
        raise ValueError("Invalid displacement multiplication matrix")

    _compose_vector_fields_2d[floating](d1, d2, premult_index, premult_disp,
                                        time_scaling, comp, stats)
    return np.asarray(comp), np.asarray(stats)


cdef void _compose_vector_fields_3d(floating[:, :, :, :] d1,
                                    floating[:, :, :, :] d2,
                                    double[:, :] premult_index,
                                    double[:, :] premult_disp,
                                    double t,
                                    floating[:, :, :, :] comp,
                                    double[:] stats) nogil:
    r"""Computes the composition of two 3D displacement fields

    Computes the composition of the two 3-D displacements d1 and d2. The
    evaluation of d2 at non-lattice points is computed using tri-linear
    interpolation. The actual composition is computed as:

    comp[i] = d1[i] + t * d2[ A * i + B * d1[i] ]

    where t = time_scaling, A = premult_index and B=premult_disp and i denotes
    the voxel coordinates of a voxel in d1's grid. Using this parameters it is
    possible to compose vector fields with arbitrary discretization: let R and
    S be the voxel-to-space transformation associated to d1 and d2,
    respectively then the composition at a voxel with coordinates i in d1's
    grid is given by:

    comp[i] = d1[i] + R*i + d2[Sinv*(R*i + d1[i])] - R*i

    (the R*i terms cancel each other) where Sinv = S^{-1}
    we can then define A = Sinv * R and B = Sinv to compute the composition
    using this function.

    Parameters
    ----------
    d1 : array, shape (S, R, C, 3)
        first displacement field to be applied. S, R, C are the number of
        slices, rows and columns of the displacement field, respectively.
    d2 : array, shape (S', R', C', 3)
        second displacement field to be applied. R', C' are the number of rows
        and columns of the displacement field, respectively.
    premult_index : array, shape (4, 4)
        the matrix A in the explanation above
    premult_disp : array, shape (4, 4)
        the matrix B in the explanation above
    time_scaling : float
        this corresponds to the time scaling 't' in the above explanation
    comp : array, shape (S, R, C, 3), same dimension as d1
        on output, this array will contain the composition of the two fields
    stats : array, shape (3,)
        on output, this array will contain three statistics of the vector norms
        of the composition (maximum, mean, standard_deviation)

    Returns
    -------
    comp : array, shape (S, R, C, 3), same dimension as d1
        on output, this array will contain the composition of the two fields
    stats : array, shape (3,)
        on output, this array will contain three statistics of the vector norms
        of the composition (maximum, mean, standard_deviation)

    Notes
    -----
    If d1[s,r,c] lies outside the domain of d2, then comp[s,r,c] will contain
    a zero vector.

    Warning: it is possible to use the same array reference for d1 and comp to
    effectively update d1 to the composition of d1 and d2 because previously
    updated values from d1 are no longer used (this is done to save memory and
    time). However, using the same array for d2 and comp may not be the
    intended operation (see comment below).
    """
    cdef:
        cnp.npy_intp ns1 = d1.shape[0]
        cnp.npy_intp nr1 = d1.shape[1]
        cnp.npy_intp nc1 = d1.shape[2]
        cnp.npy_intp ns2 = d2.shape[0]
        cnp.npy_intp nr2 = d2.shape[1]
        cnp.npy_intp nc2 = d2.shape[2]
        int inside, cnt = 0
        double maxNorm = 0
        double meanNorm = 0
        double stdNorm = 0
        double nn
        cnp.npy_intp i, j, k
        double di, dj, dk, dii, djj, dkk, diii, djjj, dkkk
    for k in range(ns1):
        for i in range(nr1):
            for j in range(nc1):

                # This is the only place we access d1[k, i, j]
                dkk = d1[k, i, j, 0]
                dii = d1[k, i, j, 1]
                djj = d1[k, i, j, 2]

                if premult_disp is None:
                    dk = dkk
                    di = dii
                    dj = djj
                else:
                    dk = _apply_affine_3d_x0(dkk, dii, djj, 0, premult_disp)
                    di = _apply_affine_3d_x1(dkk, dii, djj, 0, premult_disp)
                    dj = _apply_affine_3d_x2(dkk, dii, djj, 0, premult_disp)

                if premult_index is None:
                    dkkk = k
                    diii = i
                    djjj = j
                else:
                    dkkk = _apply_affine_3d_x0(k, i, j, 1, premult_index)
                    diii = _apply_affine_3d_x1(k, i, j, 1, premult_index)
                    djjj = _apply_affine_3d_x2(k, i, j, 1, premult_index)

                dkkk += dk
                diii += di
                djjj += dj

                # If d1 and comp are the same array, this will correctly update
                # d1[k,i,j], which will never be accessed again
                # If d2 and comp are the same array, then (dkkk, diii, djjj)
                # may be in the neighborhood of a previously updated vector
                # from d2, which may be problematic
                inside = _interpolate_vector_3d[floating](d2, dkkk, diii, djjj,
                                                          &comp[k, i, j, 0])

                if inside == 1:
                    comp[k, i, j, 0] = t * comp[k, i, j, 0] + dkk
                    comp[k, i, j, 1] = t * comp[k, i, j, 1] + dii
                    comp[k, i, j, 2] = t * comp[k, i, j, 2] + djj
                    nn = (comp[k, i, j, 0] ** 2 + comp[k, i, j, 1] ** 2 +
                          comp[k, i, j, 2]**2)
                    meanNorm += nn
                    stdNorm += nn * nn
                    cnt += 1
                    if(maxNorm < nn):
                        maxNorm = nn
                else:
                    comp[k, i, j, 0] = 0
                    comp[k, i, j, 1] = 0
                    comp[k, i, j, 2] = 0
    meanNorm /= cnt
    stats[0] = sqrt(maxNorm)
    stats[1] = sqrt(meanNorm)
    stats[2] = sqrt(stdNorm / cnt - meanNorm * meanNorm)


def compose_vector_fields_3d(floating[:, :, :, :] d1, floating[:, :, :, :] d2,
                             double[:, :] premult_index,
                             double[:, :] premult_disp,
                             double time_scaling,
                             floating[:, :, :, :] comp):
    r"""Computes the composition of two 3D displacement fields

    Computes the composition of the two 3-D displacements d1 and d2. The
    evaluation of d2 at non-lattice points is computed using tri-linear
    interpolation. The actual composition is computed as:

    comp[i] = d1[i] + t * d2[ A * i + B * d1[i] ]

    where t = time_scaling, A = premult_index and B=premult_disp and i denotes
    the voxel coordinates of a voxel in d1's grid. Using this parameters it is
    possible to compose vector fields with arbitrary discretization: let R and
    S be the voxel-to-space transformation associated to d1 and d2,
    respectively then the composition at a voxel with coordinates i in d1's
    grid is given by:

    comp[i] = d1[i] + R*i + d2[Sinv*(R*i + d1[i])] - R*i

    (the R*i terms cancel each other) where Sinv = S^{-1}
    we can then define A = Sinv * R and B = Sinv to compute the composition
    using this function.

    Parameters
    ----------
    d1 : array, shape (S, R, C, 3)
        first displacement field to be applied. S, R, C are the number of
        slices, rows and columns of the displacement field, respectively.
    d2 : array, shape (S', R', C', 3)
        second displacement field to be applied. R', C' are the number of rows
        and columns of the displacement field, respectively.
    premult_index : array, shape (4, 4)
        the matrix A in the explanation above
    premult_disp : array, shape (4, 4)
        the matrix B in the explanation above
    time_scaling : float
        this corresponds to the time scaling 't' in the above explanation
    comp : array, shape (S, R, C, 3), same dimension as d1
        the buffer to write the composition to. If None, the buffer will be
        created internally

    Returns
    -------
    comp : array, shape (S, R, C, 3), same dimension as d1
        on output, this array will contain the composition of the two fields
    stats : array, shape (3,)
        on output, this array will contain three statistics of the vector norms
        of the composition (maximum, mean, standard_deviation)

    Notes
    -----
    If d1[s,r,c] lies outside the domain of d2, then comp[s,r,c] will contain
    a zero vector.
    """
    cdef:
        double[:] stats = np.zeros(shape=(3,), dtype=np.float64)

    if comp is None:
        comp = np.zeros_like(d1)

    if not is_valid_affine(premult_index, 3):
        raise ValueError("Invalid index pre-multiplication matrix")
    if not is_valid_affine(premult_disp, 3):
        raise ValueError("Invalid displacement pre-multiplication matrix")

    _compose_vector_fields_3d[floating](d1, d2, premult_index, premult_disp,
                                        time_scaling, comp, stats)
    return np.asarray(comp), np.asarray(stats)


def invert_vector_field_fixed_point_2d(floating[:, :, :] d,
                                       double[:, :] d_world2grid,
                                       double[:] spacing,
                                       int max_iter, double tolerance,
                                       floating[:, :, :] start=None):
    r"""Computes the inverse of a 2D displacement fields

    Computes the inverse of the given 2-D displacement field d using the
    fixed-point algorithm [1].

    [1] Chen, M., Lu, W., Chen, Q., Ruchala, K. J., & Olivera, G. H. (2008).
        A simple fixed-point approach to invert a deformation field.
        Medical Physics, 35(1), 81. doi:10.1118/1.2816107

    Parameters
    ----------
    d : array, shape (R, C, 2)
        the 2-D displacement field to be inverted
    d_world2grid : array, shape (3, 3)
        the space-to-grid transformation associated to the displacement field
        d (transforming physical space coordinates to voxel coordinates of the
        displacement field grid)
    spacing :array, shape (2,)
        the spacing between voxels (voxel size along each axis)
    max_iter : int
        maximum number of iterations to be performed
    tolerance : float
        maximum tolerated inversion error
    start : array, shape (R, C)
        an approximation to the inverse displacement field (if no approximation
        is available, None can be provided and the start displacement field
        will be zero)

    Returns
    -------
    p : array, shape (R, C, 2)
        the inverse displacement field

    Notes
    -----
    We assume that the displacement field is an endomorphism so that the shape
    and voxel-to-space transformation of the inverse field's discretization is
    the same as those of the input displacement field. The 'inversion error' at
    iteration t is defined as the mean norm of the displacement vectors of the
    input displacement field composed with the inverse at iteration t.
    """
    cdef:
        cnp.npy_intp nr = d.shape[0]
        cnp.npy_intp nc = d.shape[1]
        int iter_count, current, flag
        double difmag, mag, maxlen, step_factor
        double epsilon
        double error = 1 + tolerance
        double di, dj, dii, djj
        double sr = spacing[0], sc = spacing[1]

    ftype = np.asarray(d).dtype
    cdef:
        double[:] stats = np.zeros(shape=(2,), dtype=np.float64)
        double[:] substats = np.empty(shape=(3,), dtype=np.float64)
        double[:, :] norms = np.zeros(shape=(nr, nc), dtype=np.float64)
        floating[:, :, :] p = np.zeros(shape=(nr, nc, 2), dtype=ftype)
        floating[:, :, :] q = np.zeros(shape=(nr, nc, 2), dtype=ftype)

    if not is_valid_affine(d_world2grid, 2):
        raise ValueError("Invalid world-to-image transform")

    if start is not None:
        p[...] = start

    with nogil:
        iter_count = 0
        while (iter_count < max_iter) and (tolerance < error):
            if iter_count == 0:
                epsilon = 0.75
            else:
                epsilon = 0.5
            _compose_vector_fields_2d[floating](p, d, None, d_world2grid,
                                                1.0, q, substats)
            difmag = 0
            error = 0
            for i in range(nr):
                for j in range(nc):
                    mag = sqrt((q[i, j, 0]/sr) ** 2 + (q[i, j, 1]/sc) ** 2)
                    norms[i, j] = mag
                    error += mag
                    if(difmag < mag):
                        difmag = mag
            maxlen = difmag * epsilon
            for i in range(nr):
                for j in range(nc):
                    if norms[i, j] > maxlen:
                        step_factor = epsilon * maxlen / norms[i, j]
                    else:
                        step_factor = epsilon
                    p[i, j, 0] = p[i, j, 0] - step_factor * q[i, j, 0]
                    p[i, j, 1] = p[i, j, 1] - step_factor * q[i, j, 1]
            error /= (nr * nc)
            iter_count += 1
        stats[0] = substats[1]
        stats[1] = iter_count
    return np.asarray(p)


def invert_vector_field_fixed_point_3d(floating[:, :, :, :] d,
                                       double[:, :] d_world2grid,
                                       double[:] spacing,
                                       int max_iter, double tol,
                                       floating[:, :, :, :] start=None):
    r"""Computes the inverse of a 3D displacement fields

    Computes the inverse of the given 3-D displacement field d using the
    fixed-point algorithm [1].

    [1] Chen, M., Lu, W., Chen, Q., Ruchala, K. J., & Olivera, G. H. (2008).
        A simple fixed-point approach to invert a deformation field.
        Medical Physics, 35(1), 81. doi:10.1118/1.2816107

    Parameters
    ----------
    d : array, shape (S, R, C, 3)
        the 3-D displacement field to be inverted
    d_world2grid : array, shape (4, 4)
        the space-to-grid transformation associated to the displacement field
        d (transforming physical space coordinates to voxel coordinates of the
        displacement field grid)
    spacing :array, shape (3,)
        the spacing between voxels (voxel size along each axis)
    max_iter : int
        maximum number of iterations to be performed
    tol : float
        maximum tolerated inversion error
    start : array, shape (S, R, C)
        an approximation to the inverse displacement field (if no approximation
        is available, None can be provided and the start displacement field
        will be zero)

    Returns
    -------
    p : array, shape (S, R, C, 3)
        the inverse displacement field

    Notes
    -----
    We assume that the displacement field is an endomorphism so that the shape
    and voxel-to-space transformation of the inverse field's discretization is
    the same as those of the input displacement field. The 'inversion error' at
    iteration t is defined as the mean norm of the displacement vectors of the
    input displacement field composed with the inverse at iteration t.
    """
    cdef:
        cnp.npy_intp ns = d.shape[0]
        cnp.npy_intp nr = d.shape[1]
        cnp.npy_intp nc = d.shape[2]
        int iter_count, current
        double dkk, dii, djj, dk, di, dj
        double difmag, mag, maxlen, step_factor
        double epsilon = 0.5
        double error = 1 + tol
        double ss = spacing[0], sr = spacing[1], sc = spacing[2]

    ftype = np.asarray(d).dtype
    cdef:
        double[:] stats = np.zeros(shape=(2,), dtype=np.float64)
        double[:] substats = np.zeros(shape=(3,), dtype=np.float64)
        double[:, :, :] norms = np.zeros(shape=(ns, nr, nc), dtype=np.float64)
        floating[:, :, :, :] p = np.zeros(shape=(ns, nr, nc, 3), dtype=ftype)
        floating[:, :, :, :] q = np.zeros(shape=(ns, nr, nc, 3), dtype=ftype)

    if not is_valid_affine(d_world2grid, 3):
        raise ValueError("Invalid world-to-image transform")

    if start is not None:
        p[...] = start

    with nogil:
        iter_count = 0
        difmag = 1
        while (0.1 < difmag) and (iter_count < max_iter) and (tol < error):
            if iter_count == 0:
                epsilon = 0.75
            else:
                epsilon = 0.5
            _compose_vector_fields_3d[floating](p, d, None, d_world2grid,
                                                1.0, q, substats)
            difmag = 0
            error = 0
            for k in range(ns):
                for i in range(nr):
                    for j in range(nc):
                        mag = sqrt((q[k, i, j, 0]/ss) ** 2 +
                                   (q[k, i, j, 1]/sr) ** 2 +
                                   (q[k, i, j, 2]/sc) ** 2)
                        norms[k, i, j] = mag
                        error += mag
                        if(difmag < mag):
                            difmag = mag
            maxlen = difmag*epsilon
            for k in range(ns):
                for i in range(nr):
                    for j in range(nc):
                        if norms[k, i, j] > maxlen:
                            step_factor = epsilon * maxlen / norms[k, i, j]
                        else:
                            step_factor = epsilon
                        p[k, i, j, 0] = (p[k, i, j, 0] -
                                         step_factor * q[k, i, j, 0])
                        p[k, i, j, 1] = (p[k, i, j, 1] -
                                         step_factor * q[k, i, j, 1])
                        p[k, i, j, 2] = (p[k, i, j, 2] -
                                         step_factor * q[k, i, j, 2])
            error /= (ns * nr * nc)
            iter_count += 1
        stats[0] = error
        stats[1] = iter_count
    return np.asarray(p)


def simplify_warp_function_2d(floating[:, :, :] d,
                              double[:, :] affine_idx_in,
                              double[:, :] affine_idx_out,
                              double[:, :] affine_disp,
                              int[:] out_shape):
    r"""
    Simplifies a nonlinear warping function combined with an affine transform

    Modifies the given deformation field by incorporating into it a
    an affine transformation and voxel-to-space transforms associated to
    the discretization of its domain and codomain.
    The resulting transformation may be regarded as operating on the
    image spaces given by the domain and codomain discretization.
    More precisely, the resulting transform is of the form:

    (1) T[i] = W * d[U * i] + V * i

    Where U = affine_idx_in, V = affine_idx_out, W = affine_disp.

    Parameters
    ----------
    d : array, shape (R', C', 2)
        the non-linear part of the transformation (displacement field)
    affine_idx_in : array, shape (3, 3)
        the matrix U in eq. (1) above
    affine_idx_out : array, shape (3, 3)
        the matrix V in eq. (1) above
    affine_disp : array, shape (3, 3)
        the matrix W in eq. (1) above
    out_shape : array, shape (2,)
        the number of rows and columns of the sampling grid

    Returns
    -------
    out : array, shape = out_shape
        the deformation field `out` associated with `T` in eq. (1) such that:
        T[i] = i + out[i]

    Notes
    -----
    Both the direct and inverse transforms of a DiffeomorphicMap can be written
    in this form:

    Direct:  Let D be the voxel-to-space transform of the domain's
             discretization, P be the pre-align matrix, Rinv the space-to-voxel
             transform of the reference grid (the grid the displacement field
             is defined on) and Cinv be the space-to-voxel transform of the
             codomain's discretization. Then, for each i in the domain's grid,
             the direct transform is given by

             (2) T[i] = Cinv * d[Rinv * P * D * i] + Cinv * P * D * i

             and we identify U = Rinv * P * D, V = Cinv * P * D, W = Cinv

    Inverse: Let C be the voxel-to-space transform of the codomain's
             discretization, Pinv be the inverse of the pre-align matrix, Rinv
             the space-to-voxel transform of the reference grid (the grid the
             displacement field is defined on) and Dinv be the space-to-voxel
             transform of the domain's discretization. Then, for each j in the
             codomain's grid, the inverse transform is given by

             (3) Tinv[j] = Dinv * Pinv * d[Rinv * C * j] + Dinv * Pinv * C * j

             and we identify U = Rinv * C, V = Dinv * Pinv * C, W = Dinv * Pinv
    """
    cdef:
        cnp.npy_intp nrows = out_shape[0]
        cnp.npy_intp ncols = out_shape[1]
        cnp.npy_intp i, j
        double di, dj, dii, djj
        floating[:] tmp = np.zeros((2,), dtype=np.asarray(d).dtype)
        floating[:, :, :] out = np.zeros(shape=(nrows, ncols, 2),
                                         dtype=np.asarray(d).dtype)

    if not is_valid_affine(affine_idx_in, 2):
        raise ValueError("Invalid inner index multiplication matrix")
    if not is_valid_affine(affine_idx_out, 2):
        raise ValueError("Invalid outer index multiplication matrix")
    if not is_valid_affine(affine_disp, 2):
        raise ValueError("Invalid displacement multiplication matrix")

    with nogil:

        for i in range(nrows):
            for j in range(ncols):
                # Apply inner index pre-multiplication
                if affine_idx_in is None:
                    dii = d[i, j, 0]
                    djj = d[i, j, 1]
                else:
                    di = _apply_affine_2d_x0(
                        i, j, 1, affine_idx_in)
                    dj = _apply_affine_2d_x1(
                        i, j, 1, affine_idx_in)
                    _interpolate_vector_2d[floating](d, di, dj, &tmp[0])
                    dii = tmp[0]
                    djj = tmp[1]

                # Apply displacement multiplication
                if affine_disp is not None:
                    di = _apply_affine_2d_x0(
                        dii, djj, 0, affine_disp)
                    dj = _apply_affine_2d_x1(
                        dii, djj, 0, affine_disp)
                else:
                    di = dii
                    dj = djj

                # Apply outer index multiplication and add the displacements
                if affine_idx_out is not None:
                    out[i, j, 0] = di + _apply_affine_2d_x0(i, j, 1,
                                                            affine_idx_out) - i
                    out[i, j, 1] = dj + _apply_affine_2d_x1(i, j, 1,
                                                            affine_idx_out) - j
                else:
                    out[i, j, 0] = di
                    out[i, j, 1] = dj
    return np.asarray(out)


def simplify_warp_function_3d(floating[:, :, :, :] d,
                              double[:, :] affine_idx_in,
                              double[:, :] affine_idx_out,
                              double[:, :] affine_disp,
                              int[:] out_shape):
    r"""
    Simplifies a nonlinear warping function combined with an affine transform

    Modifies the given deformation field by incorporating into it
    an affine transformation and voxel-to-space transforms associated with
    the discretization of its domain and codomain.

    The resulting transformation may be regarded as operating on the
    image spaces given by the domain and codomain discretization.
    More precisely, the resulting transform is of the form:

    (1) T[i] = W * d[U * i] + V * i

    Where U = affine_idx_in, V = affine_idx_out, W = affine_disp.

    Parameters
    ----------
    d : array, shape (S', R', C', 3)
        the non-linear part of the transformation (displacement field)
    affine_idx_in : array, shape (4, 4)
        the matrix U in eq. (1) above
    affine_idx_out : array, shape (4, 4)
        the matrix V in eq. (1) above
    affine_disp : array, shape (4, 4)
        the matrix W in eq. (1) above
    out_shape : array, shape (3,)
        the number of slices, rows and columns of the sampling grid

    Returns
    -------
    out : array, shape = out_shape
        the deformation field `out` associated with `T` in eq. (1) such that:
        T[i] = i + out[i]

    Notes
    -----
    Both the direct and inverse transforms of a DiffeomorphicMap can be written
    in this form:

    Direct:  Let D be the voxel-to-space transform of the domain's
             discretization, P be the pre-align matrix, Rinv the space-to-voxel
             transform of the reference grid (the grid the displacement field
             is defined on) and Cinv be the space-to-voxel transform of the
             codomain's discretization. Then, for each i in the domain's grid,
             the direct transform is given by

             (2) T[i] = Cinv * d[Rinv * P * D * i] + Cinv * P * D * i

             and we identify U = Rinv * P * D, V = Cinv * P * D, W = Cinv

    Inverse: Let C be the voxel-to-space transform of the codomain's
             discretization, Pinv be the inverse of the pre-align matrix, Rinv
             the space-to-voxel transform of the reference grid (the grid the
             displacement field is defined on) and Dinv be the space-to-voxel
             transform of the domain's discretization. Then, for each j in the
             codomain's grid, the inverse transform is given by

             (3) Tinv[j] = Dinv * Pinv * d[Rinv * C * j] + Dinv * Pinv * C * j

             and we identify U = Rinv * C, V = Dinv * Pinv * C, W = Dinv * Pinv

    """
    cdef:
        cnp.npy_intp nslices = out_shape[0]
        cnp.npy_intp nrows = out_shape[1]
        cnp.npy_intp ncols = out_shape[2]
        cnp.npy_intp i, j, k, inside
        double di, dj, dk, dii, djj, dkk
        floating[:] tmp = np.zeros((3,), dtype=np.asarray(d).dtype)
        floating[:, :, :, :] out = np.zeros(shape=(nslices, nrows, ncols, 3),
                                            dtype=np.asarray(d).dtype)

    if not is_valid_affine(affine_idx_in, 3):
        raise ValueError("Invalid inner index multiplication matrix")
    if not is_valid_affine(affine_idx_out, 3):
        raise ValueError("Invalid outer index multiplication matrix")
    if not is_valid_affine(affine_disp, 3):
        raise ValueError("Invalid displacement multiplication matrix")

    with nogil:

        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if affine_idx_in is None:
                        dkk = d[k, i, j, 0]
                        dii = d[k, i, j, 1]
                        djj = d[k, i, j, 2]
                    else:
                        dk = _apply_affine_3d_x0(
                            k, i, j, 1, affine_idx_in)
                        di = _apply_affine_3d_x1(
                            k, i, j, 1, affine_idx_in)
                        dj = _apply_affine_3d_x2(
                            k, i, j, 1, affine_idx_in)
                        inside = _interpolate_vector_3d[floating](d, dk, di,
                                                                  dj, &tmp[0])
                        dkk = tmp[0]
                        dii = tmp[1]
                        djj = tmp[2]

                    if affine_disp is not None:
                        dk = _apply_affine_3d_x0(
                            dkk, dii, djj, 0, affine_disp)
                        di = _apply_affine_3d_x1(
                            dkk, dii, djj, 0, affine_disp)
                        dj = _apply_affine_3d_x2(
                            dkk, dii, djj, 0, affine_disp)
                    else:
                        dk = dkk
                        di = dii
                        dj = djj

                    if affine_idx_out is not None:
                        out[k, i, j, 0] = dk +\
                            _apply_affine_3d_x0(k, i, j, 1, affine_idx_out) - k
                        out[k, i, j, 1] = di +\
                            _apply_affine_3d_x1(k, i, j, 1, affine_idx_out) - i
                        out[k, i, j, 2] = dj +\
                            _apply_affine_3d_x2(k, i, j, 1, affine_idx_out) - j
                    else:
                        out[k, i, j, 0] = dk
                        out[k, i, j, 1] = di
                        out[k, i, j, 2] = dj
    return np.asarray(out)


def reorient_vector_field_2d(floating[:, :, :] d,
                             double[:, :] affine):
    r"""Linearly transforms all vectors of a 2D displacement field

    Modifies the input displacement field by multiplying each displacement
    vector by the given matrix. Note that the elements of the displacement
    field are vectors, not points, so their last homogeneous coordinate is
    zero, not one, and therefore the translation component of the affine
    transform will not have any effect on them.

    Parameters
    ----------
    d : array, shape (R, C, 2)
        the displacement field to be re-oriented
    affine: array, shape (3, 3)
        the matrix to be applied
    """
    cdef:
        cnp.npy_intp nrows = d.shape[0]
        cnp.npy_intp ncols = d.shape[1]
        cnp.npy_intp i, j
        double di, dj

    if not is_valid_affine(affine, 2):
        raise ValueError("Invalid affine transform matrix")

    if affine is None:
        return

    with nogil:
        for i in range(nrows):
            for j in range(ncols):
                di = d[i, j, 0]
                dj = d[i, j, 1]
                d[i, j, 0] = _apply_affine_2d_x0(di, dj, 0, affine)
                d[i, j, 1] = _apply_affine_2d_x1(di, dj, 0, affine)


def reorient_vector_field_3d(floating[:, :, :, :] d,
                             double[:, :] affine):
    r"""Linearly transforms all vectors of a 3D displacement field

    Modifies the input displacement field by multiplying each displacement
    vector by the given matrix. Note that the elements of the displacement
    field are vectors, not points, so their last homogeneous coordinate is
    zero, not one, and therefore the translation component of the affine
    transform will not have any effect on them.

    Parameters
    ----------
    d : array, shape (S, R, C, 3)
        the displacement field to be re-oriented
    affine : array, shape (4, 4)
        the matrix to be applied
    """
    cdef:
        cnp.npy_intp nslices = d.shape[0]
        cnp.npy_intp nrows = d.shape[1]
        cnp.npy_intp ncols = d.shape[2]
        cnp.npy_intp i, j, k
        double di, dj, dk

    if not is_valid_affine(affine, 3):
        raise ValueError("Invalid affine transform matrix")

    if affine is None:
        return

    with nogil:
        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    dk = d[k, i, j, 0]
                    di = d[k, i, j, 1]
                    dj = d[k, i, j, 2]
                    d[k, i, j, 0] = _apply_affine_3d_x0(dk, di, dj, 0, affine)
                    d[k, i, j, 1] = _apply_affine_3d_x1(dk, di, dj, 0, affine)
                    d[k, i, j, 2] = _apply_affine_3d_x2(dk, di, dj, 0, affine)


def downsample_scalar_field_3d(floating[:, :, :] field):
    r"""Down-samples the input volume by a factor of 2

    Down-samples the input volume by a factor of 2. The value at each voxel
    of the resulting volume is the average of its surrounding voxels in the
    original volume.

    Parameters
    ----------
    field : array, shape (S, R, C)
        the volume to be down-sampled

    Returns
    -------
    down : array, shape (S', R', C')
        the down-sampled displacement field, where S' = ceil(S/2),
        R'= ceil(R/2), C'=ceil(C/2)
    """
    ftype = np.asarray(field).dtype
    cdef:
        cnp.npy_intp ns = field.shape[0]
        cnp.npy_intp nr = field.shape[1]
        cnp.npy_intp nc = field.shape[2]
        cnp.npy_intp nns = (ns + 1) // 2
        cnp.npy_intp nnr = (nr + 1) // 2
        cnp.npy_intp nnc = (nc + 1) // 2
        cnp.npy_intp i, j, k, ii, jj, kk
        floating[:, :, :] down = np.zeros((nns, nnr, nnc), dtype=ftype)
        int[:, :, :] cnt = np.zeros((nns, nnr, nnc), dtype=np.int32)

    with nogil:
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
    return np.asarray(down)


def downsample_displacement_field_3d(floating[:, :, :, :] field):
    r"""Down-samples the input 3D vector field by a factor of 2

    Down-samples the input vector field by a factor of 2. This operation
    is equivalent to dividing the input image into 2x2x2 cubes and averaging
    the 8 vectors. The resulting field consists of these average vectors.

    Parameters
    ----------
    field : array, shape (S, R, C)
        the vector field to be down-sampled

    Returns
    -------
    down : array, shape (S', R', C')
        the down-sampled displacement field, where S' = ceil(S/2),
        R'= ceil(R/2), C'=ceil(C/2)
    """
    ftype = np.asarray(field).dtype
    cdef:
        cnp.npy_intp ns = field.shape[0]
        cnp.npy_intp nr = field.shape[1]
        cnp.npy_intp nc = field.shape[2]
        cnp.npy_intp nns = (ns + 1) // 2
        cnp.npy_intp nnr = (nr + 1) // 2
        cnp.npy_intp nnc = (nc + 1) // 2
        cnp.npy_intp i, j, k, ii, jj, kk
        floating[:, :, :, :] down = np.zeros((nns, nnr, nnc, 3), dtype=ftype)
        int[:, :, :] cnt = np.zeros((nns, nnr, nnc), dtype=np.int32)

    with nogil:

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
    return np.asarray(down)


def downsample_scalar_field_2d(floating[:, :] field):
    r"""Down-samples the input 2D image by a factor of 2

    Down-samples the input image by a factor of 2. The value at each pixel
    of the resulting image is the average of its surrounding pixels in the
    original image.

    Parameters
    ----------
    field : array, shape (R, C)
        the image to be down-sampled

    Returns
    -------
    down : array, shape (R', C')
        the down-sampled displacement field, where R'= ceil(R/2), C'=ceil(C/2)
    """
    ftype = np.asarray(field).dtype
    cdef:
        cnp.npy_intp nr = field.shape[0]
        cnp.npy_intp nc = field.shape[1]
        cnp.npy_intp nnr = (nr + 1) // 2
        cnp.npy_intp nnc = (nc + 1) // 2
        cnp.npy_intp i, j, ii, jj
        floating[:, :] down = np.zeros(shape=(nnr, nnc), dtype=ftype)
        int[:, :] cnt = np.zeros(shape=(nnr, nnc), dtype=np.int32)
    with nogil:

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
    return np.asarray(down)


def downsample_displacement_field_2d(floating[:, :, :] field):
    r"""Down-samples the 2D input vector field by a factor of 2
    Down-samples the input vector field by a factor of 2. The value at each
    pixel of the resulting field is the average of its surrounding pixels in
    the original field.

    Parameters
    ----------
    field : array, shape (R, C)
        the vector field to be down-sampled

    Returns
    -------
    down : array, shape (R', C')
        the down-sampled displacement field, where R'= ceil(R/2), C'=ceil(C/2),
    """
    ftype = np.asarray(field).dtype
    cdef:
        cnp.npy_intp nr = field.shape[0]
        cnp.npy_intp nc = field.shape[1]
        cnp.npy_intp nnr = (nr + 1) // 2
        cnp.npy_intp nnc = (nc + 1) // 2
        cnp.npy_intp i, j, ii, jj
        floating[:, :, :] down = np.zeros((nnr, nnc, 2), dtype=ftype)
        int[:, :] cnt = np.zeros((nnr, nnc), dtype=np.int32)

    with nogil:

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
    return np.asarray(down)


def warp_3d(floating[:, :, :] volume, floating[:, :, :, :] d1,
            double[:, :] affine_idx_in=None,
            double[:, :] affine_idx_out=None,
            double[:, :] affine_disp=None,
            int[:] out_shape=None):
    r"""Warps a 3D volume using trilinear interpolation

    Deforms the input volume under the given transformation. The warped volume
    is computed using tri-linear interpolation and is given by:

    (1) warped[i] = volume[ C * d1[A*i] + B*i ]

    where A = affine_idx_in, B = affine_idx_out, C = affine_disp and i denotes
    the discrete coordinates of a voxel in the sampling grid of
    shape = out_shape.

    Parameters
    ----------
    volume : array, shape (S, R, C)
        the input volume to be transformed
    d1 : array, shape (S', R', C', 3)
        the displacement field driving the transformation
    affine_idx_in : array, shape (4, 4)
        the matrix A in eq. (1) above
    affine_idx_out : array, shape (4, 4)
        the matrix B in eq. (1) above
    affine_disp : array, shape (4, 4)
        the matrix C in eq. (1) above
    out_shape : array, shape (3,)
        the number of slices, rows and columns of the sampling grid

    Returns
    -------
    warped : array, shape = out_shape
        the transformed volume

    Notes
    -----
    To illustrate the use of this function, consider a displacement field d1
    with grid-to-space transformation R, a volume with grid-to-space
    transformation T and let's say we want to sample the warped volume on a
    grid with grid-to-space transformation S (sampling grid). For each voxel
    in the sampling grid with discrete coordinates i, the warped volume is
    given by:

    (2) warped[i] = volume[Tinv * ( d1[Rinv * S * i] + S * i ) ]

    where Tinv = T^{-1} and Rinv = R^{-1}. By identifying A = Rinv * S,
    B = Tinv * S, C = Tinv we can use this function to efficiently warp the
    input image.
    """
    cdef:
        cnp.npy_intp nslices = volume.shape[0]
        cnp.npy_intp nrows = volume.shape[1]
        cnp.npy_intp ncols = volume.shape[2]
        cnp.npy_intp nsVol = volume.shape[0]
        cnp.npy_intp nrVol = volume.shape[1]
        cnp.npy_intp ncVol = volume.shape[2]
        cnp.npy_intp i, j, k
        int inside
        double dkk, dii, djj, dk, di, dj

    if not is_valid_affine(affine_idx_in, 3):
        raise ValueError("Invalid inner index multiplication matrix")
    if not is_valid_affine(affine_idx_out, 3):
        raise ValueError("Invalid outer index multiplication matrix")
    if not is_valid_affine(affine_disp, 3):
        raise ValueError("Invalid displacement multiplication matrix")

    if out_shape is not None:
        nslices = out_shape[0]
        nrows = out_shape[1]
        ncols = out_shape[2]
    elif d1 is not None:
        nslices = d1.shape[0]
        nrows = d1.shape[1]
        ncols = d1.shape[2]

    cdef floating[:, :, :] warped = np.zeros(shape=(nslices, nrows, ncols),
                                             dtype=np.asarray(volume).dtype)
    cdef floating[:] tmp = np.zeros(shape=(3,), dtype = np.asarray(d1).dtype)

    with nogil:

        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if affine_idx_in is None:
                        dkk = d1[k, i, j, 0]
                        dii = d1[k, i, j, 1]
                        djj = d1[k, i, j, 2]
                    else:
                        dk = _apply_affine_3d_x0(
                            k, i, j, 1, affine_idx_in)
                        di = _apply_affine_3d_x1(
                            k, i, j, 1, affine_idx_in)
                        dj = _apply_affine_3d_x2(
                            k, i, j, 1, affine_idx_in)
                        inside = _interpolate_vector_3d[floating](d1, dk, di,
                                                                  dj, &tmp[0])
                        dkk = tmp[0]
                        dii = tmp[1]
                        djj = tmp[2]

                    if affine_disp is not None:
                        dk = _apply_affine_3d_x0(
                            dkk, dii, djj, 0, affine_disp)
                        di = _apply_affine_3d_x1(
                            dkk, dii, djj, 0, affine_disp)
                        dj = _apply_affine_3d_x2(
                            dkk, dii, djj, 0, affine_disp)
                    else:
                        dk = dkk
                        di = dii
                        dj = djj

                    if affine_idx_out is not None:
                        dkk = dk + _apply_affine_3d_x0(k, i, j, 1,
                                                       affine_idx_out)
                        dii = di + _apply_affine_3d_x1(k, i, j, 1,
                                                       affine_idx_out)
                        djj = dj + _apply_affine_3d_x2(k, i, j, 1,
                                                       affine_idx_out)
                    else:
                        dkk = dk + k
                        dii = di + i
                        djj = dj + j

                    inside = _interpolate_scalar_3d[floating](volume, dkk,
                                                              dii, djj,
                                                              &warped[k,i,j])
    return np.asarray(warped)


def transform_3d_affine(floating[:, :, :] volume, int[:] ref_shape,
                        double[:, :] affine):
    r"""Transforms a 3D volume by an affine transform with trilinear interp.

    Deforms the input volume under the given affine transformation using
    tri-linear interpolation. The shape of the resulting transformation
    is given by ref_shape. If the affine matrix is None, it is taken as the
    identity.

    Parameters
    ----------
    volume : array, shape (S, R, C)
        the input volume to be transformed
    ref_shape : array, shape (3,)
        the shape of the resulting volume
    affine : array, shape (4, 4)
        the affine transform to be applied

    Returns
    -------
    out : array, shape (S', R', C')
        the transformed volume

    Notes
    -----
    The reason it is necessary to provide the intended shape of the resulting
    volume is because the affine transformation is defined on all R^{3}
    but we must sample a finite lattice. Also the resulting shape may not be
    necessarily equal to the input shape, unless we are interested on
    endomorphisms only and not general diffeomorphisms.
    """
    cdef:
        cnp.npy_intp nslices = ref_shape[0]
        cnp.npy_intp nrows = ref_shape[1]
        cnp.npy_intp ncols = ref_shape[2]
        cnp.npy_intp nsVol = volume.shape[0]
        cnp.npy_intp nrVol = volume.shape[1]
        cnp.npy_intp ncVol = volume.shape[2]
        cnp.npy_intp i, j, k, ii, jj, kk
        int inside
        double dkk, dii, djj, tmp0, tmp1
        double alpha, beta, gamma, calpha, cbeta, cgamma
        floating[:, :, :] out = np.zeros(shape=(nslices, nrows, ncols),
                                         dtype=np.asarray(volume).dtype)

    if not is_valid_affine(affine, 3):
        raise ValueError("Invalid affine transform matrix")

    with nogil:

        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if affine is not None:
                        dkk = _apply_affine_3d_x0(k, i, j, 1, affine)
                        dii = _apply_affine_3d_x1(k, i, j, 1, affine)
                        djj = _apply_affine_3d_x2(k, i, j, 1, affine)
                    else:
                        dkk = k
                        dii = i
                        djj = j
                    inside = _interpolate_scalar_3d[floating](volume, dkk,
                        dii, djj, &out[k,i,j])
    return np.asarray(out)


def warp_3d_nn(number[:, :, :] volume, floating[:, :, :, :] d1,
               double[:, :] affine_idx_in=None,
               double[:, :] affine_idx_out=None,
               double[:, :] affine_disp=None,
               int[:] out_shape=None):
    r"""Warps a 3D volume using using nearest-neighbor interpolation

    Deforms the input volume under the given transformation. The warped volume
    is computed using nearest-neighbor interpolation and is given by:

    (1) warped[i] = volume[ C * d1[A*i] + B*i ]

    where A = affine_idx_in, B = affine_idx_out, C = affine_disp and i denotes
    the discrete coordinates of a voxel in the sampling grid of
    shape = out_shape.

    Parameters
    ----------
    volume : array, shape (S, R, C)
        the input volume to be transformed
    d1 : array, shape (S', R', C', 3)
        the displacement field driving the transformation
    affine_idx_in : array, shape (4, 4)
        the matrix A in eq. (1) above
    affine_idx_out : array, shape (4, 4)
        the matrix B in eq. (1) above
    affine_disp : array, shape (4, 4)
        the matrix C in eq. (1) above
    out_shape : array, shape (3,)
        the number of slices, rows and columns of the sampling grid

    Returns
    -------
    warped : array, shape = out_shape
        the transformed volume

    Notes
    -----
    To illustrate the use of this function, consider a displacement field d1
    with grid-to-space transformation R, a volume with grid-to-space
    transformation T and let's say we want to sample the warped volume on a
    grid with grid-to-space transformation S (sampling grid). For each voxel
    in the sampling grid with discrete coordinates i, the warped volume is
    given by:

    (2) warped[i] = volume[Tinv * ( d1[Rinv * S * i] + S * i ) ]

    where Tinv = T^{-1} and Rinv = R^{-1}. By identifying A = Rinv * S,
    B = Tinv * S, C = Tinv we can use this function to efficiently warp the
    input image.
    """
    cdef:
        cnp.npy_intp nslices = volume.shape[0]
        cnp.npy_intp nrows = volume.shape[1]
        cnp.npy_intp ncols = volume.shape[2]
        cnp.npy_intp nsVol = volume.shape[0]
        cnp.npy_intp nrVol = volume.shape[1]
        cnp.npy_intp ncVol = volume.shape[2]
        cnp.npy_intp i, j, k
        int inside
        double dkk, dii, djj, dk, di, dj

    if not is_valid_affine(affine_idx_in, 3):
        raise ValueError("Invalid inner index multiplication matrix")
    if not is_valid_affine(affine_idx_out, 3):
        raise ValueError("Invalid outer index multiplication matrix")
    if not is_valid_affine(affine_disp, 3):
        raise ValueError("Invalid displacement multiplication matrix")

    if out_shape is not None:
        nslices = out_shape[0]
        nrows = out_shape[1]
        ncols = out_shape[2]
    elif d1 is not None:
        nslices = d1.shape[0]
        nrows = d1.shape[1]
        ncols = d1.shape[2]

    cdef number[:, :, :] warped = np.zeros(shape=(nslices, nrows, ncols),
                                           dtype=np.asarray(volume).dtype)
    cdef floating[:] tmp = np.zeros(shape=(3,), dtype = np.asarray(d1).dtype)

    with nogil:

        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if affine_idx_in is None:
                        dkk = d1[k, i, j, 0]
                        dii = d1[k, i, j, 1]
                        djj = d1[k, i, j, 2]
                    else:
                        dk = _apply_affine_3d_x0(
                            k, i, j, 1, affine_idx_in)
                        di = _apply_affine_3d_x1(
                            k, i, j, 1, affine_idx_in)
                        dj = _apply_affine_3d_x2(
                            k, i, j, 1, affine_idx_in)
                        inside = _interpolate_vector_3d[floating](d1, dk, di,
                                                                  dj, &tmp[0])
                        dkk = tmp[0]
                        dii = tmp[1]
                        djj = tmp[2]

                    if affine_disp is not None:
                        dk = _apply_affine_3d_x0(
                            dkk, dii, djj, 0, affine_disp)
                        di = _apply_affine_3d_x1(
                            dkk, dii, djj, 0, affine_disp)
                        dj = _apply_affine_3d_x2(
                            dkk, dii, djj, 0, affine_disp)
                    else:
                        dk = dkk
                        di = dii
                        dj = djj

                    if affine_idx_out is not None:
                        dkk = dk + _apply_affine_3d_x0(k, i, j, 1,
                                                       affine_idx_out)
                        dii = di + _apply_affine_3d_x1(k, i, j, 1,
                                                       affine_idx_out)
                        djj = dj + _apply_affine_3d_x2(k, i, j, 1,
                                                       affine_idx_out)
                    else:
                        dkk = dk + k
                        dii = di + i
                        djj = dj + j

                    inside = _interpolate_scalar_nn_3d[number](volume, dkk, dii, djj,
                                                       &warped[k,i,j])
    return np.asarray(warped)


def transform_3d_affine_nn(number[:, :, :] volume, int[:] ref_shape,
                           double[:, :] affine=None):
    r"""Transforms a 3D volume by an affine transform with NN interpolation

    Deforms the input volume under the given affine transformation using
    nearest neighbor interpolation. The shape of the resulting volume
    is given by ref_shape. If the affine matrix is None, it is taken as the
    identity.

    Parameters
    ----------
    volume : array, shape (S, R, C)
        the input volume to be transformed
    ref_shape : array, shape (3,)
        the shape of the resulting volume
    affine : array, shape (4, 4)
        the affine transform to be applied

    Returns
    -------
    out : array, shape (S', R', C')
        the transformed volume

    Notes
    -----
    The reason it is necessary to provide the intended shape of the resulting
    volume is because the affine transformation is defined on all R^{3}
    but we must sample a finite lattice. Also the resulting shape may not be
    necessarily equal to the input shape, unless we are interested on
    endomorphisms only and not general diffeomorphisms.
    """
    cdef:
        cnp.npy_intp nslices = ref_shape[0]
        cnp.npy_intp nrows = ref_shape[1]
        cnp.npy_intp ncols = ref_shape[2]
        cnp.npy_intp nsVol = volume.shape[0]
        cnp.npy_intp nrVol = volume.shape[1]
        cnp.npy_intp ncVol = volume.shape[2]
        double dkk, dii, djj, tmp0, tmp1
        double alpha, beta, gamma, calpha, cbeta, cgamma
        cnp.npy_intp k, i, j, kk, ii, jj
        number[:, :, :] out = np.zeros((nslices, nrows, ncols),
                                        dtype=np.asarray(volume).dtype)

    if not is_valid_affine(affine, 3):
        raise ValueError("Invalid affine transform matrix")

    with nogil:

        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if affine is not None:
                        dkk = _apply_affine_3d_x0(k, i, j, 1, affine)
                        dii = _apply_affine_3d_x1(k, i, j, 1, affine)
                        djj = _apply_affine_3d_x2(k, i, j, 1, affine)
                    else:
                        dkk = k
                        dii = i
                        djj = j
                    _interpolate_scalar_nn_3d[number](volume, dkk, dii, djj,
                                                      &out[k,i,j])
    return np.asarray(out)


def warp_2d(floating[:, :] image, floating[:, :, :] d1,
            double[:, :] affine_idx_in=None,
            double[:, :] affine_idx_out=None,
            double[:, :] affine_disp=None,
            int[:] out_shape=None):
    r"""Warps a 2D image using bilinear interpolation

    Deforms the input image under the given transformation. The warped image
    is computed using bi-linear interpolation and is given by:

    (1) warped[i] = image[ C * d1[A*i] + B*i ]

    where A = affine_idx_in, B = affine_idx_out, C = affine_disp and i denotes
    the discrete coordinates of a voxel in the sampling grid of
    shape = out_shape.

    Parameters
    ----------
    image : array, shape (R, C)
        the input image to be transformed
    d1 : array, shape (R', C', 2)
        the displacement field driving the transformation
    affine_idx_in : array, shape (3, 3)
        the matrix A in eq. (1) above
    affine_idx_out : array, shape (3, 3)
        the matrix B in eq. (1) above
    affine_disp : array, shape (3, 3)
        the matrix C in eq. (1) above
    out_shape : array, shape (2,)
        the number of rows and columns of the sampling grid

    Returns
    -------
    warped : array, shape = out_shape
        the transformed image

    Notes
    -----
    To illustrate the use of this function, consider a displacement field d1
    with grid-to-space transformation R, an image with grid-to-space
    transformation T and let's say we want to sample the warped image on a
    grid with grid-to-space transformation S (sampling grid). For each voxel
    in the sampling grid with discrete coordinates i, the warped image is
    given by:

    (2) warped[i] = image[Tinv * ( d1[Rinv * S * i] + S * i ) ]

    where Tinv = T^{-1} and Rinv = R^{-1}. By identifying A = Rinv * S,
    B = Tinv * S, C = Tinv we can use this function to efficiently warp the
    input image.
    """
    cdef:
        cnp.npy_intp nrows = image.shape[0]
        cnp.npy_intp ncols = image.shape[1]
        cnp.npy_intp nrVol = image.shape[0]
        cnp.npy_intp ncVol = image.shape[1]
        cnp.npy_intp i, j, ii, jj
        double di, dj, dii, djj

    if not is_valid_affine(affine_idx_in, 2):
        raise ValueError("Invalid inner index multiplication matrix")
    if not is_valid_affine(affine_idx_out, 2):
        raise ValueError("Invalid outer index multiplication matrix")
    if not is_valid_affine(affine_disp, 2):
        raise ValueError("Invalid displacement multiplication matrix")

    if out_shape is not None:
        nrows = out_shape[0]
        ncols = out_shape[1]
    elif d1 is not None:
        nrows = d1.shape[0]
        ncols = d1.shape[1]
    cdef floating[:, :] warped = np.zeros(shape=(nrows, ncols),
                                          dtype=np.asarray(image).dtype)
    cdef floating[:] tmp = np.zeros(shape=(2,), dtype=np.asarray(d1).dtype)

    with nogil:

        for i in range(nrows):
            for j in range(ncols):
                # Apply inner index pre-multiplication
                if affine_idx_in is None:
                    dii = d1[i, j, 0]
                    djj = d1[i, j, 1]
                else:
                    di = _apply_affine_2d_x0(
                        i, j, 1, affine_idx_in)
                    dj = _apply_affine_2d_x1(
                        i, j, 1, affine_idx_in)
                    _interpolate_vector_2d[floating](d1, di, dj, &tmp[0])
                    dii = tmp[0]
                    djj = tmp[1]

                # Apply displacement multiplication
                if affine_disp is not None:
                    di = _apply_affine_2d_x0(
                        dii, djj, 0, affine_disp)
                    dj = _apply_affine_2d_x1(
                        dii, djj, 0, affine_disp)
                else:
                    di = dii
                    dj = djj

                # Apply outer index multiplication and add the displacements
                if affine_idx_out is not None:
                    dii = di + _apply_affine_2d_x0(i, j, 1, affine_idx_out)
                    djj = dj + _apply_affine_2d_x1(i, j, 1, affine_idx_out)
                else:
                    dii = di + i
                    djj = dj + j

                # Interpolate the input image at the resulting location
                _interpolate_scalar_2d[floating](image, dii, djj,
                                                 &warped[i, j])
    return np.asarray(warped)


def transform_2d_affine(floating[:, :] image, int[:] ref_shape,
                        double[:, :] affine=None):
    r"""Transforms a 2D image by an affine transform with bilinear interp.

    Deforms the input image under the given affine transformation using
    tri-linear interpolation. The shape of the resulting image
    is given by ref_shape. If the affine matrix is None, it is taken as the
    identity.

    Parameters
    ----------
    image : array, shape (R, C)
        the input image to be transformed
    ref_shape : array, shape (2,)
        the shape of the resulting image
    affine : array, shape (3, 3)
        the affine transform to be applied

    Returns
    -------
    out : array, shape (R', C')
        the transformed image

    Notes
    -----
    The reason it is necessary to provide the intended shape of the resulting
    image is because the affine transformation is defined on all R^{2}
    but we must sample a finite lattice. Also the resulting shape may not be
    necessarily equal to the input shape, unless we are interested on
    endomorphisms only and not general diffeomorphisms.
    """
    cdef:
        cnp.npy_intp nrows = ref_shape[0]
        cnp.npy_intp ncols = ref_shape[1]
        cnp.npy_intp nrVol = image.shape[0]
        cnp.npy_intp ncVol = image.shape[1]
        cnp.npy_intp i, j, ii, jj
        double dii, djj, tmp0
        double alpha, beta, calpha, cbeta
        floating[:, :] out = np.zeros(shape=(nrows, ncols),
                                      dtype=np.asarray(image).dtype)

    if not is_valid_affine(affine, 2):
        raise ValueError("Invalid affine transform matrix")

    with nogil:

        for i in range(nrows):
            for j in range(ncols):
                if affine is not None:
                    dii = _apply_affine_2d_x0(i, j, 1, affine)
                    djj = _apply_affine_2d_x1(i, j, 1, affine)
                else:
                    dii = i
                    djj = j
                _interpolate_scalar_2d[floating](image, dii, djj,
                                                 &out[i, j])
    return np.asarray(out)


def warp_2d_nn(number[:, :] image, floating[:, :, :] d1,
               double[:, :] affine_idx_in=None,
               double[:, :] affine_idx_out=None,
               double[:, :] affine_disp=None,
               int[:] out_shape=None):
    r"""Warps a 2D image using nearest neighbor interpolation

    Deforms the input image under the given transformation. The warped image
    is computed using nearest-neighbor interpolation and is given by:

    (1) warped[i] = image[ C * d1[A*i] + B*i ]

    where A = affine_idx_in, B = affine_idx_out, C = affine_disp and i denotes
    the discrete coordinates of a voxel in the sampling grid of
    shape = out_shape.

    Parameters
    ----------
    image : array, shape (R, C)
        the input image to be transformed
    d1 : array, shape (R', C', 2)
        the displacement field driving the transformation
    affine_idx_in : array, shape (3, 3)
        the matrix A in eq. (1) above
    affine_idx_out : array, shape (3, 3)
        the matrix B in eq. (1) above
    affine_disp : array, shape (3, 3)
        the matrix C in eq. (1) above
    out_shape : array, shape (2,)
        the number of rows and columns of the sampling grid

    Returns
    -------
    warped : array, shape = out_shape
        the transformed image

    Notes
    -----
    To illustrate the use of this function, consider a displacement field d1
    with grid-to-space transformation R, an image with grid-to-space
    transformation T and let's say we want to sample the warped image on a
    grid with grid-to-space transformation S (sampling grid). For each voxel
    in the sampling grid with discrete coordinates i, the warped image is
    given by:

    (2) warped[i] = image[Tinv * ( d1[Rinv * S * i] + S * i ) ]

    where Tinv = T^{-1} and Rinv = R^{-1}. By identifying A = Rinv * S,
    B = Tinv * S, C = Tinv we can use this function to efficiently warp the
    input image.
    """
    cdef:
        cnp.npy_intp nrows = image.shape[0]
        cnp.npy_intp ncols = image.shape[1]
        cnp.npy_intp nrVol = image.shape[0]
        cnp.npy_intp ncVol = image.shape[1]
        cnp.npy_intp i, j, ii, jj
        double di, dj, dii, djj

    if not is_valid_affine(affine_idx_in, 2):
        raise ValueError("Invalid inner index multiplication matrix")
    if not is_valid_affine(affine_idx_out, 2):
        raise ValueError("Invalid outer index multiplication matrix")
    if not is_valid_affine(affine_disp, 2):
        raise ValueError("Invalid displacement multiplication matrix")

    if out_shape is not None:
        nrows = out_shape[0]
        ncols = out_shape[1]
    elif d1 is not None:
        nrows = d1.shape[0]
        ncols = d1.shape[1]
    cdef number[:, :] warped = np.zeros(shape=(nrows, ncols),
                                        dtype=np.asarray(image).dtype)
    cdef floating[:] tmp = np.zeros(shape=(2,), dtype=np.asarray(d1).dtype)

    with nogil:

        for i in range(nrows):
            for j in range(ncols):
                # Apply inner index pre-multiplication
                if affine_idx_in is None:
                    dii = d1[i, j, 0]
                    djj = d1[i, j, 1]
                else:
                    di = _apply_affine_2d_x0(
                        i, j, 1, affine_idx_in)
                    dj = _apply_affine_2d_x1(
                        i, j, 1, affine_idx_in)
                    _interpolate_vector_2d[floating](d1, di, dj, &tmp[0])
                    dii = tmp[0]
                    djj = tmp[1]

                # Apply displacement multiplication
                if affine_disp is not None:
                    di = _apply_affine_2d_x0(
                        dii, djj, 0, affine_disp)
                    dj = _apply_affine_2d_x1(
                        dii, djj, 0, affine_disp)
                else:
                    di = dii
                    dj = djj

                # Apply outer index multiplication and add the displacements
                if affine_idx_out is not None:
                    dii = di + _apply_affine_2d_x0(i, j, 1, affine_idx_out)
                    djj = dj + _apply_affine_2d_x1(i, j, 1, affine_idx_out)
                else:
                    dii = di + i
                    djj = dj + j

                # Interpolate the input image at the resulting location
                _interpolate_scalar_nn_2d[number](image, dii, djj,
                                                  &warped[i, j])
    return np.asarray(warped)


def transform_2d_affine_nn(number[:, :] image, int[:] ref_shape,
                           double[:, :] affine=None):
    r"""Transforms a 2D image by an affine transform with NN interpolation

    Deforms the input image under the given affine transformation using
    nearest neighbor interpolation. The shape of the resulting image
    is given by ref_shape. If the affine matrix is None, it is taken as the
    identity.

    Parameters
    ----------
    image : array, shape (R, C)
        the input image to be transformed
    ref_shape : array, shape (2,)
        the shape of the resulting image
    affine : array, shape (3, 3)
        the affine transform to be applied

    Returns
    -------
    out : array, shape (R', C')
        the transformed image

    Notes
    -----
    The reason it is necessary to provide the intended shape of the resulting
    image is because the affine transformation is defined on all R^{2}
    but we must sample a finite lattice. Also the resulting shape may not be
    necessarily equal to the input shape, unless we are interested on
    endomorphisms only and not general diffeomorphisms.
    """
    cdef:
        cnp.npy_intp nrows = ref_shape[0]
        cnp.npy_intp ncols = ref_shape[1]
        cnp.npy_intp nrVol = image.shape[0]
        cnp.npy_intp ncVol = image.shape[1]
        double dii, djj, tmp0
        double alpha, beta, calpha, cbeta
        cnp.npy_intp i, j, ii, jj
        number[:, :] out = np.zeros((nrows, ncols),
                                    dtype=np.asarray(image).dtype)

    if not is_valid_affine(affine, 2):
        raise ValueError("Invalid affine transform matrix")

    with nogil:

        for i in range(nrows):
            for j in range(ncols):
                if affine is not None:
                    dii = _apply_affine_2d_x0(i, j, 1, affine)
                    djj = _apply_affine_2d_x1(i, j, 1, affine)
                else:
                    dii = i
                    djj = j
                _interpolate_scalar_nn_2d[number](image, dii, djj,
                                                  &out[i, j])
    return np.asarray(out)


def resample_displacement_field_3d(floating[:, :, :, :] field,
                                   double[:] factors, int[:] out_shape):
    r"""Resamples a 3D vector field to a custom target shape

    Resamples the given 3D displacement field on a grid of the requested shape,
    using the given scale factors. More precisely, the resulting displacement
    field at each grid cell i is given by

    D[i] = field[Diag(factors) * i]

    Parameters
    ----------
    factors : array, shape (3,)
        the scaling factors mapping (integer) grid coordinates in the resampled
        grid to (floating point) grid coordinates in the original grid
    out_shape : array, shape (3,)
        the desired shape of the resulting grid

    Returns
    -------
    expanded : array, shape = out_shape + (3, )
        the resampled displacement field
    """
    ftype = np.asarray(field).dtype
    cdef:
        cnp.npy_intp tslices = out_shape[0]
        cnp.npy_intp trows = out_shape[1]
        cnp.npy_intp tcols = out_shape[2]
        cnp.npy_intp k, i, j
        int inside
        double dkk, dii, djj
        floating[:, :, :, :] expanded = np.zeros((tslices, trows, tcols, 3),
                                                 dtype=ftype)

    for k in range(tslices):
        for i in range(trows):
            for j in range(tcols):
                dkk = <double> k * factors[0]
                dii = <double> i * factors[1]
                djj = <double> j * factors[2]
                _interpolate_vector_3d[floating](field, dkk, dii, djj,
                                                 &expanded[k, i, j, 0])
    return np.asarray(expanded)


def resample_displacement_field_2d(floating[:, :, :] field, double[:] factors,
                                   int[:] out_shape):
    r"""Resamples a 2D vector field to a custom target shape

    Resamples the given 2D displacement field on a grid of the requested shape,
    using the given scale factors. More precisely, the resulting displacement
    field at each grid cell i is given by

    D[i] = field[Diag(factors) * i]

    Parameters
    ----------
    factors : array, shape (2,)
        the scaling factors mapping (integer) grid coordinates in the resampled
        grid to (floating point) grid coordinates in the original grid
    out_shape : array, shape (2,)
        the desired shape of the resulting grid

    Returns
    -------
    expanded : array, shape = out_shape + (2, )
        the resampled displacement field
    """
    ftype = np.asarray(field).dtype
    cdef:
        cnp.npy_intp trows = out_shape[0]
        cnp.npy_intp tcols = out_shape[1]
        cnp.npy_intp i, j
        int inside
        double dii, djj
        floating[:, :, :] expanded = np.zeros((trows, tcols, 2), dtype=ftype)

    for i in range(trows):
        for j in range(tcols):
            dii = i*factors[0]
            djj = j*factors[1]
            inside = _interpolate_vector_2d[floating](field, dii, djj,
                                                      &expanded[i, j, 0])
    return np.asarray(expanded)


def create_random_displacement_2d(int[:] from_shape,
                                  double[:, :] from_grid2world,
                                  int[:] to_shape,
                                  double[:, :] to_grid2world):
    r"""Creates a random 2D displacement 'exactly' mapping points of two grids

    Creates a random 2D displacement field mapping points of an input discrete
    domain (with dimensions given by from_shape) to points of an output
    discrete domain (with shape given by to_shape). The affine matrices
    bringing discrete coordinates to physical space are given by from_grid2world
    (for the displacement field discretization) and to_grid2world (for the target
    discretization). Since this function is intended to be used for testing,
    voxels in the input domain will never be assigned to boundary voxels on the
    output domain.

    Parameters
    ----------
    from_shape : array, shape (2,)
        the grid shape where the displacement field will be defined on.
    from_grid2world : array, shape (3,3)
        the grid-to-space transformation of the displacement field
    to_shape : array, shape (2,)
        the grid shape where the deformation field will map the input grid to.
    to_grid2world : array, shape (3,3)
        the grid-to-space transformation of the mapped grid

    Returns
    -------
    output : array, shape = from_shape
        the random displacement field in the physical domain
    int_field : array, shape = from_shape
        the assignment of each point in the input grid to the target grid
    """
    cdef:
        cnp.npy_intp i, j, ri, rj
        double di, dj, dii, djj
        int[:, :, :] int_field = np.empty(tuple(from_shape) + (2,),
                                            dtype=np.int32)
        double[:, :, :] output = np.zeros(tuple(from_shape) + (2,),
                                          dtype=np.float64)
        cnp.npy_intp dom_size = from_shape[0]*from_shape[1]

    if not is_valid_affine(from_grid2world, 2):
        raise ValueError("Invalid 'from' affine transform matrix")
    if not is_valid_affine(to_grid2world, 2):
        raise ValueError("Invalid 'to' affine transform matrix")

    # compute the actual displacement field in the physical space
    for i in range(from_shape[0]):
        for j in range(from_shape[1]):
            # randomly choose where each input grid point will be mapped to in
            # the target grid
            ri = np.random.randint(1, to_shape[0]-1)
            rj = np.random.randint(1, to_shape[1]-1)
            int_field[i, j, 0] = ri
            int_field[i, j, 1] = rj

            # convert the input point to physical coordinates
            if from_grid2world is not None:
                di = _apply_affine_2d_x0(i, j, 1, from_grid2world)
                dj = _apply_affine_2d_x1(i, j, 1, from_grid2world)
            else:
                di = i
                dj = j

            # convert the output point to physical coordinates
            if to_grid2world is not None:
                dii = _apply_affine_2d_x0(ri, rj, 1, to_grid2world)
                djj = _apply_affine_2d_x1(ri, rj, 1, to_grid2world)
            else:
                dii = ri
                djj = rj

            # the displacement vector at (i,j) must be the target point minus
            # the original point, both in physical space

            output[i, j, 0] = dii - di
            output[i, j, 1] = djj - dj

    return np.asarray(output), np.asarray(int_field)


def create_random_displacement_3d(int[:] from_shape, double[:, :] from_grid2world,
                                  int[:] to_shape, double[:, :] to_grid2world):
    r"""Creates a random 3D displacement 'exactly' mapping points of two grids

    Creates a random 3D displacement field mapping points of an input discrete
    domain (with dimensions given by from_shape) to points of an output
    discrete domain (with shape given by to_shape). The affine matrices
    bringing discrete coordinates to physical space are given by from_grid2world
    (for the displacement field discretization) and to_grid2world (for the target
    discretization). Since this function is intended to be used for testing,
    voxels in the input domain will never be assigned to boundary voxels on the
    output domain.

    Parameters
    ----------
    from_shape : array, shape (3,)
        the grid shape where the displacement field will be defined on.
    from_grid2world : array, shape (4,4)
        the grid-to-space transformation of the displacement field
    to_shape : array, shape (3,)
        the grid shape where the deformation field will map the input grid to.
    to_grid2world : array, shape (4,4)
        the grid-to-space transformation of the mapped grid

    Returns
    -------
    output : array, shape = from_shape
        the random displacement field in the physical domain
    int_field : array, shape = from_shape
        the assignment of each point in the input grid to the target grid
    """
    cdef:
        cnp.npy_intp i, j, k, ri, rj, rk
        double di, dj, dk, dii, djj, dkk
        int[:, :, :, :] int_field = np.empty(tuple(from_shape) + (3,),
                                               dtype=np.int32)
        double[:, :, :, :] output = np.zeros(tuple(from_shape) + (3,),
                                             dtype=np.float64)
        cnp.npy_intp dom_size = from_shape[0]*from_shape[1]*from_shape[2]

    if not is_valid_affine(from_grid2world, 3):
        raise ValueError("Invalid 'from' affine transform matrix")
    if not is_valid_affine(to_grid2world, 3):
        raise ValueError("Invalid 'to' affine transform matrix")

    # compute the actual displacement field in the physical space
    for k in range(from_shape[0]):
        for i in range(from_shape[1]):
            for j in range(from_shape[2]):
                # randomly choose the location of each point on the target grid
                rk = np.random.randint(1, to_shape[0]-1)
                ri = np.random.randint(1, to_shape[1]-1)
                rj = np.random.randint(1, to_shape[2]-1)
                int_field[k, i, j, 0] = rk
                int_field[k, i, j, 1] = ri
                int_field[k, i, j, 2] = rj

                # convert the input point to physical coordinates
                if from_grid2world is not None:
                    dk = _apply_affine_3d_x0(k, i, j, 1, from_grid2world)
                    di = _apply_affine_3d_x1(k, i, j, 1, from_grid2world)
                    dj = _apply_affine_3d_x2(k, i, j, 1, from_grid2world)
                else:
                    dk = k
                    di = i
                    dj = j

                # convert the output point to physical coordinates
                if to_grid2world is not None:
                    dkk = _apply_affine_3d_x0(rk, ri, rj, 1, to_grid2world)
                    dii = _apply_affine_3d_x1(rk, ri, rj, 1, to_grid2world)
                    djj = _apply_affine_3d_x2(rk, ri, rj, 1, to_grid2world)
                else:
                    dkk = rk
                    dii = ri
                    djj = rj

                # the displacement vector at (i,j) must be the target point minus
                # the original point, both in physical space

                output[k, i, j, 0] = dkk - dk
                output[k, i, j, 1] = dii - di
                output[k, i, j, 2] = djj - dj

    return np.asarray(output), np.asarray(int_field)


def create_harmonic_fields_2d(cnp.npy_intp nrows, cnp.npy_intp ncols,
                              double b, double m):
    r"""Creates an invertible 2D displacement field

    Creates the invertible displacement fields used in Chen et al. eqs.
    9 and 10 [1]

    Parameters
    ----------
    nrows : int
        number of rows in the resulting harmonic field
    ncols : int
        number of columns in the resulting harmonic field
    b, m : float
        parameters of the harmonic field (as in [1]). To understand the effect
        of these parameters, please consider plotting a deformed image
        (a circle or a grid) under the deformation field, or see examples
        in [1]

    Returns
    -------
    d : array, shape (nrows, ncols, 2)
        the harmonic displacement field
    inv : array, shape (nrows, ncols, 2)
        the analytical inverse of the harmonic displacement field

    [1] Chen, M., Lu, W., Chen, Q., Ruchala, K. J., & Olivera, G. H. (2008).
        A simple fixed-point approach to invert a deformation field.
        Medical Physics, 35(1), 81. doi:10.1118/1.2816107
    """
    cdef:
        cnp.npy_intp mid_row = nrows/2
        cnp.npy_intp mid_col = ncols/2
        cnp.npy_intp i, j, ii, jj
        double theta
        double[:, :, :] d = np.zeros((nrows, ncols, 2), dtype=np.float64)
        double[:, :, :] inv = np.zeros((nrows, ncols, 2), dtype=np.float64)
    for i in range(nrows):
        for j in range(ncols):
            ii = i - mid_row
            jj = j - mid_col
            theta = atan2(ii, jj)
            d[i, j, 0] = ii * (1.0 / (1 + b * cos(m * theta)) - 1.0)
            d[i, j, 1] = jj * (1.0 / (1 + b * cos(m * theta)) - 1.0)
            inv[i, j, 0] = b * cos(m * theta) * ii
            inv[i, j, 1] = b * cos(m * theta) * jj

    return np.asarray(d), np.asarray(inv)


def create_harmonic_fields_3d(int nslices, cnp.npy_intp nrows,
                              cnp.npy_intp ncols, double b, double m):
    r"""Creates an invertible 3D displacement field

    Creates the invertible displacement fields used in Chen et al. eqs.
    9 and 10 [1] computing the angle theta along z-slides.

    Parameters
    ----------
    nslices : int
        number of slices in the resulting harmonic field
    nrows : int
        number of rows in the resulting harmonic field
    ncols : int
        number of columns in the resulting harmonic field
    b, f : float
        parameters of the harmonic field (as in [1]). To understand the effect
        of these parameters, please consider plotting a deformed image
        (e.g. a circle or a grid) under the deformation field, or see examples
        in [1]

    Returns
    -------
    d : array, shape (nslices, nrows, ncols, 3)
        the harmonic displacement field
    inv : array, shape (nslices, nrows, ncols, 3)
        the analytical inverse of the harmonic displacement field

    [1] Chen, M., Lu, W., Chen, Q., Ruchala, K. J., & Olivera, G. H. (2008).
        A simple fixed-point approach to invert a deformation field.
        Medical Physics, 35(1), 81. doi:10.1118/1.2816107
    """
    cdef:
        cnp.npy_intp mid_slice = nslices / 2
        cnp.npy_intp mid_row = nrows / 2
        cnp.npy_intp mid_col = ncols / 2
        cnp.npy_intp i, j, k, ii, jj, kk
        double theta
        double[:, :, :, :] d = np.zeros((nslices, nrows, ncols, 3),
                                        dtype=np.float64)
        double[:, :, :, :] inv = np.zeros((nslices, nrows, ncols, 3),
                                          dtype=np.float64)
    for k in range(nslices):
        for i in range(nrows):
            for j in range(ncols):
                kk = k - mid_slice
                ii = i - mid_row
                jj = j - mid_col
                theta = atan2(ii, jj)
                d[k, i, j, 0] = kk * (1.0 / (1 + b * cos(m * theta)) - 1.0)
                d[k, i, j, 1] = ii * (1.0 / (1 + b * cos(m * theta)) - 1.0)
                d[k, i, j, 2] = jj * (1.0 / (1 + b * cos(m * theta)) - 1.0)
                inv[k, i, j, 0] = b * cos(m * theta) * kk
                inv[k, i, j, 1] = b * cos(m * theta) * ii
                inv[k, i, j, 2] = b * cos(m * theta) * jj

    return np.asarray(d), np.asarray(inv)


def create_circle(cnp.npy_intp nrows, cnp.npy_intp ncols, cnp.npy_intp radius):
    r"""
    Create a binary 2D image where pixel values are 1 iff their distance
    to the center of the image is less than or equal to radius.

    Parameters
    ----------
    nrows : int
        number of rows of the resulting image
    ncols : int
        number of columns of the resulting image
    radius : int
        the radius of the circle

    Returns
    -------
    c : array, shape (nrows, ncols)
        the binary image of the circle with the requested dimensions
    """
    cdef:
        cnp.npy_intp mid_row = nrows/2
        cnp.npy_intp mid_col = ncols/2
        cnp.npy_intp i, j, ii, jj
        double r
        double[:, :] c = np.zeros((nrows, ncols), dtype=np.float64)
    for i in range(nrows):
        for j in range(ncols):
            ii = i - mid_row
            jj = j - mid_col
            r = sqrt(ii*ii + jj*jj)
            if r <= radius:
                c[i, j] = 1
            else:
                c[i, j] = 0
    return np.asarray(c)


def create_sphere(cnp.npy_intp nslices, cnp.npy_intp nrows,
                  cnp.npy_intp ncols, cnp.npy_intp radius):
    r"""
    Create a binary 3D image where voxel values are 1 iff their distance
    to the center of the image is less than or equal to radius.

    Parameters
    ----------
    nslices : int
        number if slices of the resulting image
    nrows : int
        number of rows of the resulting image
    ncols : int
        number of columns of the resulting image
    radius : int
        the radius of the sphere

    Returns
    -------
    c : array, shape (nslices, nrows, ncols)
        the binary image of the sphere with the requested dimensions
    """
    cdef:
        cnp.npy_intp mid_slice = nslices/2
        cnp.npy_intp mid_row = nrows/2
        cnp.npy_intp mid_col = ncols/2
        cnp.npy_intp i, j, k, ii, jj, kk
        double r
        double[:, :, :] s = np.zeros((nslices, nrows, ncols), dtype=np.float64)

    for k in range(nslices):
        for i in range(nrows):
            for j in range(ncols):
                kk = k - mid_slice
                ii = i - mid_row
                jj = j - mid_col
                r = sqrt(ii*ii + jj*jj + kk*kk)
                if r <= radius:
                    s[k, i, j] = 1
                else:
                    s[k, i, j] = 0
    return np.asarray(s)


def _gradient_3d(floating[:, :, :] img, double[:, :] img_world2grid,
                 double[:] img_spacing, double[:, :] out_grid2world,
                 floating[:, :, :, :] out, int[:, :, :] inside):
    r""" Gradient of a 3D image in physical space coordinates

    Each grid cell (i, j, k) in the sampling grid (determined by
    out.shape) is mapped to its corresponding physical point (x, y, z) by
    multiplying out_grid2world (its grid-to-space transform) by (i, j, k),
    then the image is interpolated, at

    P1=(x + h, y, z), Q1=(x - h, y, z)
    P2=(x, y + h, z), Q2=(x, y - h, z)
    P3=(x, y, z + h), Q3=(x, y, z - h)

    (by mapping Pi and Qi to the grid using img_world2grid: the inverse of the
    grid-to-space transform of img). The displacement parameter h is of
    magnitude 0.5 (in physical space units), therefore the approximated partial
    derivatives are given by the difference between the image interpolated at
    Pi and Qi.

    Parameters
    ----------
    img : array, shape (S, R, C)
        the input volume whose gradient will be computed
    img_world2grid : array, shape (4, 4)
        the space-to-grid transform matrix associated to img
    img_spacing : array, shape (3,)
        the spacing between voxels (voxel size along each axis) of the input
        volume
    out_grid2world : array, shape (4, 4)
        the grid-to-space transform associated to the sampling grid
    out : array, shape (S', R', C', 3)
        the buffer in which to store the image gradient
    inside : array, shape (S', R', C')
        the buffer in which to store the flags indicating whether the sample
        point lies inside (=1) or outside (=0) the image grid
    """
    cdef:
        cnp.npy_intp nslices = out.shape[0]
        cnp.npy_intp nrows = out.shape[1]
        cnp.npy_intp ncols = out.shape[2]
        cnp.npy_intp i, j, k, in_flag
        double tmp
        double[:] x = np.empty(shape=(3,), dtype=np.float64)
        double[:] dx = np.empty(shape=(3,), dtype=np.float64)
        double[:] h = np.empty(shape=(3,), dtype=np.float64)
        double[:] q = np.empty(shape=(3,), dtype=np.float64)
    with nogil:
        h[0] = 0.5 * img_spacing[0]
        h[1] = 0.5 * img_spacing[1]
        h[2] = 0.5 * img_spacing[2]
        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    inside[k, i, j] = 1
                    # Compute coordinates of index (k, i, j) in physical space
                    x[0] = _apply_affine_3d_x0(k, i, j, 1, out_grid2world)
                    x[1] = _apply_affine_3d_x1(k, i, j, 1, out_grid2world)
                    x[2] = _apply_affine_3d_x2(k, i, j, 1, out_grid2world)
                    dx[:] = x[:]
                    for p in range(3):
                        # Compute coordinates of point dx on img's grid
                        dx[p] = x[p] - h[p]
                        q[0] = _apply_affine_3d_x0(dx[0], dx[1], dx[2], 1,
                                                   img_world2grid)
                        q[1] = _apply_affine_3d_x1(dx[0], dx[1], dx[2], 1,
                                                   img_world2grid)
                        q[2] = _apply_affine_3d_x2(dx[0], dx[1], dx[2], 1,
                                                   img_world2grid)
                        # Interpolate img at q
                        in_flag = _interpolate_scalar_3d[floating](img, q[0],
                            q[1], q[2], &out[k, i, j, p])
                        if in_flag == 0:
                            out[k, i, j, p] = 0
                            inside[k, i, j] = 0
                            continue
                        tmp = out[k, i, j, p]
                        # Compute coordinates of point dx on img's grid
                        dx[p] = x[p] + h[p]
                        q[0] = _apply_affine_3d_x0(dx[0], dx[1], dx[2], 1,
                                                   img_world2grid)
                        q[1] = _apply_affine_3d_x1(dx[0], dx[1], dx[2], 1,
                                                   img_world2grid)
                        q[2] = _apply_affine_3d_x2(dx[0], dx[1], dx[2], 1,
                                                   img_world2grid)
                        # Interpolate img at q
                        in_flag = _interpolate_scalar_3d[floating](img, q[0],
                            q[1], q[2], &out[k, i, j, p])
                        if in_flag == 0:
                            out[k, i, j, p] = 0
                            inside[k, i, j] = 0
                            continue
                        out[k, i, j, p] = (out[k, i, j, p] - tmp) / img_spacing[p]
                        dx[p] = x[p]


def _sparse_gradient_3d(floating[:, :, :] img,
                        double[:, :] img_world2grid,
                        double[:] img_spacing,
                        double[:, :] sample_points,
                        floating[:, :] out, int[:] inside):
    r""" Gradient of a 3D image evaluated at a set of points in physical space

    For each row (x_i, y_i, z_i) in sample_points, the image is interpolated at

    P1=(x_i + h, y_i, z_i), Q1=(x_i - h, y_i, z_i)
    P2=(x_i, y_i + h, z_i), Q2=(x_i, y_i - h, z_i)
    P3=(x_i, y_i, z_i + h), Q3=(x_i, y_i, z_i - h)

    (by mapping Pi and Qi to the grid using img_world2grid: the inverse of the
    grid-to-space transform of img). The displacement parameter h is of
    magnitude 0.5 (in physical space units), therefore the approximated partial
    derivatives are given by the difference between the image interpolated at
    Pi and Qi.

    Parameters
    ----------
    img : array, shape (S, R, C)
        the input volume whose gradient will be computed
    img_world2grid : array, shape (4, 4)
        the space-to-grid transform matrix associated to img
    img_spacing : array, shape (3,)
        the spacing between voxels (voxel size along each axis) of the input
    sample_points: array, shape (n, 3)
        list of points where the derivative will be evaluated
        (one point per row)
    out : array, shape (n, 3)
        the buffer in which to store the image gradient
    """
    cdef:
        cnp.npy_intp n = sample_points.shape[0]
        cnp.npy_intp i, in_flag
        double tmp
        double[:] dx = np.empty(shape=(3,), dtype=np.float64)
        double[:] h = np.empty(shape=(3,), dtype=np.float64)
        double[:] q = np.empty(shape=(3,), dtype=np.float64)
    with nogil:
        h[0] = 0.5 * img_spacing[0]
        h[1] = 0.5 * img_spacing[1]
        h[2] = 0.5 * img_spacing[2]
        for i in range(n):
            inside[i] = 1
            dx[0] = sample_points[i, 0]
            dx[1] = sample_points[i, 1]
            dx[2] = sample_points[i, 2]
            for p in range(3):
                # Compute coordinates of point dx on img's grid
                dx[p] = sample_points[i, p] - h[p]
                q[0] = _apply_affine_3d_x0(dx[0], dx[1], dx[2], 1,
                                           img_world2grid)
                q[1] = _apply_affine_3d_x1(dx[0], dx[1], dx[2], 1,
                                           img_world2grid)
                q[2] = _apply_affine_3d_x2(dx[0], dx[1], dx[2], 1,
                                           img_world2grid)
                # Interpolate img at q
                in_flag = _interpolate_scalar_3d[floating](img, q[0], q[1],
                    q[2], &out[i, p])
                if in_flag == 0:
                    out[i, p] = 0
                    inside[i] = 0
                    continue
                tmp = out[i, p]
                # Compute coordinates of point dx on img's grid
                dx[p] = sample_points[i, p] + h[p]
                q[0] = _apply_affine_3d_x0(dx[0], dx[1], dx[2], 1,
                                           img_world2grid)
                q[1] = _apply_affine_3d_x1(dx[0], dx[1], dx[2], 1,
                                           img_world2grid)
                q[2] = _apply_affine_3d_x2(dx[0], dx[1], dx[2], 1,
                                           img_world2grid)
                # Interpolate img at q
                in_flag = _interpolate_scalar_3d[floating](img, q[0], q[1], q[2],
                                                 &out[i, p])
                if in_flag == 0:
                    out[i, p] = 0
                    inside[i] = 0
                    continue
                out[i, p] = (out[i, p] - tmp) / img_spacing[p]
                dx[p] = sample_points[i, p]


def _gradient_2d(floating[:, :] img, double[:, :] img_world2grid,
                 double[:] img_spacing, double[:, :] out_grid2world,
                 floating[:, :, :] out, int[:, :] inside):
    r""" Gradient of a 2D image in physical space coordinates

    Each grid cell (i, j) in the sampling grid (determined by
    out.shape) is mapped to its corresponding physical point (x, y) by
    multiplying out_grid2world (its grid-to-space transform) by (i, j), then
    the image is interpolated, at

    P1=(x + h, y), Q1=(x - h, y)
    P2=(x, y + h), Q2=(x, y - h)

    (by mapping Pi and Qi to the grid using img_world2grid: the inverse of the
    grid-to-space transform of img). The displacement parameter h is of
    magnitude 0.5 (in physical space units), therefore the approximated partial
    derivatives are given by the difference between the image interpolated at
    Pi and Qi.

    Parameters
    ----------
    img : array, shape (R, C)
        the input image whose gradient will be computed
    img_world2grid : array, shape (3, 3)
        the space-to-grid transform matrix associated to img
    img_spacing : array, shape (2,)
        the spacing between pixels (pixel size along each axis) of the input
        image
    out_grid2world : array, shape (3, 3)
        the grid-to-space transform associated to the sampling grid
    out : array, shape (S', R', 2)
        the buffer in which to store the image gradient
    inside : array, shape (S', R')
        the buffer in which to store the flags indicating whether the sample
        point lies inside (=1) or outside (=0) the image grid
    """
    cdef:
        cnp.npy_intp nrows = out.shape[0]
        cnp.npy_intp ncols = out.shape[1]
        cnp.npy_intp i, j, k, in_flag
        double tmp
        double[:] x = np.empty(shape=(2,), dtype=np.float64)
        double[:] dx = np.empty(shape=(2,), dtype=np.float64)
        double[:] h = np.empty(shape=(2,), dtype=np.float64)
        double[:] q = np.empty(shape=(2,), dtype=np.float64)
    with nogil:
        h[0] = 0.5 * img_spacing[0]
        h[1] = 0.5 * img_spacing[1]
        for i in range(nrows):
            for j in range(ncols):
                inside[i, j] = 1
                # Compute coordinates of index (i, j) in physical space
                x[0] = _apply_affine_2d_x0(i, j, 1, out_grid2world)
                x[1] = _apply_affine_2d_x1(i, j, 1, out_grid2world)
                dx[:] = x[:]
                for p in range(2):
                    # Compute coordinates of point dx on img's grid
                    dx[p] = x[p] - h[p]
                    q[0] = _apply_affine_2d_x0(dx[0], dx[1], 1, img_world2grid)
                    q[1] = _apply_affine_2d_x1(dx[0], dx[1], 1, img_world2grid)
                    # Interpolate img at q
                    in_flag = _interpolate_scalar_2d[floating](img, q[0],
                        q[1], &out[i, j, p])
                    if in_flag == 0:
                        out[i, j, p] = 0
                        inside[i, j] = 0
                        continue
                    tmp = out[i, j, p]
                    # Compute coordinates of point dx on img's grid
                    dx[p] = x[p] + h[p]
                    q[0] = _apply_affine_2d_x0(dx[0], dx[1], 1, img_world2grid)
                    q[1] = _apply_affine_2d_x1(dx[0], dx[1], 1, img_world2grid)
                    # Interpolate img at q
                    in_flag = _interpolate_scalar_2d[floating](img, q[0],
                        q[1], &out[i, j, p])
                    if in_flag == 0:
                        out[i, j, p] = 0
                        inside[i, j] = 0
                        continue
                    out[i, j, p] = (out[i, j, p] - tmp) / img_spacing[p]
                    dx[p] = x[p]


def _sparse_gradient_2d(floating[:, :] img, double[:, :] img_world2grid,
                        double[:] img_spacing,
                        double[:, :] sample_points,
                        floating[:, :] out, int[:] inside):
    r""" Gradient of a 2D image evaluated at a set of points in physical space

    For each row (x_i, y_i) in sample_points, the image is interpolated at

    P1=(x_i + h, y_i), Q1=(x_i - h, y_i)
    P2=(x_i, y_i + h), Q2=(x_i, y_i - h)

    (by mapping Pi and Qi to the grid using img_world2grid: the inverse of the
    grid-to-space transform of img). The displacement parameter h is of
    magnitude 0.5 (in physical space units), therefore the approximated partial
    derivatives are given by the difference between the image interpolated at
    Pi and Qi.

    Parameters
    ----------
    img : array, shape (R, C)
        the input volume whose gradient will be computed
    img_world2grid : array, shape (3, 3)
        the space-to-grid transform matrix associated to img
    img_spacing : array, shape (2,)
        the spacing between pixels (pixel size along each axis) of the input
    sample_points: array, shape (n, 2)
        list of points where the derivative will be evaluated
        (one point per row)
    out : array, shape (n, 2)
        the buffer in which to store the image gradient
    inside : array, shape (n,)
        the buffer in which to store the flags indicating whether the sample
        point lies inside (=1) or outside (=0) the image grid
    """
    cdef:
        cnp.npy_intp n = sample_points.shape[0]
        cnp.npy_intp i, in_flag
        double tmp
        double[:] dx = np.empty(shape=(2,), dtype=np.float64)
        double[:] h = np.empty(shape=(2,), dtype=np.float64)
        double[:] q = np.empty(shape=(2,), dtype=np.float64)
    with nogil:
        h[0] = 0.5 * img_spacing[0]
        h[1] = 0.5 * img_spacing[1]
        for i in range(n):
            inside[i] = 1
            dx[0] = sample_points[i, 0]
            dx[1] = sample_points[i, 1]
            for p in range(2):
                # Compute coordinates of point dx on img's grid
                dx[p] = sample_points[i, p] - h[p]
                q[0] = _apply_affine_2d_x0(dx[0], dx[1], 1, img_world2grid)
                q[1] = _apply_affine_2d_x1(dx[0], dx[1], 1, img_world2grid)
                # Interpolate img at q
                in_flag = _interpolate_scalar_2d[floating](img, q[0], q[1],
                                                           &out[i, p])
                if in_flag == 0:
                    out[i, p] = 0
                    inside[i] = 0
                    continue
                tmp = out[i, p]
                # Compute coordinates of point dx on img's grid
                dx[p] = sample_points[i, p] + h[p]
                q[0] = _apply_affine_2d_x0(dx[0], dx[1], 1, img_world2grid)
                q[1] = _apply_affine_2d_x1(dx[0], dx[1], 1, img_world2grid)
                # Interpolate img at q
                in_flag = _interpolate_scalar_2d[floating](img, q[0], q[1],
                                                           &out[i, p])
                if in_flag == 0:
                    out[i, p] = 0
                    inside[i] = 0
                    continue
                out[i, p] = (out[i, p] - tmp) / img_spacing[p]
                dx[p] = sample_points[i, p]


def gradient(img, img_world2grid, img_spacing, out_shape,
             out_grid2world):
    r""" Gradient of an image in physical space

    Parameters
    ----------
    img : 2D or 3D array, shape (R, C) or (S, R, C)
        the input image whose gradient will be computed
    img_world2grid : array, shape (dim+1, dim+1)
        the space-to-grid transform matrix associated to img
    img_spacing : array, shape (dim,)
        the spacing between voxels (voxel size along each axis) of the input
        image
    out_shape : array, shape (dim,)
        the number of (slices), rows and columns of the sampling grid
    out_grid2world : array, shape (dim+1, dim+1)
        the grid-to-space transform associated to the sampling grid

    Returns
    -------
    out : array, shape (R', C', 2) or (S', R', C', 3)
        the buffer in which to store the image gradient, where
        (S'), R', C' are given by out_shape
    """
    dim = len(img.shape)
    if not is_valid_affine(img_world2grid, dim):
        raise ValueError("Invalid image affine transform")
    if not is_valid_affine(out_grid2world, dim):
        raise ValueError("Invalid sampling grid affine transform")
    if len(img_spacing) < dim:
        raise ValueError("Invalid spacings")
    ftype = img.dtype.type
    out = np.empty(tuple(out_shape)+(dim,), dtype=ftype)
    inside = np.empty(tuple(out_shape), dtype=np.int32)
    # Select joint density gradient 2D or 3D
    if dim == 2:
        jd_grad = _gradient_2d
    elif dim == 3:
        jd_grad = _gradient_3d
    else:
        raise ValueError('Undefined gradient for image dimension %d' % (dim,))
    if img_world2grid.dtype != np.float64:
        img_world2grid = img_world2grid.astype(np.float64)
    if img_spacing.dtype != np.float64:
        img_spacing = img_spacing.astype(np.float64)
    if out_grid2world.dtype != np.float64:
        out_grid2world = out_grid2world.astype(np.float64)
    jd_grad(img, img_world2grid, img_spacing, out_grid2world, out, inside)
    return np.asarray(out), np.asarray(inside)


def sparse_gradient(img, img_world2grid, img_spacing, sample_points):
    r""" Gradient of an image in physical space

    Parameters
    ----------
    img : 2D or 3D array, shape (R, C) or (S, R, C)
        the input image whose gradient will be computed
    img_world2grid : array, shape (dim+1, dim+1)
        the space-to-grid transform matrix associated to img
    img_spacing : array, shape (dim,)
        the spacing between voxels (voxel size along each axis) of the input
        image
    sample_points: array, shape (n, dim)
        list of points where the derivative will be evaluated
        (one point per row)

    Returns
    -------
    out : array, shape (n, dim)
        the gradient at each point stored at its corresponding row
    """
    dim = len(img.shape)
    if not is_valid_affine(img_world2grid, dim):
        raise ValueError("Invalid affine transform matrix")
    if len(img_spacing) < dim:
        raise ValueError("Invalid spacings")

    ftype = img.dtype.type
    n = sample_points.shape[0]
    out = np.empty(shape=(n, dim), dtype=ftype)
    inside = np.empty(shape=(n,), dtype=np.int32)
    # Select joint density gradient 2D or 3D
    if dim == 2:
        jd_grad = _sparse_gradient_2d
    else:
        jd_grad = _sparse_gradient_3d
    if img_world2grid.dtype != np.float64:
        img_world2grid = img_world2grid.astype(np.float64)
    if img_spacing.dtype != np.float64:
        img_spacing = img_spacing.astype(np.float64)
    jd_grad(img, img_world2grid, img_spacing, sample_points, out, inside)
    return np.asarray(out), np.asarray(inside)
