""" Utility functions used by the Sum of Squared Differences (SSD) metric """

import numpy as np
cimport cython
cimport numpy as cnp
from dipy.align.fused_types cimport floating
cdef extern from "dpy_math.h" nogil:
    int dpy_isinf(double)
    double sqrt(double)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _solve_2d_symmetric_positive_definite(double* A, double* y,
                                                double det,
                                                double* out) noexcept nogil:
    r"""Solves a 2-variable symmetric positive-definite linear system

    The C implementation of the public-facing Python function
    ``solve_2d_symmetric_positive_definite``.

    Solves the symmetric positive-definite linear system $Mx = y$ given by::

        M = [[A[0], A[1]],
             [A[1], A[2]]]

    Parameters
    ----------
    A : array, shape (3,)
        the array containing the entries of the symmetric 2x2 matrix
    y : array, shape (2,)
        right-hand side of the system to be solved
    out : array, shape (2,)
        the array the output will be stored in
    """
    out[1] = (A[0] * y[1] - A[1] * y[0]) / det
    out[0] = (y[0] - A[1] * out[1]) / A[0]


def solve_2d_symmetric_positive_definite(A, y, double det):
    r"""Solves a 2-variable symmetric positive-definite linear system

    Solves the symmetric positive-definite linear system $Mx = y$ given by::

        M = [[A[0], A[1]],
             [A[1], A[2]]]

    Parameters
    ----------
    A : array, shape (3,)
        the array containing the entries of the symmetric 2x2 matrix
    y : array, shape (2,)
        right-hand side of the system to be solved

    Returns
    -------
    out : array, shape (2,)
        the array the output will be stored in
    """
    cdef:
        cnp.ndarray out = np.zeros(2, dtype=float)

    _solve_2d_symmetric_positive_definite(
        <double*> cnp.PyArray_DATA(np.ascontiguousarray(A, float)),
        <double*> cnp.PyArray_DATA(np.ascontiguousarray(y, float)),
        det,
        <double*> cnp.PyArray_DATA(out))
    return np.asarray(out)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int _solve_3d_symmetric_positive_definite(double* g,
                                               double* y,
                                               double tau,
                                               double* out) nogil:
    r"""Solves a 3-variable symmetric positive-definite linear system

    Solves the symmetric semi-positive-definite linear system $Mx = y$ given by
    $M = (g g^{T} + \tau I)$

    The C implementation of the public-facing Python function
    ``solve_3d_symmetric_positive_definite``.

    Parameters
    ----------
    g : array, shape (3,)
        the vector in the outer product above
    y : array, shape (3,)
        right-hand side of the system to be solved
    tau : double
        $\tau$ in $M = (g g^{T} + \tau I)$
    out : array, shape (3,)
        the array the output will be stored in

    Returns
    -------
    is_singular : int
        1 if M is singular, otherwise 0
    """
    cdef:
        double a,b,c,d,e,f, y0, y1, y2, sub_det
    a = g[0] ** 2 + tau
    if a < 1e-9:
        return 1
    b = g[0] * g[1]
    sub_det = (a * (g[1] ** 2 + tau) - b * b)
    if sub_det < 1e-9:
        return 1
    c = g[0] * g[2]
    d = (a * (g[1] ** 2 + tau) - b * b) / a
    e = (a * (g[1] * g[2]) - b * c) / a
    f = (a * (g[2] ** 2 + tau) - c * c) / a - (e * e * a) / sub_det
    if f < 1e-9:
        return 1
    y0 = y[0]
    y1 = (y[1] * a - y0 * b) / a
    y2 = (y[2] * a - c * y0) / a - (e * (y[1] * a - b * y0)) / sub_det
    out[2] = y2 / f
    out[1] = (y1 - e * out[2]) / d
    out[0] = (y0 - b * out[1] - c * out[2]) / a
    return 0


def solve_3d_symmetric_positive_definite(g, y, double tau):
    r"""Solves a 3-variable symmetric positive-definite linear system

    Solves the symmetric semi-positive-definite linear system $Mx = y$ given by
    $M = (g g^{T} + \tau I)$.

    Parameters
    ----------
    g : array, shape (3,)
        the vector in the outer product above
    y : array, shape (3,)
        right-hand side of the system to be solved
    tau : double
        $\tau$ in $M = (g g^{T} + \tau I)$

    Returns
    -------
    out : array, shape (3,)
        the array the output will be stored in
    is_singular : int
        1 if M is singular, otherwise 0
    """
    cdef:
        cnp.ndarray out = np.zeros(3, dtype=float)
        int is_singular
    is_singular = _solve_3d_symmetric_positive_definite(
        <double*> cnp.PyArray_DATA(np.ascontiguousarray(g, float)),
        <double*> cnp.PyArray_DATA(np.ascontiguousarray(y, float)),
        tau,
        <double*> cnp.PyArray_DATA(out))
    return np.asarray(out), is_singular


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double iterate_residual_displacement_field_ssd_2d(
                floating[:, :] delta_field, floating[:, :] sigmasq_field,
                floating[:, :, :] grad, floating[:, :, :] target,
                double lambda_param, floating[:, :, :] displacement_field):
    r"""One iteration of a large linear system solver for 2D SSD registration

    Performs one iteration at one level of the Multi-resolution Gauss-Seidel
    solver proposed by :footcite:t:`Bruhn2005`.

    Parameters
    ----------
    delta_field : array, shape (R, C)
        the difference between the static and moving image (the 'derivative
        w.r.t. time' in the optical flow model)
    sigmasq_field : array, shape (R, C)
        the variance of the gray level value at each voxel, according to the
        EM model (for SSD, it is 1 for all voxels). Inf and 0 values
        are processed specially to support infinite and zero variance.
    grad : array, shape (R, C, 2)
        the gradient of the moving image
    target : array, shape (R, C, 2)
        right-hand side of the linear system to be solved in the Weickert's
        multi-resolution algorithm
    lambda_param : float
        smoothness parameter of the objective function
    displacement_field : array, shape (R, C, 2)
        current displacement field to start the iteration from

    Returns
    -------
    max_displacement : float
        the norm of the maximum change in the displacement field after the
        iteration

    References
    ----------
    .. footbibliography::
    """
    ftype = np.asarray(delta_field).dtype
    cdef:
        int NUM_NEIGHBORS = 4
        int* dRow = [-1, 0, 1,  0]
        int* dCol = [0, 1, 0, -1]
        cnp.npy_intp nrows = delta_field.shape[0]
        cnp.npy_intp ncols = delta_field.shape[1]
        cnp.npy_intp r, c, dr, dc, nn, k

        double* b = [0, 0]
        double* d = [0, 0]
        double* y = [0, 0]
        double* A = [0, 0, 0]

        double xx, yy, opt, nrm2, delta, sigmasq, max_displacement, det

    max_displacement = 0

    with nogil:

        for r in range(nrows):
            for c in range(ncols):
                delta = delta_field[r, c]
                sigmasq = sigmasq_field[r, c] if sigmasq_field is not None else 1
                if target is None:
                    b[0] = delta_field[r, c] * grad[r, c, 0]
                    b[1] = delta_field[r, c] * grad[r, c, 1]
                else:
                    b[0] = target[r, c, 0]
                    b[1] = target[r, c, 1]
                nn = 0
                y[0] = 0
                y[1] = 0
                for k in range(NUM_NEIGHBORS):
                    dr = r + dRow[k]
                    if dr < 0 or dr >= nrows:
                        continue
                    dc = c + dCol[k]
                    if dc < 0 or dc >= ncols:
                        continue
                    nn += 1
                    y[0] += displacement_field[dr, dc, 0]
                    y[1] += displacement_field[dr, dc, 1]
                if dpy_isinf(sigmasq) != 0:
                    xx = displacement_field[r, c, 0]
                    yy = displacement_field[r, c, 1]
                    displacement_field[r, c, 0] = y[0] / nn
                    displacement_field[r, c, 1] = y[1] / nn
                    xx -= displacement_field[r, c, 0]
                    yy -= displacement_field[r, c, 1]
                    opt = xx * xx + yy * yy
                    if max_displacement < opt:
                        max_displacement = opt
                else:
                    A[0] = grad[r, c, 0] ** 2 + sigmasq * lambda_param * nn
                    A[1] = grad[r, c, 0] * grad[r, c, 1]
                    A[2] = grad[r, c, 1] ** 2 + sigmasq * lambda_param * nn
                    det = A[0] * A[2] - A[1] * A[1]
                    if det < 1e-9:
                        nrm2 = (grad[r, c, 0] ** 2 +
                                grad[r, c, 1] ** 2)
                        if nrm2 < 1e-9:
                            displacement_field[r, c, 0] = 0
                            displacement_field[r, c, 1] = 0
                        else:
                            displacement_field[r, c, 0] = (b[0]) / nrm2
                            displacement_field[r, c, 1] = (b[1]) / nrm2
                    else:
                        y[0] = b[0] + sigmasq * lambda_param * y[0]
                        y[1] = b[1] + sigmasq * lambda_param * y[1]
                        _solve_2d_symmetric_positive_definite(A, y, det, d)
                        xx = displacement_field[r, c, 0] - d[0]
                        yy = displacement_field[r, c, 1] - d[1]
                        displacement_field[r, c, 0] = d[0]
                        displacement_field[r, c, 1] = d[1]
                        opt = xx * xx + yy * yy
                        if max_displacement < opt:
                            max_displacement = opt
    return sqrt(max_displacement)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double compute_energy_ssd_2d(floating[:, :] delta_field):
    r"""Sum of squared differences between two 2D images

    Computes the Sum of Squared Differences between the static and moving image.
    Those differences are given by delta_field

    Parameters
    ----------
    delta_field : array, shape (R, C)
        the difference between the static and moving image (the 'derivative
        w.r.t. time' in the optical flow model)

    Returns
    -------
    energy : float
        the SSD energy at this iteration

    Notes
    -----
    The numeric value of the energy is used only to detect convergence.
    This function returns only the energy corresponding to the data term
    (excluding the energy corresponding to the regularization term) because
    the Greedy-SyN algorithm is an unconstrained gradient descent algorithm
    in the space of diffeomorphisms: in each iteration it makes a step
    along the negative smoothed gradient --of the data term-- and then makes
    sure the resulting diffeomorphisms are invertible using an explicit
    inversion algorithm. Since it is not clear how to reflect the energy
    corresponding to this re-projection to the space of diffeomorphisms,
    a more precise energy computation including the regularization term
    is useless. Instead, convergence is checked considering the data-term
    energy only and detecting oscilations in the energy profile.

    """
    cdef:
        cnp.npy_intp nrows = delta_field.shape[0]
        cnp.npy_intp ncols = delta_field.shape[1]
        cnp.npy_intp r, c
        double energy = 0

    with nogil:
        for r in range(nrows):
            for c in range(ncols):
                energy += delta_field[r, c] ** 2
    return energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double iterate_residual_displacement_field_ssd_3d(
                floating[:, :, :] delta_field, floating[:, :, :] sigmasq_field,
                floating[:, :, :, :] grad, floating[:, :, :, :] target,
                double lambda_param, floating[:, :, :, :] disp):
    r"""One iteration of a large linear system solver for 3D SSD registration

    Performs one iteration at one level of the Multi-resolution Gauss-Seidel
    solver proposed by :footcite:t:`Bruhn2005`.

    Parameters
    ----------
    delta_field : array, shape (S, R, C)
        the difference between the static and moving image (the 'derivative
        w.r.t. time' in the optical flow model)
    sigmasq_field : array, shape (S, R, C)
        the variance of the gray level value at each voxel, according to the
        EM model (for SSD, it is 1 for all voxels). Inf and 0 values
        are processed specially to support infinite and zero variance.
    grad : array, shape (S, R, C, 3)
        the gradient of the moving image
    target : array, shape (S, R, C, 3)
        right-hand side of the linear system to be solved in the Weickert's
        multi-resolution algorithm
    lambda_param : float
        smoothness parameter of the objective function
    disp : array, shape (S, R, C, 3)
        the displacement field to start the optimization from

    Returns
    -------
    max_displacement : float
        the norm of the maximum change in the displacement field after the
        iteration

    References
    ----------
    .. footbibliography::
    """
    ftype = np.asarray(delta_field).dtype
    cdef:
        int NUM_NEIGHBORS = 6
        int* dSlice = [-1, 0, 0, 0,  0, 1]
        int* dRow = [0, -1, 0, 1,  0, 0]
        int* dCol = [0,  0, 1, 0, -1, 0]
        cnp.npy_intp nslices = delta_field.shape[0]
        cnp.npy_intp nrows = delta_field.shape[1]
        cnp.npy_intp ncols = delta_field.shape[2]
        int nn
        double* g = [0, 0, 0]
        double* b = [0, 0, 0]
        double* d = [0, 0, 0]
        double* y = [0, 0, 0]
        double* A = [0, 0, 0, 0, 0, 0]
        double xx, yy, zz, opt, nrm2, delta, sigmasq, max_displacement
        cnp.npy_intp dr, ds, dc, s, r, c
    max_displacement = 0

    with nogil:

        for s in range(nslices):
            for r in range(nrows):
                for c in range(ncols):
                    g[0] = grad[s, r, c, 0]
                    g[1] = grad[s, r, c, 1]
                    g[2] = grad[s, r, c, 2]
                    delta = delta_field[s, r, c]
                    sigmasq = sigmasq_field[s, r, c] if sigmasq_field is not None else 1
                    if target is None:
                        b[0] = delta_field[s, r, c] * g[0]
                        b[1] = delta_field[s, r, c] * g[1]
                        b[2] = delta_field[s, r, c] * g[2]
                    else:
                        b[0] = target[s, r, c, 0]
                        b[1] = target[s, r, c, 1]
                        b[2] = target[s, r, c, 2]
                    nn = 0
                    y[0] = 0
                    y[1] = 0
                    y[2] = 0
                    for k in range(NUM_NEIGHBORS):
                        ds = s + dSlice[k]
                        if ds < 0 or ds >= nslices:
                            continue
                        dr = r + dRow[k]
                        if dr < 0 or dr >= nrows:
                            continue
                        dc = c + dCol[k]
                        if dc < 0 or dc >= ncols:
                            continue
                        nn += 1
                        y[0] += disp[ds, dr, dc, 0]
                        y[1] += disp[ds, dr, dc, 1]
                        y[2] += disp[ds, dr, dc, 2]
                    if dpy_isinf(sigmasq) != 0:
                        xx = disp[s, r, c, 0]
                        yy = disp[s, r, c, 1]
                        zz = disp[s, r, c, 2]
                        disp[s, r, c, 0] = y[0] / nn
                        disp[s, r, c, 1] = y[1] / nn
                        disp[s, r, c, 2] = y[2] / nn
                        xx -= disp[s, r, c, 0]
                        yy -= disp[s, r, c, 1]
                        zz -= disp[s, r, c, 2]
                        opt = xx * xx + yy * yy + zz * zz
                        if max_displacement < opt:
                            max_displacement = opt
                    elif sigmasq < 1e-9:
                            nrm2 = g[0] ** 2 + g[1] ** 2 + g[2] ** 2
                            if nrm2 < 1e-9:
                                disp[s, r, c, 0] = 0
                                disp[s, r, c, 1] = 0
                                disp[s, r, c, 2] = 0
                            else:
                                disp[s, r, c, 0] = (b[0]) / nrm2
                                disp[s, r, c, 1] = (b[1]) / nrm2
                                disp[s, r, c, 2] = (b[2]) / nrm2
                    else:
                        tau = sigmasq * lambda_param * nn
                        y[0] = b[0] + sigmasq * lambda_param * y[0]
                        y[1] = b[1] + sigmasq * lambda_param * y[1]
                        y[2] = b[2] + sigmasq * lambda_param * y[2]
                        is_singular = _solve_3d_symmetric_positive_definite(
                                                                g, y, tau, d)
                        if is_singular == 1:
                            nrm2 = g[0] ** 2 + g[1] ** 2 + g[2] ** 2
                            if nrm2 < 1e-9:
                                disp[s, r, c, 0] = 0
                                disp[s, r, c, 1] = 0
                                disp[s, r, c, 2] = 0
                            else:
                                disp[s, r, c, 0] = (b[0]) / nrm2
                                disp[s, r, c, 1] = (b[1]) / nrm2
                                disp[s, r, c, 2] = (b[2]) / nrm2
                        xx = disp[s, r, c, 0] - d[0]
                        yy = disp[s, r, c, 1] - d[1]
                        zz = disp[s, r, c, 2] - d[2]
                        disp[s, r, c, 0] = d[0]
                        disp[s, r, c, 1] = d[1]
                        disp[s, r, c, 2] = d[2]
                        opt = xx * xx + yy * yy + zz * zz
                        if max_displacement < opt:
                            max_displacement = opt
    return sqrt(max_displacement)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double compute_energy_ssd_3d(floating[:, :, :] delta_field):
    r"""Sum of squared differences between two 3D volumes

    Computes the Sum of Squared Differences between the static and moving volume
    Those differences are given by delta_field

    Parameters
    ----------
    delta_field : array, shape (R, C)
        the difference between the static and moving image (the 'derivative
        w.r.t. time' in the optical flow model)

    Returns
    -------
    energy : float
        the SSD energy at this iteration

    Notes
    -----
    The numeric value of the energy is used only to detect convergence.
    This function returns only the energy corresponding to the data term
    (excluding the energy corresponding to the regularization term) because
    the Greedy-SyN algorithm is an unconstrained gradient descent algorithm
    in the space of diffeomorphisms: in each iteration it makes a step
    along the negative smoothed gradient --of the data term-- and then makes
    sure the resulting diffeomorphisms are invertible using an explicit
    inversion algorithm. Since it is not clear how to reflect the energy
    corresponding to this re-projection to the space of diffeomorphisms,
    a more precise energy computation including the regularization term
    is useless. Instead, convergence is checked considering the data-term
    energy only and detecting oscilations in the energy profile.
    """
    cdef:
        cnp.npy_intp nslices = delta_field.shape[0]
        cnp.npy_intp nrows = delta_field.shape[1]
        cnp.npy_intp ncols = delta_field.shape[2]
        cnp.npy_intp s, r, c
        double energy = 0
    with nogil:
        for s in range(nslices):
            for r in range(nrows):
                for c in range(ncols):
                    energy += delta_field[s, r, c] ** 2
    return energy


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_residual_displacement_field_ssd_3d(
        floating[:, :, :] delta_field, floating[:, :, :] sigmasq_field,
        floating[:, :, :, :] gradient_field, floating[:, :, :, :] target,
        double lambda_param, floating[:, :, :, :] disp,
        floating[:, :, :, :] residual):
    r"""The residual displacement field to be fit on the next iteration

    Computes the residual displacement field corresponding to the current
    displacement field (given by 'disp') in the Multi-resolution
    Gauss-Seidel solver proposed by :footcite:t:`Bruhn2005`.

    Parameters
    ----------
    delta_field : array, shape (S, R, C)
        the difference between the static and moving image (the 'derivative
        w.r.t. time' in the optical flow model)
    sigmasq_field : array, shape (S, R, C)
        the variance of the gray level value at each voxel, according to the
        EM model (for SSD, it is 1 for all voxels). Inf and 0 values
        are processed specially to support infinite and zero variance.
    gradient_field : array, shape (S, R, C, 3)
        the gradient of the moving image
    target : array, shape (S, R, C, 3)
        right-hand side of the linear system to be solved in the Weickert's
        multi-resolution algorithm
    lambda_param : float
        smoothness parameter in the objective function
    disp : array, shape (S, R, C, 3)
        the current displacement field to compute the residual from
    residual : array, shape (S, R, C, 3)
        the displacement field to put the residual to

    Returns
    -------
    residual : array, shape (S, R, C, 3)
        the residual displacement field. If residual was None a input, then
        a new field is returned, otherwise the same array is returned

    References
    ----------
    .. footbibliography::
    """
    ftype = np.asarray(delta_field).dtype
    cdef:
        int NUM_NEIGHBORS = 6
        int* dSlice = [-1,  0, 0, 0,  0, 1]
        int* dRow = [0, -1, 0, 1,  0, 0]
        int* dCol = [0,  0, 1, 0, -1, 0]
        double* b = [0, 0, 0]
        double* y = [0, 0, 0]

        cnp.npy_intp nslices = delta_field.shape[0]
        cnp.npy_intp nrows = delta_field.shape[1]
        cnp.npy_intp ncols = delta_field.shape[2]
        double delta, sigmasq, dotP
        cnp.npy_intp s, r, c, ds, dr, dc
    if residual is None:
        residual = np.empty(shape=(nslices, nrows, ncols, 3), dtype=ftype)

    with nogil:

        for s in range(nslices):
            for r in range(nrows):
                for c in range(ncols):
                    delta = delta_field[s, r, c]
                    sigmasq = sigmasq_field[s, r, c] if sigmasq_field is not None else 1
                    if target is None:
                        b[0] = delta * gradient_field[s, r, c, 0]
                        b[1] = delta * gradient_field[s, r, c, 1]
                        b[2] = delta * gradient_field[s, r, c, 2]
                    else:
                        b[0] = target[s, r, c, 0]
                        b[1] = target[s, r, c, 1]
                        b[2] = target[s, r, c, 2]
                    y[0] = 0
                    y[1] = 0
                    y[2] = 0
                    for k in range(NUM_NEIGHBORS):
                        ds = s + dSlice[k]
                        if ds < 0 or ds >= nslices:
                            continue
                        dr = r + dRow[k]
                        if dr < 0 or dr >= nrows:
                            continue
                        dc = c + dCol[k]
                        if dc < 0 or dc >= ncols:
                            continue
                        y[0] += (disp[s, r, c, 0] - disp[ds, dr, dc, 0])
                        y[1] += (disp[s, r, c, 1] - disp[ds, dr, dc, 1])
                        y[2] += (disp[s, r, c, 2] - disp[ds, dr, dc, 2])
                    if dpy_isinf(sigmasq) != 0:
                        residual[s, r, c, 0] = -lambda_param * y[0]
                        residual[s, r, c, 1] = -lambda_param * y[1]
                        residual[s, r, c, 2] = -lambda_param * y[2]
                    else:
                        dotP = (gradient_field[s, r, c, 0] * disp[s, r, c, 0] +
                                gradient_field[s, r, c, 1] * disp[s, r, c, 1] +
                                gradient_field[s, r, c, 2] * disp[s, r, c, 2])
                        residual[s, r, c, 0] = (b[0] -
                                                (gradient_field[s, r, c, 0] * dotP +
                                                 sigmasq * lambda_param * y[0]))
                        residual[s, r, c, 1] = (b[1] -
                                                (gradient_field[s, r, c, 1] * dotP +
                                                 sigmasq * lambda_param * y[1]))
                        residual[s, r, c, 2] = (b[2] -
                                                (gradient_field[s, r, c, 2] * dotP +
                                                 sigmasq * lambda_param * y[2]))
    return np.asarray(residual)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_residual_displacement_field_ssd_2d(
        floating[:, :] delta_field, floating[:, :] sigmasq_field,
        floating[:, :, :] gradient_field, floating[:, :, :] target,
        double lambda_param, floating[:, :, :] d,
        floating[:, :, :] residual):
    r"""The residual displacement field to be fit on the next iteration

    Computes the residual displacement field corresponding to the current
    displacement field in the Multi-resolution Gauss-Seidel solver proposed by
    :footcite:t:`Bruhn2005`.

    Parameters
    ----------
    delta_field : array, shape (R, C)
        the difference between the static and moving image (the 'derivative
        w.r.t. time' in the optical flow model)
    sigmasq_field : array, shape (R, C)
        the variance of the gray level value at each voxel, according to the
        EM model (for SSD, it is 1 for all voxels). Inf and 0 values
        are processed specially to support infinite and zero variance.
    gradient_field : array, shape (R, C, 2)
        the gradient of the moving image
    target : array, shape (R, C, 2)
        right-hand side of the linear system to be solved in the Weickert's
        multi-resolution algorithm
    lambda_param : float
        smoothness parameter in the objective function
    d : array, shape (R, C, 2)
        the current displacement field to compute the residual from
    residual : array, shape (R, C, 2)
        the displacement field to put the residual to

    Returns
    -------
    residual : array, shape (R, C, 2)
        the residual displacement field. If residual was None a input, then
        a new field is returned, otherwise the same array is returned

    References
    ----------
    .. footbibliography::
    """
    ftype = np.asarray(delta_field).dtype
    cdef:
        int NUM_NEIGHBORS = 4
        int* dRow = [-1, 0, 1,  0]
        int* dCol = [0, 1, 0, -1]
        double* b = [0, 0]
        double* y = [0, 0]

        cnp.npy_intp nrows = delta_field.shape[0]
        cnp.npy_intp ncols = delta_field.shape[1]
        double delta, sigmasq, dotP
        cnp.npy_intp r, c, dr, dc
    if residual is None:
        residual = np.empty(shape=(nrows, ncols, 2), dtype=ftype)

    with nogil:

        for r in range(nrows):
            for c in range(ncols):
                delta = delta_field[r, c]
                sigmasq = sigmasq_field[r, c] if sigmasq_field is not None else 1
                if target is None:
                    b[0] = delta * gradient_field[r, c, 0]
                    b[1] = delta * gradient_field[r, c, 1]
                else:
                    b[0] = target[r, c, 0]
                    b[1] = target[r, c, 1]
                y[0] = 0  # reset y
                y[1] = 0
                nn=0
                for k in range(NUM_NEIGHBORS):
                    dr = r + dRow[k]
                    if dr < 0 or dr >= nrows:
                        continue
                    dc = c + dCol[k]
                    if dc < 0 or dc >= ncols:
                        continue
                    y[0] += (d[r, c, 0] - d[dr, dc, 0])
                    y[1] += (d[r, c, 1] - d[dr, dc, 1])

                if dpy_isinf(sigmasq) != 0:
                    residual[r, c, 0] = -lambda_param * y[0]
                    residual[r, c, 1] = -lambda_param * y[1]
                else:
                    dotP = (gradient_field[r, c, 0] * d[r, c, 0] +
                            gradient_field[r, c, 1] * d[r, c, 1])
                    residual[r, c, 0] = (b[0] -
                                         (gradient_field[r, c, 0] * dotP +
                                          sigmasq * lambda_param * y[0]))
                    residual[r, c, 1] = (b[1] -
                                         (gradient_field[r, c, 1] * dotP +
                                          sigmasq * lambda_param * y[1]))
    return np.asarray(residual)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_ssd_demons_step_2d(floating[:,:] delta_field,
                               floating[:,:,:] gradient_moving,
                               double sigma_sq_x,
                               floating[:,:,:] out):
    r"""Demons step for 2D SSD-driven registration

    Computes the demons step for SSD-driven registration
    ( eq. 4 in :footcite:p:`Bruhn2005` )

    Parameters
    ----------
    delta_field : array, shape (R, C)
        the difference between the static and moving image (the 'derivative
        w.r.t. time' in the optical flow model)
    gradient_field : array, shape (R, C, 2)
        the gradient of the moving image
    sigma_sq_x : float
        parameter controlling the amount of regularization. It corresponds to
        $\sigma_x^2$ in algorithm 1 of :footcite:t:`Vercauteren2009`.
    out : array, shape (R, C, 2)
        if None, a new array will be created to store the demons step. Otherwise
        the provided array will be used.

    Returns
    -------
    demons_step : array, shape (R, C, 2)
        the demons step to be applied for updating the current displacement
        field
    energy : float
        the current ssd energy (before applying the returned demons_step)

    References
    ----------
    .. footbibliography::
    """
    cdef:
        cnp.npy_intp nr = delta_field.shape[0]
        cnp.npy_intp nc = delta_field.shape[1]
        cnp.npy_intp i, j
        double delta, delta_2, nrm2, energy, den

    if out is None:
        out = np.zeros((nr, nc, 2), dtype=np.asarray(delta_field).dtype)

    with nogil:

        energy = 0
        for i in range(nr):
            for j in range(nc):
                delta = delta_field[i,j]
                delta_2 = delta**2
                energy += delta_2
                nrm2 = gradient_moving[i, j, 0]**2 + gradient_moving[i, j, 1]**2
                den = delta_2/sigma_sq_x + nrm2
                if den <1e-9:
                    out[i, j, 0] = 0
                    out[i, j, 1] = 0
                else:
                    out[i, j, 0] = delta * gradient_moving[i, j, 0] / den
                    out[i, j, 1] = delta * gradient_moving[i, j, 1] / den

    return np.asarray(out), energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_ssd_demons_step_3d(floating[:,:,:] delta_field,
                               floating[:,:,:,:] gradient_moving,
                               double sigma_sq_x,
                               floating[:,:,:,:] out):
    r"""Demons step for 3D SSD-driven registration

    Computes the demons step for SSD-driven registration
    ( eq. 4 in :footcite:p:`Bruhn2005` )

    Parameters
    ----------
    delta_field : array, shape (S, R, C)
        the difference between the static and moving image (the 'derivative
        w.r.t. time' in the optical flow model)
    gradient_field : array, shape (S, R, C, 2)
        the gradient of the moving image
    sigma_sq_x : float
        parameter controlling the amount of regularization. It corresponds to
        $\sigma_x^2$ in algorithm 1 of :footcite:t:`Vercauteren2009`.
    out : array, shape (S, R, C, 2)
        if None, a new array will be created to store the demons step. Otherwise
        the provided array will be used.

    Returns
    -------
    demons_step : array, shape (S, R, C, 3)
        the demons step to be applied for updating the current displacement
        field
    energy : float
        the current ssd energy (before applying the returned demons_step)

    References
    ----------
    .. footbibliography::
    """
    cdef:
        cnp.npy_intp ns = delta_field.shape[0]
        cnp.npy_intp nr = delta_field.shape[1]
        cnp.npy_intp nc = delta_field.shape[2]
        cnp.npy_intp i, j, k
        double delta, delta_2, nrm2, energy, den

    if out is None:
        out = np.zeros((ns, nr, nc, 3), dtype=np.asarray(delta_field).dtype)

    with nogil:

        energy = 0
        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    delta = delta_field[k,i,j]
                    delta_2 = delta**2
                    energy += delta_2
                    nrm2 = (gradient_moving[k, i, j, 0]**2 +
                            gradient_moving[k, i, j, 1]**2 +
                            gradient_moving[k, i, j, 2]**2)
                    den = delta_2/sigma_sq_x + nrm2
                    if den < 1e-9:
                        out[k, i, j, 0] = 0
                        out[k, i, j, 1] = 0
                        out[k, i, j, 2] = 0
                    else:
                        out[k, i, j, 0] = (delta *
                                           gradient_moving[k, i, j, 0] / den)
                        out[k, i, j, 1] = (delta *
                                           gradient_moving[k, i, j, 1] / den)
                        out[k, i, j, 2] = (delta *
                                           gradient_moving[k, i, j, 2] / den)
    return np.asarray(out), energy
