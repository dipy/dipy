import numpy as np
cimport cython
from fused_types cimport floating, number

cdef extern from "math.h":
    int isinf(double) nogil
    double sqrt(double x) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void solve2DSymmetricPositiveDefiniteSystem(floating[:] A, floating[:] y,
                                                 floating[:] out) nogil:
    r"""
    Solves the symmetric positive-definite linear system Mx = y given by
    M=[[A[0], A[1]],
       [A[1], A[2]]].
    Returns the result in out

    Parameters
    ----------
    A : array, shape (3,)
        the array containing the entries of the symmetric 2x2 matrix
    y : array, shape (2,)
        right-hand side of the system to be solved
    out : array, shape (2,)
        the array the output will be stored in
    """
    cdef double den = (A[0] * A[2] - A[1] * A[1])
    out[1] = (A[0] * y[1] - A[1] * y[0]) / den
    out[0] = (y[0] - A[1] * out[1]) / A[0]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void solve3DSymmetricPositiveDefiniteSystem(floating[:] A, floating[:] y,
                                                 floating[:] out) nogil:
    r"""
    Solves the symmetric positive-definite linear system Mx = y given by
    M=[[A[0], A[1], A[2]],
       [A[1], A[3], A[4]],
       [A[2], A[4], A[5]]].
    Returns the result in out

    Parameters
    ----------
    A : array, shape (6,)
        the array containing the entries of the symmetric 2x2 matrix
    y : array, shape (3,)
        right-hand side of the system to be solved
    out : array, shape (3,)
        the array the output will be stored in
    """
    cdef:
        double a = A[0]
        double b = A[1]
        double c = A[2]
        double d = (a * A[3] - b * b) / a
        double e = (a * A[4] - b * c) / a
        double f = (a * A[5] - c * c) / a - (e * e * a) / (a * A[3] - b * b)
        double y0 = y[0]
        double y1 = (y[1] * a - y0 * b) / a
        double y2 = (y[2] * a - A[2] * y0) / a - (e * (y[1] * a - b * y0)) / (a * A[3] - b * b)
    out[2] = y2 / f
    out[1] = (y1 - e * out[2]) / d
    out[0] = (y0 - b * out[1] - c * out[2]) / a


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double iterate_residual_displacement_field_SSD2D(floating[:, :] delta_field,
                                                       floating[:, :] sigma_field,
                                                       floating[:, :, :] gradient_field,
                                                       floating[:, :, :] target,
                                                       double lambda_param,
                                                       floating[:, :, :] displacement_field):
    r"""
    Performs one iteration at one level of the Multi-resolution Gauss-Seidel 
    solver proposed by Bruhn and Weickert[1].
    
    [1] Weickert, J. (2005). Towards Ultimate Motion Estimation : Combining
        Highest Accuracy with Real-Time Performance Faculty of Mathematics
        and Computer Science.

    Parameters
    ----------
    delta_field : array, shape (R, C)
        the difference between the static and moving image (the 'derivatice
        w.r.t. time' in the optical flow model)
    sigma_field : array, shape (R, C)
        the variance of the gray level value at each voxel, according to the 
        EM model (for SSD, it is 1 for all voxels). Inf and 0 values
        are processed specially to support infinite and zero variance.
    gradient_field : array, shape (R, C, 2)
        the gradient of the moving image
    target : array, shape (R, C, 2)
        right-hand side of the linear system to be solved in the Weickert's
        multiresolution algorithm
    lambda_param : float
        smoothness parameter of the objective function
    displacement_field : array, shape (R, C, 2)
        current displacement field to start the iteration from

    Returns
    -------
    max_displacement : float
        the norm of the maximum change in the displacement field after the 
        iteration
    """
    cdef:
        int NUM_NEIGHBORS = 4
        int[:] dRow = np.array([-1, 0, 1,  0], dtype=np.int32)
        int[:] dCol = np.array([0, 1, 0, -1], dtype=np.int32)
        int nrows = delta_field.shape[0]
        int ncols = delta_field.shape[1]
        int r, c, dr, dc, nn, k
        floating[:] b = np.ndarray(shape=(2,), dtype=np.asarray(delta_field).dtype)
        floating[:] d = np.ndarray(shape=(2,), dtype=np.asarray(delta_field).dtype)
        floating[:] y = np.ndarray(shape=(2,), dtype=np.asarray(delta_field).dtype)
        floating[:] A = np.ndarray(shape=(3,), dtype=np.asarray(delta_field).dtype)
        floating xx, yy, opt, nrm2, delta, sigma, max_displacement
    max_displacement = 0

    with nogil:

        for r in range(nrows):
            for c in range(ncols):
                delta = delta_field[r, c]
                sigma = sigma_field[r, c] if sigma_field != None else 1
                if(target == None):
                    b[0] = delta_field[r, c] * gradient_field[r, c, 0]
                    b[1] = delta_field[r, c] * gradient_field[r, c, 1]
                else:
                    b[0] = target[r, c, 0]
                    b[1] = target[r, c, 1]
                nn = 0
                y[:] = 0
                for k in range(NUM_NEIGHBORS):
                    dr = r + dRow[k]
                    if((dr < 0) or (dr >= nrows)):
                        continue
                    dc = c + dCol[k]
                    if((dc < 0) or (dc >= ncols)):
                        continue
                    nn += 1
                    y[0] += displacement_field[dr, dc, 0]
                    y[1] += displacement_field[dr, dc, 1]
                if(isinf(sigma)):
                    xx = displacement_field[r, c, 0]
                    yy = displacement_field[r, c, 1]
                    displacement_field[r, c, 0] = y[0] / nn
                    displacement_field[r, c, 1] = y[1] / nn
                    xx -= displacement_field[r, c, 0]
                    yy -= displacement_field[r, c, 1]
                    opt = xx * xx + yy * yy
                    if(max_displacement < opt):
                        max_displacement = opt
                elif(sigma == 0):
                    nrm2 = gradient_field[r, c, 0] ** 2 + \
                        gradient_field[r, c, 1] ** 2
                    if(nrm2 == 0):
                        displacement_field[r, c, 0] = 0
                        displacement_field[r, c, 1] = 0                        
                    else:
                        displacement_field[r, c, 0] = (b[0]) / nrm2
                        displacement_field[r, c, 1] = (b[1]) / nrm2
                else:
                    y[0] = b[0] + sigma * lambda_param * y[0]
                    y[1] = b[1] + sigma * lambda_param * y[1]
                    A[0] = gradient_field[r, c, 0] ** 2 + sigma * lambda_param * nn
                    A[1] = gradient_field[r, c, 0] * gradient_field[r, c, 1]
                    A[2] = gradient_field[r, c, 1] ** 2 + sigma * lambda_param * nn
                    xx = displacement_field[r, c, 0]
                    yy = displacement_field[r, c, 1]
                    solve2DSymmetricPositiveDefiniteSystem(A, y, d)
                    displacement_field[r, c, 0] = d[0]
                    displacement_field[r, c, 1] = d[1]
                    xx -= d[0]
                    yy -= d[1]
                    opt = xx * xx + yy * yy
                    if(max_displacement < opt):
                        max_displacement = opt
    return sqrt(max_displacement)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double compute_energy_SSD2D(floating[:, :] delta_field,
                                  floating[:, :] sigma_field,
                                  floating[:, :, :] gradient_field,
                                  double lambda_param,
                                  floating[:, :, :] displacement_field):
    r"""
    Computes the Sum of Squared Differences between the static and moving image.
    Those differences are given by delta_field

    Parameters
    ----------
    delta_field : array, shape (R, C)
        the difference between the static and moving image (the 'derivatice
        w.r.t. time' in the optical flow model)
    sigma_field : array, shape (R, C)
        the variance of the gray level value at each voxel, according to the 
        EM model (for SSD, it is 1 for all voxels). Inf and 0 values
        are processed specially to support infinite and zero variance.
    gradient_field : array, shape (R, C, 2)
        the gradient of the moving image
    lambda_param : float
        smoothness parameter of the objective function
    displacement_field : array, shape (R, C, 2)
        current displacement field to start the iteration from

    Returns
    -------
    energy : float
        the SSD energy at this iteration 

    Notes
    -----
    Currently, this function only computes the SSD, but it is a special case
    of the EM formulation. We are leaving the extra parameters as placeholders
    for future generalization to the EM metric energy computation
    """
    cdef:
        int nrows = delta_field.shape[0]
        int ncols = delta_field.shape[1]
        floating energy = 0

    with nogil:
        for r in range(nrows):
            for c in range(ncols):
                energy += delta_field[r, c] ** 2
    return energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double iterate_residual_displacement_field_SSD3D(floating[:, :, :] delta_field,
                                                       floating[:, :, :] sigma_field,
                                                       floating[:, :, :, :] gradient_field,
                                                       floating[:, :, :, :] target,
                                                       double lambda_param,
                                                       floating[:, :, :, :] displacement_field):
    r"""
    Performs one iteration at one level of the Multi-resolution Gauss-Seidel 
    solver proposed by Bruhn and Weickert[1].
    
    [1] Weickert, J. (2005). Towards Ultimate Motion Estimation : Combining
        Highest Accuracy with Real-Time Performance Faculty of Mathematics
        and Computer Science.

    Parameters
    ----------
    delta_field : array, shape (S, R, C)
        the difference between the static and moving image (the 'derivatice
        w.r.t. time' in the optical flow model)
    sigma_field : array, shape (S, R, C)
        the variance of the gray level value at each voxel, according to the 
        EM model (for SSD, it is 1 for all voxels). Inf and 0 values
        are processed specially to support infinite and zero variance.
    gradient_field : array, shape (S, R, C, 3)
        the gradient of the moving image
    target : array, shape (S, R, C, 3)
        right-hand side of the linear system to be solved in the Weickert's
        multiresolution algorithm
    lambda_param : float
        smoothness parameter of the objective function
    displacement_field : array, shape (S, R, C, 3)
        the displacement field to start the optimization from

    Returns
    -------
    max_displacement : float
        the norm of the maximum change in the displacement field after the 
        iteration
    """
    cdef:
        int NUM_NEIGHBORS = 6
        int[:] dSlice = np.array([-1,  0, 0, 0,  0, 1], dtype=np.int32)
        int[:] dRow = np.array([0, -1, 0, 1,  0, 0], dtype=np.int32)
        int[:] dCol = np.array([0,  0, 1, 0, -1, 0], dtype=np.int32)
        int nslices = delta_field.shape[0]
        int nrows = delta_field.shape[1]
        int ncols = delta_field.shape[2]
        int nn        
        floating[:] b = np.ndarray(shape=(3,), dtype=np.asarray(delta_field).dtype)
        floating[:] d = np.ndarray(shape=(3,), dtype=np.asarray(delta_field).dtype)
        floating[:] y = np.ndarray(shape=(3,), dtype=np.asarray(delta_field).dtype)
        floating[:] A = np.ndarray(shape=(6,), dtype=np.asarray(delta_field).dtype)
        floating xx, yy, zz, opt, nrm2, delta, sigma, max_displacement
        int dr, ds, dc, s, r, c
    max_displacement = 0

    with nogil:

        for s in range(nslices):
            for r in range(nrows):
                for c in range(ncols):
                    delta = delta_field[s, r, c]
                    sigma = sigma_field[s, r, c] if sigma_field != None else 1
                    if(target == None):
                        b[0] = delta_field[s, r, c] * gradient_field[s, r, c, 0]
                        b[1] = delta_field[s, r, c] * gradient_field[s, r, c, 1]
                        b[2] = delta_field[s, r, c] * gradient_field[s, r, c, 2]
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
                        if((ds < 0) or (ds >= nslices)):
                            continue
                        dr = r + dRow[k]
                        if((dr < 0) or (dr >= nrows)):
                            continue
                        dc = c + dCol[k]
                        if((dc < 0) or (dc >= ncols)):
                            continue
                        nn += 1
                        y[0] += displacement_field[ds, dr, dc, 0]
                        y[1] += displacement_field[ds, dr, dc, 1]
                        y[2] += displacement_field[ds, dr, dc, 2]
                    if(isinf(sigma)):
                        xx = displacement_field[s, r, c, 0]
                        yy = displacement_field[s, r, c, 1]
                        zz = displacement_field[s, r, c, 2]
                        displacement_field[s, r, c, 0] = y[0] / nn
                        displacement_field[s, r, c, 1] = y[1] / nn
                        displacement_field[s, r, c, 2] = y[2] / nn
                        xx -= displacement_field[s, r, c, 0]
                        yy -= displacement_field[s, r, c, 1]
                        zz -= displacement_field[s, r, c, 2]
                        opt = xx * xx + yy * yy + zz * zz
                        if(max_displacement < opt):
                            max_displacement = opt
                    elif(sigma == 0):
                            nrm2 = gradient_field[s, r, c, 0] ** 2 + \
                                gradient_field[s, r, c, 1] ** 2 + \
                                gradient_field[s, r, c, 2] ** 2
                            if(nrm2 == 0):                                
                                displacement_field[s, r, c, 0] = 0
                                displacement_field[s, r, c, 1] = 0
                                displacement_field[s, r, c, 2] = 0
                            else:
                                displacement_field[s, r, c, 0] = (b[0]) / nrm2
                                displacement_field[s, r, c, 1] = (b[1]) / nrm2
                                displacement_field[s, r, c, 2] = (b[2]) / nrm2
                    else:
                        y[0] = b[0] + sigma * lambda_param * y[0]
                        y[1] = b[1] + sigma * lambda_param * y[1]
                        y[2] = b[2] + sigma * lambda_param * y[2]
                        A[0] = gradient_field[s, r, c, 0] * \
                            gradient_field[s, r, c, 0] + sigma * lambda_param * nn
                        A[1] = gradient_field[s, r, c, 0] * \
                            gradient_field[s, r, c, 1]
                        A[2] = gradient_field[s, r, c, 0] * \
                            gradient_field[s, r, c, 2]
                        A[3] = gradient_field[s, r, c, 1] * \
                            gradient_field[s, r, c, 1] + sigma * lambda_param * nn
                        A[4] = gradient_field[s, r, c, 1] * \
                            gradient_field[s, r, c, 2]
                        A[5] = gradient_field[s, r, c, 2] ** 2 + \
                            sigma * lambda_param * nn
                        xx = displacement_field[s, r, c, 0]
                        yy = displacement_field[s, r, c, 1]
                        zz = displacement_field[s, r, c, 2]
                        solve3DSymmetricPositiveDefiniteSystem(A, y, d)
                        displacement_field[s, r, c, 0] = d[0]
                        displacement_field[s, r, c, 1] = d[1]
                        displacement_field[s, r, c, 2] = d[2]
                        xx -= displacement_field[s, r, c, 0]
                        yy -= displacement_field[s, r, c, 1]
                        zz -= displacement_field[s, r, c, 2]
                        opt = xx * xx + yy * yy + zz * zz
                        if(max_displacement < opt):
                            max_displacement = opt
    return sqrt(max_displacement)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double compute_energy_SSD3D(floating[:, :, :] delta_field,
                                  floating[:, :, :] sigma_field,
                                  floating[:, :, :, :] gradient_field,
                                  double lambda_param,
                                  floating[:, :, :, :] displacement_field):
    r"""
    Computes the Sum of Squared Differences between the static and moving volume
    Those differences are given by delta_field

    Parameters
    ----------
    delta_field : array, shape (R, C)
        the difference between the static and moving image (the 'derivatice
        w.r.t. time' in the optical flow model)
    sigma_field : array, shape (R, C)
        the variance of the gray level value at each voxel, according to the 
        EM model (for SSD, it is 1 for all voxels). Inf and 0 values
        are processed specially to support infinite and zero variance.
    gradient_field : array, shape (R, C, 2)
        the gradient of the moving image
    lambda_param : float
        smoothness parameter of the objective function
    displacement_field : array, shape (R, C, 2)
        current displacement field to start the iteration from

    Returns
    -------
    energy : float
        the SSD energy at this iteration 

    Notes
    -----
    Currently, this function only computes the SSD, but it is a special case
    of the EM formulation. We are leaving the extra parameters as placeholders
    for future generalization to the EM metric energy computation
    """
    cdef:
        int nslices = delta_field.shape[0]
        int nrows = delta_field.shape[1]
        int ncols = delta_field.shape[2]
        floating energy = 0
    with nogil:
        for s in range(nslices):
            for r in range(nrows):
                for c in range(ncols):
                    energy += delta_field[s, r, c] ** 2
    return energy


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_residual_displacement_field_SSD3D(floating[:, :, :] delta_field,
                                              floating[:, :, :] sigma_field,
                                              floating[:, :, :, :] gradient_field,
                                              floating[:, :, :, :] target,
                                              double lambda_param,
                                              floating[:, :, :, :] displacement_field,
                                              floating[:, :, :, :] residual):
    r"""
    Computes the residual displacement field corresponding to the current 
    displacement field (given by 'displacement_field') in the Multi-resolution 
    Gauss-Seidel solver proposed by Bruhn and Weickert[1].
    
    [1] Weickert, J. (2005). Towards Ultimate Motion Estimation : Combining
        Highest Accuracy with Real-Time Performance Faculty of Mathematics
        and Computer Science.
    
    Parameters
    ----------
    delta_field : array, shape (S, R, C)
        the difference between the static and moving image (the 'derivatice
        w.r.t. time' in the optical flow model)
    sigma_field : array, shape (S, R, C)
        the variance of the gray level value at each voxel, according to the 
        EM model (for SSD, it is 1 for all voxels). Inf and 0 values
        are processed specially to support infinite and zero variance.
    gradient_field : array, shape (S, R, C, 3)
        the gradient of the moving image
    target : array, shape (S, R, C, 3)
        right-hand side of the linear system to be solved in the Weickert's
        multiresolution algorithm
    lambda_param : float
        smoothness parameter in the objective function
    displacement_field : array, shape (S, R, C, 3)
        the current displacement field to compute the residual from
    residual : array, shape (S, R, C, 3)
        the displacement field to put the residual to

    Returns
    -------
    residual : array, shape (S, R, C, 3)
        the residual displacement field. If residual was None a input, then
        a new field is returned, otherwise the same array is returned
    """
    cdef:
        int NUM_NEIGHBORS = 6
        int[:] dSlice = np.array([-1,  0, 0, 0,  0, 1], dtype=np.int32)
        int[:] dRow = np.array([0, -1, 0, 1,  0, 0], dtype=np.int32)
        int[:] dCol = np.array([0,  0, 1, 0, -1, 0], dtype=np.int32)
        floating[:] b = np.ndarray(shape=(3,), dtype=np.asarray(delta_field).dtype)
        floating[:] y = np.ndarray(shape=(3,), dtype=np.asarray(delta_field).dtype)
        int nslices = delta_field.shape[0]
        int nrows = delta_field.shape[1]
        int ncols = delta_field.shape[2]
        floating delta, sigma, dotP
        int s, r, c, ds, dr, dc
    if residual == None:
        residual = np.empty(shape=(nslices, nrows, ncols, 3), dtype=np.asarray(delta_field).dtype)
    for s in range(nslices):
        for r in range(nrows):
            for c in range(ncols):
                delta = delta_field[s, r, c]
                sigma = sigma_field[s, r, c] if sigma_field != None else 1
                if(target == None):
                    b[0] = delta * gradient_field[s, r, c, 0]
                    b[1] = delta * gradient_field[s, r, c, 1]
                    b[2] = delta * gradient_field[s, r, c, 2]
                else:
                    b[0] = target[s, r, c, 0]
                    b[1] = target[s, r, c, 1]
                    b[2] = target[s, r, c, 2]
                y[...] = 0
                for k in range(NUM_NEIGHBORS):
                    ds = s + dSlice[k]
                    if((ds < 0) or (ds >= nslices)):
                        continue
                    dr = r + dRow[k]
                    if((dr < 0) or (dr >= nrows)):
                        continue
                    dc = c + dCol[k]
                    if((dc < 0) or (dc >= ncols)):
                        continue
                    y[0] += displacement_field[s, r, c, 0] - \
                        displacement_field[ds, dr, dc, 0]
                    y[1] += displacement_field[s, r, c, 1] - \
                        displacement_field[ds, dr, dc, 1]
                    y[2] += displacement_field[s, r, c, 2] - \
                        displacement_field[ds, dr, dc, 2]
                if(isinf(sigma)):
                    residual[s, r, c, 0] = -lambda_param * y[0]
                    residual[s, r, c, 1] = -lambda_param * y[1]
                    residual[s, r, c, 2] = -lambda_param * y[2]
                else:
                    dotP = gradient_field[s, r, c, 0] * displacement_field[s, r, c, 0] + gradient_field[s, r, c, 1] * \
                        displacement_field[s, r, c, 1] + \
                        gradient_field[s, r, c, 2] * \
                        displacement_field[s, r, c, 2]
                    residual[s, r, c, 0] = b[0] - \
                        (gradient_field[s, r, c, 0]
                         * dotP + sigma * lambda_param * y[0])
                    residual[s, r, c, 1] = b[1] - \
                        (gradient_field[s, r, c, 1]
                         * dotP + sigma * lambda_param * y[1])
                    residual[s, r, c, 2] = b[2] - \
                        (gradient_field[s, r, c, 2]
                         * dotP + sigma * lambda_param * y[2])
    return residual


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_residual_displacement_field_SSD2D(floating[:, :] delta_field,
                                                floating[:, :] sigma_field,
                                                floating[:, :, :] gradient_field,
                                                floating[:, :, :] target,
                                                double lambda_param,
                                                floating[:, :, :] displacement_field,
                                                floating[:, :, :] residual):
    r"""
    Computes the residual displacement field corresponding to the current 
    displacement field in the Multi-resolution Gauss-Seidel solver proposed by 
    Bruhn and Weickert[1].

    [1] Weickert, J. (2005). Towards Ultimate Motion Estimation : Combining
        Highest Accuracy with Real-Time Performance Faculty of Mathematics
        and Computer Science.
    
    Parameters
    ----------
    delta_field : array, shape (R, C)
        the difference between the static and moving image (the 'derivatice
        w.r.t. time' in the optical flow model)
    sigma_field : array, shape (R, C)
        the variance of the gray level value at each voxel, according to the 
        EM model (for SSD, it is 1 for all voxels). Inf and 0 values
        are processed specially to support infinite and zero variance.
    gradient_field : array, shape (R, C, 2)
        the gradient of the moving image
    target : array, shape (R, C, 2)
        right-hand side of the linear system to be solved in the Weickert's
        multiresolution algorithm
    lambda_param : float
        smoothness parameter in the objective function
    displacement_field : array, shape (R, C, 2)
        the current displacement field to compute the residual from
    residual : array, shape (R, C, 2)
        the displacement field to put the residual to

    Returns
    -------
    residual : array, shape (R, C, 2)
        the residual displacement field. If residual was None a input, then
        a new field is returned, otherwise the same array is returned
    """
    cdef:
        int NUM_NEIGHBORS = 4
        int[:] dRow = np.array([-1, 0, 1,  0], dtype=np.int32)
        int[:] dCol = np.array([0, 1, 0, -1], dtype=np.int32)
        floating[:] b = np.ndarray(shape=(2,), dtype=np.asarray(delta_field).dtype)
        floating[:] y = np.ndarray(shape=(2,), dtype=np.asarray(delta_field).dtype)
        int nrows = delta_field.shape[0]
        int ncols = delta_field.shape[1]
        floating delta, sigma, dotP
        int r, c, dr, dc
    if residual == None:
        residual = np.empty(shape=(nrows, ncols, 2), dtype=np.asarray(delta_field).dtype)
    for r in range(nrows):
        for c in range(ncols):
            delta = delta_field[r, c]
            sigma = sigma_field[r, c] if sigma_field != None else 1
            if(target == None):
                b[0] = delta * gradient_field[r, c, 0]
                b[1] = delta * gradient_field[r, c, 1]
            else:
                b[0] = target[r, c, 0]
                b[1] = target[r, c, 1]
            y[...] = 0
            for k in range(NUM_NEIGHBORS):
                dr = r + dRow[k]
                if((dr < 0) or (dr >= nrows)):
                    continue
                dc = c + dCol[k]
                if((dc < 0) or (dc >= ncols)):
                    continue
                y[0] += displacement_field[r, c, 0] - \
                    displacement_field[dr, dc, 0]
                y[1] += displacement_field[r, c, 1] - \
                    displacement_field[dr, dc, 1]
            if(isinf(sigma)):
                residual[r, c, 0] = -lambda_param * y[0]
                residual[r, c, 1] = -lambda_param * y[1]
            else:
                dotP = gradient_field[r, c, 0] * displacement_field[r, c, 0] + \
                    gradient_field[r, c, 1] * displacement_field[r, c, 1]
                residual[r, c, 0] = b[0] - \
                    (gradient_field[r, c, 0] * dotP +
                     sigma * lambda_param * y[0])
                residual[r, c, 1] = b[1] - \
                    (gradient_field[r, c, 1] * dotP +
                     sigma * lambda_param * y[1])
    return residual


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_ssd_demons_step_2d(floating[:,:] delta_field,
                               floating[:,:,:] gradient_moving,
                               double sigma_reg_2,
                               floating[:,:,:] out):
    r"""
    Computes the demons step for SSD-driven registration ( eq. 4 in [1] )

    [1] Vercauteren, T., Pennec, X., Perchant, A., & Ayache, N. (2009).
        Diffeomorphic demons: efficient non-parametric image registration. 
        NeuroImage, 45(1 Suppl), S61–72. doi:10.1016/j.neuroimage.2008.10.040

    Parameters
    ----------
    delta_field : array, shape (R, C)
        the difference between the static and moving image (the 'derivative
        w.r.t. time' in the optical flow model)
    gradient_field : array, shape (R, C, 2)
        the gradient of the moving image
    sigma_reg_2 : float
        parameter controlling the amount of reguarization (under the Ridge 
        regression model: \min_{x} ||Ax - y||^2 + \frac{1}{'sigmadiff'}||x||^2)
        (also, it is \sigma_x in eq. 4 of [1])
    out : array, shape (R, C, 2)
        if None, a new array will be created to store the demons step. Otherwise
        the provided array will be used.

    Returns
    -------
    demons_step:
        the demons step to be applied for updating the current displacement field
    """
    cdef:
        int nr = delta_field.shape[0]
        int nc = delta_field.shape[1]
        int i, j
        double neg_delta, delta_2, nrm2, energy, den

    if out is None:
        out = np.zeros((nr, nc, 2), dtype=np.asarray(delta_field).dtype)

    with nogil:

        energy = 0
        for i in range(nr):
            for j in range(nc):
                neg_delta = -1 * delta_field[i,j]
                delta_2 = neg_delta**2 
                energy += delta_2
                nrm2 = gradient_moving[i, j, 0]**2 + gradient_moving[i, j, 1]**2
                den = delta_2/sigma_reg_2 + nrm2
                if den <1e-9:
                    out[i, j, 0] = 0
                    out[i, j, 1] = 0
                else:
                    out[i, j, 0] = neg_delta * gradient_moving[i, j, 0] / den
                    out[i, j, 1] = neg_delta * gradient_moving[i, j, 1] / den

    return out, energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_ssd_demons_step_3d(floating[:,:,:] delta_field,
                               floating[:,:,:,:] gradient_moving,
                               double sigma_reg_2,
                               floating[:,:,:,:] out):
    r"""
    Computes the demons step for SSD-driven registration ( eq. 4 in [1] )

    [1] Vercauteren, T., Pennec, X., Perchant, A., & Ayache, N. (2009).
        Diffeomorphic demons: efficient non-parametric image registration. 
        NeuroImage, 45(1 Suppl), S61–72. doi:10.1016/j.neuroimage.2008.10.040

    Parameters
    ----------
    delta_field : array, shape (S, R, C)
        the difference between the static and moving image (the 'derivatice
        w.r.t. time' in the optical flow model)
    gradient_field : array, shape (S, R, C, 2)
        the gradient of the moving image
    sigma_reg_2 : float
        parameter controlling the amount of reguarization (under the Ridge 
        regression model: \min_{x} ||Ax - y||^2 + \frac{1}{'sigmadiff'}||x||^2)
        (also, it is \sigma_x in eq. 4 of [1])
    out : array, shape (S, R, C, 2)
        if None, a new array will be created to store the demons step. Otherwise
        the provided array will be used.

    Returns
    -------
    demons_step:
        the demons step to be applied for updating the current displacement field
    """
    cdef:
        int ns = delta_field.shape[0]
        int nr = delta_field.shape[1]
        int nc = delta_field.shape[2]
        int i, j, k
        double neg_delta, delta_2, nrm2, energy, den

    if out is None:
        out = np.zeros((ns, nr, nc, 3), dtype=np.asarray(delta_field).dtype)

    with nogil:

        energy = 0
        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    neg_delta = -1 * delta_field[k,i,j]
                    delta_2 = neg_delta**2 
                    energy += delta_2
                    nrm2 = gradient_moving[k, i, j, 0]**2 + gradient_moving[k, i, j, 1]**2 + gradient_moving[k, i, j, 2]**2
                    den = delta_2/sigma_reg_2 + nrm2
                    if den < 1e-9:
                        out[k, i, j, 0] = 0
                        out[k, i, j, 1] = 0
                        out[k, i, j, 2] = 0
                    else: 
                        out[k, i, j, 0] = neg_delta * gradient_moving[k, i, j, 0] / den
                        out[k, i, j, 1] = neg_delta * gradient_moving[k, i, j, 1] / den
                        out[k, i, j, 2] = neg_delta * gradient_moving[k, i, j, 2] / den

    return out, energy
