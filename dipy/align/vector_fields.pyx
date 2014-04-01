#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport cython
from fused_types cimport floating, number

cdef extern from "math.h":
    double sqrt(double x) nogil
    double floor(double x) nogil


cdef inline double _apply_affine_3d_x0(double x0, double x1, double x2, double h,
                                       floating[:, :] aff) nogil:
    r"""
    Returns the first component of the product of the affine matrix aff by
    (x0, x1, x2)
    """
    return aff[0, 0] * x0 + aff[0, 1] * x1 + aff[0, 2] * x2 + h*aff[0, 3]


cdef inline double _apply_affine_3d_x1(double x0, double x1, double x2, double h,
                                       floating[:, :] aff) nogil:
    r"""
    Returns the first component of the product of the affine matrix aff by
    (x0, x1, x2)
    """
    return aff[1, 0] * x0 + aff[1, 1] * x1 + aff[1, 2] * x2 + h*aff[1, 3]


cdef inline double _apply_affine_3d_x2(double x0, double x1, double x2, double h,
                                       floating[:, :] aff) nogil:
    r"""
    Returns the first component of the product of the affine matrix aff by
    (x0, x1, x2)
    """
    return aff[2, 0] * x0 + aff[2, 1] * x1 + aff[2, 2] * x2 + h*aff[2, 3]


cdef inline double _apply_affine_2d_x0(double x0, double x1, double h,
                                       floating[:, :] aff) nogil:
    r"""
    Returns the first component of the product of the aff matrix aff by
    (x0, x1, x2)
    """
    return aff[0, 0] * x0 + aff[0, 1] * x1 + h*aff[0, 2]


cdef inline double _apply_affine_2d_x1(double x0, double x1, double h,
                                       floating[:, :] aff) nogil:
    r"""
    Returns the first component of the product of the affine matrix aff by
    (x0, x1, x2)
    """
    return aff[1, 0] * x0 + aff[1, 1] * x1 + h*aff[1, 2]

cdef int mult_matrices(floating[:,:] A, floating[:,:] B, floating[:,:] out) nogil:
    cdef:
        int nrA = A.shape[0]
        int ncA = A.shape[1]
        int nrB = B.shape[0]
        int ncB = B.shape[1]
        double s
    if A is None:
        if B is None:
            return 0
        else:
            for i in range(nrB):
                for j in  range(ncB):
                    out[i,j] = B[i, j]
    elif B is None:
        for i in range(nrA):
            for j in  range(ncA):
                out[i,j] = A[i, j]
    else:
        for i in range(nrA):
            for j in  range(ncB):
                s = 0
                for k in range(ncA):
                    s += A[i,k]*B[k,j]
                out[i,j] = s
    return 1



cdef inline int interpolate_vector_bilinear(floating[:,:,:] field, double dii, 
                                     double djj, floating[:] out) nogil:
    r"""
    Interpolates the 2D displacement field at (dii, djj) and stores the 
    result in out. If (dkk, dii, djj) is outside the vector field's domain, a 
    zero vector is written to out instead.

    Parameters
    ----------
    field : array, shape (R, C)
        the input 2D displacement field
    dii : floating
        the first coordinate of the interpolating position
    djj : floating
        the second coordinate of the interpolating position    
    out : array, shape (2,)
        the array which the interpolation result will be written to

    Returns
    -------
    inside : int
        if (dii, djj) is inside the domain of the displacement field, 
        inside == 1, otherwise inside == 0
    """
    cdef:
        int nr = field.shape[0]
        int nc = field.shape[1]
        int ii, jj
        double alpha, beta, calpha, cbeta
    if((dii < 0) or (djj < 0) or (dii > nr - 1) or (djj > nc - 1)):
        out[0] = 0
        out[1] = 0
        return 0
    #---top-left
    ii = <int>floor(dii)
    jj = <int>floor(djj)
    if((ii < 0) or (jj < 0) or (ii >= nr) or (jj >= nc)):
        out[0] = 0
        out[1] = 0
        return 0
    calpha = dii - ii
    cbeta = djj - jj
    alpha = 1 - calpha
    beta = 1 - cbeta
    
    out[0] = alpha * beta * field[ii, jj, 0]
    out[1] = alpha * beta * field[ii, jj, 1]
    #---top-right
    jj += 1
    if(jj < nc):
        out[0] += alpha * cbeta * field[ii, jj, 0]
        out[1] += alpha * cbeta * field[ii, jj, 1]
    #---bottom-right
    ii += 1
    if((ii >= 0) and (jj >= 0) and (ii < nr) and (jj < nc)):
        out[0] += calpha * cbeta * field[ii, jj, 0]
        out[1] += calpha * cbeta * field[ii, jj, 1]
    #---bottom-left
    jj -= 1
    if((ii >= 0) and (jj >= 0) and (ii < nr) and (jj < nc)):
        out[0] += calpha * beta * field[ii, jj, 0]
        out[1] += calpha * beta * field[ii, jj, 1]
    return 1    


cdef inline int interpolate_scalar_bilinear(floating[:,:] image, double dii, 
                                     double djj, floating *out) nogil:
    r"""
    Interpolates the 2D image at (dii, djj) and stores the 
    result in out. If (dii, djj) is outside the image's domain,
    zero is written to out instead.

    Parameters
    ----------
    image : array, shape (R, C)
        the input 2D image
    dii : floating
        the first coordinate of the interpolating position
    djj : floating
        the second coordinate of the interpolating position    
    out : array, shape (2,)
        the array which the interpolation result will be written to

    Returns
    -------
    inside : int
        if (dii, djj) is inside the domain of the image, 
        inside == 1, otherwise inside == 0
    """
    cdef:
        int nr = image.shape[0]
        int nc = image.shape[1]
        int ii, jj
        double alpha, beta, calpha, cbeta
    if((dii < 0) or (djj < 0) or (dii > nr - 1) or (djj > nc - 1)):
        out[0] = 0
        return 0
    #---top-left
    ii = <int>floor(dii)
    jj = <int>floor(djj)
    if((ii < 0) or (jj < 0) or (ii >= nr) or (jj >= nc)):
        out[0] = 0
        return 0
    calpha = dii - ii
    cbeta = djj - jj
    alpha = 1 - calpha
    beta = 1 - cbeta
    
    out[0] = alpha * beta * image[ii, jj]
    #---top-right
    jj += 1
    if(jj < nc):
        out[0] += alpha * cbeta * image[ii, jj]
    #---bottom-right
    ii += 1
    if((ii >= 0) and (jj >= 0) and (ii < nr) and (jj < nc)):
        out[0] += calpha * cbeta * image[ii, jj]
    #---bottom-left
    jj -= 1
    if((ii >= 0) and (jj >= 0) and (ii < nr) and (jj < nc)):
        out[0] += calpha * beta * image[ii, jj]
    return 1


cdef inline int interpolate_scalar_nn_2d(number[:,:] image, double dii, 
                                         double djj, number *out) nogil:
    r"""
    Interpolates the 2D image at (dii, djj) using nearest neighbor interpolation
    and stores the result in out. If (dii, djj) is outside the image's domain,
    zero is written to out instead.

    Parameters
    ----------
    image : array, shape (R, C)
        the input 2D image
    dii : float
        the first coordinate of the interpolating position
    djj : float
        the second coordinate of the interpolating position    
    out : array, shape (1,)
        the variable which the interpolation result will be written to

    Returns
    -------
    inside : int
        if (dii, djj) is inside the domain of the image, 
        inside == 1, otherwise inside == 0
    """
    cdef:
        int nr = image.shape[0]
        int nc = image.shape[1]
        int ii, jj
        double alpha, beta, calpha, cbeta
    if((dii < 0) or (djj < 0) or (dii > nr - 1) or (djj > nc - 1)):
        out[0] = 0
        return 0
    # find the top left index and the interpolation coefficients
    ii = <int>floor(dii)
    jj = <int>floor(djj)
    # no one is affected
    if((ii < 0) or (jj < 0) or (ii >= nr) or (jj >= nc)):
        out[0] = 0
        return 0
    calpha = dii - ii  # by definition these factors are nonnegative
    cbeta = djj - jj
    alpha = 1 - calpha
    beta = 1 - cbeta
    if(alpha < calpha):
        ii += 1
    if(beta < cbeta):
        jj += 1
    # no one is affected
    if((ii < 0) or (jj < 0) or (ii >= nr) or (jj >= nc)):
        out[0] = 0
        return 0
    out[0] = image[ii, jj]
    return 1


cdef inline int interpolate_scalar_nn_3d(number[:,:,:] volume, double dkk, 
                                         double dii, double djj, number *out) nogil:
    r"""
    Interpolates the 3D image at (dkk, dii, djj) using nearest neighbor interpolation
    and stores the result in out. If (dkk, dii, djj) is outside the image's domain,
    zero is written to out instead.

    Parameters
    ----------
    image : array, shape (S, R, C)
        the input 2D image
    dkk : float
        the first coordinate of the interpolating position
    dii : float
        the second coordinate of the interpolating position
    djj : float
        the third coordinate of the interpolating position    
    out : array, shape (1,)
        the variable which the interpolation result will be written to

    Returns
    -------
    inside : int
        if (dkk, dii, djj) is inside the domain of the image, 
        inside == 1, otherwise inside == 0
    """
    cdef:
        int ns = volume.shape[0]
        int nr = volume.shape[1]
        int nc = volume.shape[2]
        int kk, ii, jj
        double alpha, beta, calpha, cbeta, gamma, cgamma
    if((dkk < 0) or (dii < 0) or (djj < 0) or (dii > nr - 1) or (djj > nc - 1) or (dkk > ns - 1)):
        out[0] = 0
        return 0
    # find the top left index and the interpolation coefficients
    kk = <int>floor(dkk)
    ii = <int>floor(dii)
    jj = <int>floor(djj)
    # no one is affected
    if((kk < 0) or (ii < 0) or (jj < 0) or (kk >= ns) or (ii >= nr) or (jj >= nc)):
        out[0] = 0
        return 0
    cgamma = dkk - kk
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
    if((kk < 0) or (ii < 0) or (jj < 0) or (kk >= ns) or (ii >= nr) or (jj >= nc)):
        out[0] = 0
        return 0
    out[0] = volume[kk, ii, jj]
    return 1


cdef inline int interpolate_scalar_trilinear(floating[:,:,:] volume, 
                                             double dkk, double dii, double djj, 
                                             floating *out) nogil:
    cdef:
        int ns = volume.shape[0]
        int nr = volume.shape[1]
        int nc = volume.shape[2]
        int kk, ii, jj
        double alpha, beta, calpha, cbeta, gamma, cgamma
    if((dkk < 0) or (dii < 0) or (djj < 0) or (dkk > ns - 1) or (dii > nr - 1) or (djj > nc - 1)):
        out[0] = 0
        return 0
    # find the top left index and the interpolation coefficients
    kk = <int>floor(dkk)
    ii = <int>floor(dii)
    jj = <int>floor(djj)
    # no one is affected
    if((kk < 0) or (ii < 0) or (jj < 0) or (kk >= ns) or (ii >= nr) or (jj >= nc) ):
        out[0] = 0
        return 0
    cgamma = dkk - kk
    calpha = dii - ii
    cbeta = djj - jj
    alpha = 1 - calpha
    beta = 1 - cbeta
    gamma = 1 - cgamma
    #---top-left
    out[0] = alpha * beta * gamma * volume[kk, ii, jj]
    #---top-right
    jj += 1
    if(jj < nc):
        out[0] += alpha * cbeta * gamma * volume[kk, ii, jj]
    #---bottom-right
    ii += 1
    if((ii >= 0) and (jj >= 0) and (ii < nr) and (jj < nc)):
        out[0] += calpha * cbeta * gamma * volume[kk, ii, jj]
    #---bottom-left
    jj -= 1
    if((ii >= 0) and (jj >= 0) and (ii < nr) and (jj < nc)):
        out[0] += calpha * beta * gamma * volume[kk, ii, jj]
    kk += 1
    if(kk < ns):
        ii -= 1
        out[0] += alpha * beta * cgamma * volume[kk, ii, jj]
        jj += 1
        if(jj < nc):
            out[0] += alpha * cbeta * cgamma * volume[kk, ii, jj]
        #---bottom-right
        ii += 1
        if((ii >= 0) and (jj >= 0) and (ii < nr) and (jj < nc)):
            out[0] += calpha * cbeta * cgamma * volume[kk, ii, jj]
        #---bottom-left
        jj -= 1
        if((ii >= 0) and (jj >= 0) and (ii < nr) and (jj < nc)):
            out[0] += calpha * beta * cgamma * volume[kk, ii, jj]


cdef inline int interpolate_vector_trilinear(floating[:,:,:,:] field, double dkk, double dii, 
                                      double djj, floating[:] out) nogil:
    r"""
    Interpolates the 3D displacement field at (dkk, dii, djj) and stores the 
    result in out. If (dkk, dii, djj) is outside the vector field's domain, a 
    zero vector is written to out instead.

    Parameters
    ----------
    field : array, shape (S, R, C)
        the input 3D displacement field
    dkk : floating
        the first coordinate of the interpolating position
    dii : floating
        the second coordinate of the interpolating position
    djj : floating
        the third coordinate of the interpolating position    
    out : array, shape (3,)
        the array which the interpolation result will be written to

    Returns
    -------
    inside : int
        if (dkk, dii, djj) is inside the domain of the displacement field, 
        inside == 1, otherwise inside == 0
    """
    cdef:
        int ns = field.shape[0]
        int nr = field.shape[1]
        int nc = field.shape[2]
        int kk, ii, jj
        double alpha, beta, gamma, calpha, cbeta, cgamma
    if((dkk < 0) or (dii < 0) or (djj < 0) or (dkk > ns - 1) or (dii > nr - 1) or (djj > nc - 1)):
        out[0] = 0
        out[1] = 0
        out[2] = 0
        return 0
    #---top-left
    kk = <int>floor(dkk)
    ii = <int>floor(dii)
    jj = <int>floor(djj)
    if((kk < 0) or (ii < 0) or (jj < 0) or (kk >= ns) or (ii >= nr) or (jj >= nc)):
        out[0] = 0
        out[1] = 0
        out[2] = 0
        return 0
    cgamma = dkk - kk
    calpha = dii - ii
    cbeta = djj - jj
    alpha = 1 - calpha
    beta = 1 - cbeta
    gamma = 1 - cgamma
    
    out[0] = alpha * beta * gamma * field[kk, ii, jj, 0]
    out[1] = alpha * beta * gamma * field[kk, ii, jj, 1]
    out[2] = alpha * beta * gamma * field[kk, ii, jj, 2]
    #---top-right
    jj += 1
    if(jj < nc):
        out[0] += alpha * cbeta * gamma * field[kk, ii, jj, 0]
        out[1] += alpha * cbeta * gamma * field[kk, ii, jj, 1]
        out[2] += alpha * cbeta * gamma * field[kk, ii, jj, 2]
    #---bottom-right
    ii += 1
    if((ii >= 0) and (jj >= 0) and (ii < nr) and (jj < nc)):
        out[0] += calpha * cbeta * gamma * field[kk, ii, jj, 0]
        out[1] += calpha * cbeta * gamma * field[kk, ii, jj, 1]
        out[2] += calpha * cbeta * gamma * field[kk, ii, jj, 2]
    #---bottom-left
    jj -= 1
    if((ii >= 0) and (jj >= 0) and (ii < nr) and (jj < nc)):
        out[0] += calpha * beta * gamma * field[kk, ii, jj, 0]
        out[1] += calpha * beta * gamma * field[kk, ii, jj, 1]
        out[2] += calpha * beta * gamma * field[kk, ii, jj, 2]
    kk += 1
    if(kk < ns):
        ii -= 1
        out[0] += alpha * beta * cgamma * field[kk, ii, jj, 0]
        out[1] += alpha * beta * cgamma * field[kk, ii, jj, 1]
        out[2] += alpha * beta * cgamma * field[kk, ii, jj, 2]
        jj += 1
        if(jj < nc):
            out[0] += alpha * cbeta * cgamma * field[kk, ii, jj, 0]
            out[1] += alpha * cbeta * cgamma * field[kk, ii, jj, 1]
            out[2] += alpha * cbeta * cgamma * field[kk, ii, jj, 2]
        #---bottom-right
        ii += 1
        if((ii >= 0) and (jj >= 0) and (ii < nr) and (jj < nc)):
            out[0] += calpha * cbeta * cgamma * field[kk, ii, jj, 0]
            out[1] += calpha * cbeta * cgamma * field[kk, ii, jj, 1]
            out[2] += calpha * cbeta * cgamma * field[kk, ii, jj, 2]
        #---bottom-left
        jj -= 1
        if((ii >= 0) and (jj >= 0) and (ii < nr) and (jj < nc)):
            out[0] += calpha * beta * cgamma * field[kk, ii, jj, 0]
            out[1] += calpha * beta * cgamma * field[kk, ii, jj, 1]
            out[2] += calpha * beta * cgamma * field[kk, ii, jj, 2]
    return 1    


cdef void _compose_vector_fields_2d(floating[:, :, :] d1, floating[:, :, :] d2,
                                    floating[:, :] premult_index, 
                                    floating[:, :] premult_disp,
                                    double time_scaling,
                                    floating[:, :, :] comp, floating[:] stats) nogil:
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
    premult_index : array, shape (3, 3)
        since the displacement fields are operating on the physical space, the
        composition actually applied is of the form 
        comp[i] = d1[i] + t*d2[R2^{-1}.dot(R1) * i + R2^{-1} * d1[i]], where t
        is the time scaling, R1 and R2 are the affine matrices that transform
        discrete indices to physical space in the d1 and d2 discretizations,
        respectively. premult_index corresponds to the R2^{-1}.dot(R1) matrix
        above
    premult_disp : array, shape (3, 3)
        premult_disp corresponds to the R2^{-1} matrix in the above explanation
    time_scaling : float
        this corresponds to the time scaling 't' in the above explanation
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
        int i, j, inside
        floating di, dj, dii, djj

    for i in range(nr1):
        for j in range(nc1):
            
            comp[i, j, 0] = 0
            comp[i, j, 1] = 0

            if premult_disp is None:
                di = d1[i, j, 0]
                dj = d1[i, j, 1]
            else:
                di = _apply_affine_2d_x0(d1[i, j, 0], d1[i, j, 1], 0, premult_disp)
                dj = _apply_affine_2d_x1(d1[i, j, 0], d1[i, j, 1], 0, premult_disp)

            if premult_index is None:
                dii = i
                djj = j
            else:
                dii = _apply_affine_2d_x0(i, j, 1, premult_index)
                djj = _apply_affine_2d_x1(i, j, 1, premult_index)

            dii += di
            djj += dj

            inside = interpolate_vector_bilinear(d2, dii, djj, comp[i,j])
            if inside == 1:
                comp[i,j,0] = time_scaling * comp[i,j,0] + d1[i,j,0]
                comp[i,j,1] = time_scaling * comp[i,j,1] + d1[i,j,1]
                nn = comp[i, j, 0] ** 2 + comp[i, j, 1] ** 2
                meanNorm += nn
                stdNorm += nn * nn
                cnt += 1
                if(maxNorm < nn):
                    maxNorm = nn
    meanNorm /= cnt
    stats[0] = sqrt(maxNorm)
    stats[1] = sqrt(meanNorm)
    stats[2] = sqrt(stdNorm / cnt - meanNorm * meanNorm)


def compose_vector_fields_2d(floating[:, :, :] d1, floating[:, :, :] d2,
                             floating[:, :] premult_index, 
                             floating[:, :] premult_disp,
                             double time_scaling):
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
    premult_index : array, shape (3, 3)
        since the displacement fields are operating on the physical space, the
        composition actually applied is of the form 
        comp[i] = d1[i] + t*d2[R2^{-1}.dot(R1) * i + R2^{-1} * d1[i]], where t
        is the time scaling, R1 and R2 are the affine matrices that transform
        discrete indices to physical space in the d1 and d2 discretizations,
        respectively. premult_index corresponds to the R2^{-1}.dot(R1) matrix
        above
    premult_disp : array, shape (3, 3)
        premult_disp corresponds to the R2^{-1} matrix in the above explanation
    time_scaling : float
        this corresponds to the time scaling 't' in the above explanation

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
                                     dtype=np.asarray(d1).dtype)
    _compose_vector_fields_2d(d1, d2, premult_index, premult_disp, time_scaling, comp, stats)
    return comp, stats


cdef void _compose_vector_fields_3d(floating[:, :, :, :] d1,
                                    floating[:, :, :, :] d2,
                                    floating[:, :] premult_index, 
                                    floating[:, :] premult_disp,
                                    double t,
                                    floating[:, :, :, :] comp,
                                    floating[:] stats) nogil:
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
        floating maxNorm = 0
        floating meanNorm = 0
        floating stdNorm = 0
        floating nn
        int i, j, k, inside
        floating di, dj, dk, dii, djj, dkk
    for k in range(ns1):
        for i in range(nr1):
            for j in range(nc1):
                
                comp[k, i, j, 0] = 0
                comp[k, i, j, 1] = 0
                comp[k, i, j, 2] = 0

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
                    dkk = k
                    dii = i
                    djj = j
                else:
                    dkk = _apply_affine_3d_x0(k, i, j, 1, premult_index)
                    dii = _apply_affine_3d_x1(k, i, j, 1, premult_index)
                    djj = _apply_affine_3d_x2(k, i, j, 1, premult_index)

                dkk += dk
                dii += di
                djj += dj

                inside = interpolate_vector_trilinear(d2, dkk, dii, djj, comp[k, i,j])
                if inside == 1:
                    comp[k, i, j, 0] = t * comp[k, i, j, 0] + d1[k, i, j, 0]
                    comp[k, i, j, 1] = t * comp[k, i, j, 1] + d1[k, i, j, 1]
                    comp[k, i, j, 2] = t * comp[k, i, j, 2] + d1[k, i, j, 2]
                    nn = comp[k, i, j, 0] ** 2 + comp[k, i, j, 1] ** 2 + comp[k, i, j, 2]**2
                    meanNorm += nn
                    stdNorm += nn * nn
                    cnt += 1
                    if(maxNorm < nn):
                        maxNorm = nn
    meanNorm /= cnt
    stats[0] = sqrt(maxNorm)
    stats[1] = sqrt(meanNorm)
    stats[2] = sqrt(stdNorm / cnt - meanNorm * meanNorm)


def compose_vector_fields_3d(floating[:, :, :, :] d1, floating[:, :, :, :] d2,
                             floating[:, :] premult_index, 
                             floating[:, :] premult_disp,
                             double time_scaling):
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
        floating[:] stats = np.zeros(shape=(3,), dtype=np.asarray(d1).dtype)
    _compose_vector_fields_3d(d1, d2, premult_index, premult_disp, time_scaling,
                              comp, stats)
    return comp, stats


def invert_vector_field_fixed_point_2d(floating[:, :, :] d,
                                       floating[:, :] affine_ref,
                                       floating[:, :] affine_ref_inv,
                                       int[:] target_shape,
                                       floating[:, :] target_aff,
                                       floating[:, :] target_aff_inv,
                                       int max_iter, double tolerance,
                                       floating[:, :, :] start=None):
    r"""
    Computes the inverse of the given 2-D displacement field d using the
    fixed-point algorithm.

    Parameters
    ----------
    d : array, shape (R, C, 2)
        the 2-D displacement field to be inverted
    affine_ref : array, shape(3,3)
        the matrix transforming pixel positions in the displacement lattice
        to physical space
    affine_ref_inv : array, shape (3,3)
        the matrix transforming point coordinates in physical space 
        to pixel positions in the dislacement lattice
    target_shape : array, shape (2,)
        the expected shape of the inverse displacement field.
    target_aff : array, shape (3, 3)
        the matrix transforming pixel positions in the inverse displacement 
        lattice to physical space
    target_aff_inv : array, shape (3, 3)
        the matrix transforming point coordinates in physical space 
        to pixel positions in the inverse dislacement lattice
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
    p : array, shape target_shape+(2,) or (R, C, 2) if target_shape is None
        the inverse displacement field

    Notes
    -----
    The 'inversion error' at iteration t is defined as the mean norm of the
    displacement vectors of the input displacement field composed with the
    inverse at iteration t. If target_shape is None, the shape of the resulting
    inverse will be the same as the input displacement field.
    """
    cdef:
        int nr1 = d.shape[0]
        int nc1 = d.shape[1]
        int nr2, nc2, iter_count, current, flag
        double difmag, mag
        double epsilon = 0.25
        double error = 1 + tolerance
        double di, dj, dii, djj

    if target_shape is not None:
        nr2, nc2 = target_shape[0], target_shape[1]
    else:
        nr2, nc2 = nr1, nc1

    cdef:
        floating[:] stats = np.zeros(shape=(2,), dtype=np.asarray(d).dtype)
        floating[:, :, :] p = np.zeros(shape=(nr2, nc2, 2), dtype=np.asarray(d).dtype)
        floating[:, :, :] q = np.zeros(shape=(nr2, nc2, 2), dtype=np.asarray(d).dtype)
        floating[:, :] premult_index = np.zeros((3, 3), dtype = np.asarray(d).dtype)

    if start is not None:
        p[...] = start
    
    flag = mult_matrices(affine_ref_inv, target_aff, premult_index)

    if flag == 0:
        premult_index = None

    with nogil:
        iter_count = 0
        while (iter_count < max_iter) and (tolerance < error):
            p, q = q, p
            difmag = 0
            error = 0
            for i in range(nr2):
                for j in range(nc2):
                    p[i, j, 0] = 0
                    p[i, j, 1] = 0

                    if affine_ref_inv is None:
                        di = q[i, j, 0]
                        dj = q[i, j, 1]
                    else:
                        di = _apply_affine_2d_x0(q[i, j, 0], q[i, j, 1], 0, affine_ref_inv)
                        dj = _apply_affine_2d_x1(q[i, j, 0], q[i, j, 1], 0, affine_ref_inv)

                    if premult_index is None:
                        dii = i
                        djj = j
                    else:
                        dii = _apply_affine_2d_x0(i, j, 1, premult_index)
                        djj = _apply_affine_2d_x1(i, j, 1, premult_index)

                    dii += di
                    djj += dj

                    interpolate_vector_bilinear(d, dii, djj, p[i,j])

                    p[i, j, 0] = (1.0 - epsilon) * q[i,j,0] - epsilon * p[i,j,0]
                    p[i, j, 1] = (1.0 - epsilon) * q[i,j,1] - epsilon * p[i,j,1]
                    di = p[i, j, 0] - q[i, j, 0]
                    dj = p[i, j, 1] - q[i, j, 1]
                    mag = sqrt(di ** 2 + dj ** 2)
                    error += mag
                    if(difmag < mag):
                        difmag = mag
            error /= (nr2 * nc2)            
            iter_count += 1
        stats[0] = error
        stats[1] = iter_count
    return p


def invert_vector_field_fixed_point_3d(floating[:, :, :, :] d,
                                       floating[:, :] affine_ref,
                                       floating[:, :] affine_ref_inv,
                                       int[:] target_shape,
                                       floating[:, :] target_aff,
                                       floating[:, :] target_aff_inv,
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
        int ns2, nr2, nc2, iter_count, current
        double dkk, dii, djj, dk, di, dj
        double difmag, mag
        double epsilon = 0.25
        double error = 1 + tolerance
    if target_shape is not None:
        ns2, nr2, nc2 = target_shape[0], target_shape[1], target_shape[2]
    else:
        ns2, nr2, nc2 = ns1, nr1, nc1
    cdef:
        floating[:] stats = np.zeros(shape=(2,), dtype=np.asarray(d).dtype)
        floating[:, :, :, :] p = np.zeros(shape=(ns2, nr2, nc2, 3), dtype=np.asarray(d).dtype)
        floating[:, :, :, :] q = np.zeros(shape=(ns2, nr2, nc2, 3), dtype=np.asarray(d).dtype)
        floating[:, :] premult_index = np.eye(4, dtype = np.asarray(d).dtype)
    if start is not None:
        p[...] = start

    flag = mult_matrices(affine_ref_inv, target_aff, premult_index)

    if flag == 0:
        premult_index = None

    with nogil:
        iter_count = 0
        while (iter_count < max_iter) and (tolerance < error):
            p, q = q, p
            difmag = 0
            error = 0
            for k in range(ns2):
                for i in range(nr2):
                    for j in range(nc2):
                        
                        p[k, i, j, 0] = 0
                        p[k, i, j, 1] = 0
                        p[k, i, j, 2] = 0

                        dkk = q[k, i, j, 0]
                        dii = q[k, i, j, 1]
                        djj = q[k, i, j, 2]

                        if affine_ref_inv is None:
                            dk = dkk
                            di = dii
                            dj = djj
                        else:
                            dk = _apply_affine_3d_x0(dkk, dii, djj, 0, affine_ref_inv)
                            di = _apply_affine_3d_x1(dkk, dii, djj, 0, affine_ref_inv)
                            dj = _apply_affine_3d_x2(dkk, dii, djj, 0, affine_ref_inv)

                        if premult_index is None:
                            dkk = k
                            dii = i
                            djj = j
                        else:
                            dkk = _apply_affine_3d_x0(k, i, j, 1, premult_index)
                            dii = _apply_affine_3d_x1(k, i, j, 1, premult_index)
                            djj = _apply_affine_3d_x2(k, i, j, 1, premult_index)

                        dkk += dk
                        dii += di
                        djj += dj

                        inside = interpolate_vector_trilinear(d, dkk, dii, djj, p[k, i,j])
                        
                        p[k, i, j, 0] = (1-epsilon) * q[k, i, j, 0] - epsilon * p[k, i, j, 0]
                        p[k, i, j, 1] = (1-epsilon) * q[k, i, j, 1] - epsilon * p[k, i, j, 1]
                        p[k, i, j, 2] = (1-epsilon) * q[k, i, j, 2] - epsilon * p[k, i, j, 2]
                        
                        dk = p[k, i, j, 0] - q[k, i, j, 0]
                        di = p[k, i, j, 1] - q[k, i, j, 1]
                        dj = p[k, i, j, 2] - q[k, i, j, 2]

                        mag = sqrt(dk ** 2 + di ** 2 + dj ** 2)
                        error += mag
                        if(difmag < mag):
                            difmag = mag
            error /= (ns2 * nr2 * nc2)
            iter_count += 1
        stats[0] = error
        stats[1] = iter_count
    return p


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
    if affine is None:
        return
    cdef:
        int nrows = d.shape[0]
        int ncols = d.shape[1]
        int i, j, inside
        floating dii, djj
        floating[:,:,:] out= np.zeros_like(d)
        floating[:] tmp = np.zeros((3,), dtype = np.asarray(d).dtype)
    
    with nogil:
    
        for i in range(nrows):
            for j in range(ncols):
                dii = _apply_affine_2d_x0(i, j, 1, affine)
                djj = _apply_affine_2d_x1(i, j, 1, affine)
                inside = interpolate_vector_bilinear(d, dii, djj, tmp)
                out[i, j, 0] = tmp[0] + dii - i
                out[i, j, 1] = tmp[1] + djj - j

        for i in range(nrows):
            for j in range(ncols):
                d[i, j, 0] = out[i, j, 0]
                d[i, j, 1] = out[i, j, 1]


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
    if affine is None:
        return
    cdef:
        int nslices = d.shape[0]
        int nrows = d.shape[1]
        int ncols = d.shape[2]
        int i, j, k, inside
        floating dkk, dii, djj
        floating[:,:,:,:] out= np.zeros_like(d)
        floating[:] tmp = np.zeros((3,), dtype = np.asarray(d).dtype)
    with nogil:
        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    dkk = _apply_affine_3d_x0(k, i, j, 1, affine)
                    dii = _apply_affine_3d_x1(k, i, j, 1, affine)
                    djj = _apply_affine_3d_x2(k, i, j, 1, affine)
                    inside = interpolate_vector_trilinear(d, dkk, dii, djj, tmp)
                    out[k, i, j, 0] = tmp[0] + dkk - k
                    out[k, i, j, 1] = tmp[1] + dii - i
                    out[k, i, j, 2] = tmp[2] + djj - j

        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    d[k, i, j, 0] = out[k, i, j, 0]
                    d[k, i, j, 1] = out[k, i, j, 1]
                    d[k, i, j, 2] = out[k, i, j, 2]


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
    if affine is None:
        return
    cdef:
        int nrows = d.shape[0]
        int ncols = d.shape[1]
        floating dii, djj
        int i, j

    with nogil:
        for i in range(nrows):
            for j in range(ncols):
                dii = d[i, j, 0] + i
                djj = d[i, j, 1] + j
                d[i, j, 0] = _apply_affine_2d_x0(dii, djj, 1, affine) - i
                d[i, j, 1] = _apply_affine_2d_x1(dii, djj, 1, affine) - j


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
    if affine is None:
        return
    cdef:
        int nslices = d.shape[0]
        int nrows = d.shape[1]
        int ncols = d.shape[2]
        floating dkk, dii, djj
        int i, j, k

    with nogil:
        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    dkk = d[k, i, j, 0] + k
                    dii = d[k, i, j, 1] + i
                    djj = d[k, i, j, 2] + j
                    d[k, i, j, 0] = _apply_affine_3d_x0(dkk, dii, djj, 1, affine) - k
                    d[k, i, j, 1] = _apply_affine_3d_x1(dkk, dii, djj, 1, affine) - i
                    d[k, i, j, 2] = _apply_affine_3d_x2(dkk, dii, djj, 1, affine) - j


def consolidate_2d(floating[:,:,:] field, floating[:,:] affine_idx, 
                   floating[:,:] affine_disp):
    cdef:
        int nrows = field.shape[0]
        int ncols = field.shape[1]
        int i, j
        double di, dj, dii, djj
        floating[:, :, :] output = np.zeros(shape=(nrows, ncols, 2), 
                                        dtype=np.asarray(field).dtype)

    for i in range(nrows):
        for j in range(ncols):
            di = field[i, j, 0]
            dj = field[i, j, 1]

            #premultiply displacement
            if not affine_disp is None:
                dii = _apply_affine_2d_x0(di, dj, 0, affine_disp)
                djj = _apply_affine_2d_x1(di, dj, 0, affine_disp)
            else:
                dii = di
                djj = dj

            #premultiply index
            if not affine_idx is None:
                di = _apply_affine_2d_x0(i, j, 1, affine_idx)
                dj = _apply_affine_2d_x1(i, j, 1, affine_idx)
            else:
                di = i
                dj = j

            output[i, j, 0] = dii + di - i
            output[i, j, 1] = djj + dj - j

    return output


def consolidate_3d(floating[:,:,:,:] field, floating[:,:] affine_idx, 
                   floating[:,:] affine_disp):
    cdef:
        int nslices = field.shape[0]
        int nrows = field.shape[1]
        int ncols = field.shape[2]
        int i, j, k
        double di, dj, dk, dii, djj, dkk
        floating[:, :, :, :] output = np.zeros(shape=(nslices, nrows, ncols, 3), 
                                        dtype=np.asarray(field).dtype)
    for k in range(nslices):    
        for i in range(nrows):
            for j in range(ncols):
                dk = field[k, i, j, 0]
                di = field[k, i, j, 1]
                dj = field[k, i, j, 2]

                #premultiply displacement
                if not affine_disp is None:
                    dkk = _apply_affine_3d_x0(dk, di, dj, 0, affine_disp)
                    dii = _apply_affine_3d_x1(dk ,di, dj, 0, affine_disp)
                    djj = _apply_affine_3d_x2(dk, di, dj, 0, affine_disp)
                else:
                    dkk = dk
                    dii = di
                    djj = dj

                #premultiply index
                if not affine_idx is None:
                    dk = _apply_affine_3d_x0(k, i, j, 1, affine_idx)
                    di = _apply_affine_3d_x1(k, i, j, 1, affine_idx)
                    dj = _apply_affine_3d_x2(k, i, j, 1, affine_idx)
                else:
                    dk = k
                    di = i
                    dj = j

                output[k, i, j, 0] = dkk + dk - k
                output[k, i, j, 1] = dii + di - i
                output[k, i, j, 2] = djj + dj - j

    return output


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
        int nrows = target_shape[0]
        int ncols = target_shape[1]
        int i, j, inside
        floating dii, djj
        floating[:, :, :] up = np.zeros(shape=(nrows, ncols, 2), 
                                        dtype=np.asarray(field).dtype)
    with nogil:

        for i in range(nrows):
            for j in range(ncols):
                dii = 0.5 * i
                djj = 0.5 * j
                inside = interpolate_vector_bilinear(field, dii, djj, up[i,j])
    return up


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
        int ns = target_shape[0]
        int nr = target_shape[1]
        int nc = target_shape[2]
        int i, j, k
        floating dkk, dii, djj
        floating[:, :, :, :] up = np.zeros(shape=(ns, nr, nc, 3), 
                                           dtype=np.asarray(field).dtype)

    with nogil:

        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    dkk = 0.5 * k
                    dii = 0.5 * i
                    djj = 0.5 * j
                    interpolate_vector_trilinear(field, dkk, dii, djj, up[k, i, j])
    return up


def accumulate_upsample_displacement_field3D(floating[:, :, :, :] field,
                                             floating[:, :, :, :] up):
    r"""
    Upsamples de input 3-D displacement field by a factor of 2. The resulting
    upsampled field is added to 'up' rather than returning a new field.

    Parameters
    ----------
    field : array, shape (S, R, C, 3)
        the 3-D displacement field to be upsampled
    up : array, shape (S', R', C', 3)
        the starting field wich the result will be added to

    """
    cdef:
        int ns = up.shape[0]
        int nr = up.shape[1]
        int nc = up.shape[2]
        int i, j, k, inside
        floating dkk, dii, djj
        floating[:] tmp = np.zeros((3,), dtype = np.asarray(field).dtype)
    with nogil:

        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    dkk = 0.5 * k
                    dii = 0.5 * i
                    djj = 0.5 * j
                    inside = interpolate_vector_trilinear(field, dkk, dii, djj, tmp)
                    if inside == 1:
                        up[k, i, j, 0] += tmp[0] 
                        up[k, i, j, 1] += tmp[1] 
                        up[k, i, j, 2] += tmp[2] 
    return up


def downsample_scalar_field3D(floating[:, :, :] field):
    r"""
    Downsamples the input volume by a factor of 2. The value at each voxel
    of the resulting volume is the average of its surrounding voxels in the
    original volume.

    Parameters
    ----------
    field : array, shape (S, R, C)
        the volume to be downsampled

    Returns
    -------
    down : array, shape (S', R', C')
        the downsampled displacement field, where S' = ceil(S/2), 
        R'= ceil(R/2), C'=ceil(C/2)
    """
    cdef:
        int ns = field.shape[0]
        int nr = field.shape[1]
        int nc = field.shape[2]
        int nns = (ns + 1) // 2
        int nnr = (nr + 1) // 2
        int nnc = (nc + 1) // 2
        int i, j, k, ii, jj, kk
        floating[:, :, :] down = np.zeros((nns, nnr, nnc), dtype=np.asarray(field).dtype)
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
    return down


def downsample_displacement_field3D(floating[:, :, :, :] field):
    r"""
    Downsamples the input vector field by a factor of 2. The value at each voxel
    of the resulting volume is the average of its surrounding voxels in the
    original volume.

    Parameters
    ----------
    field : array, shape (S, R, C)
        the vector field to be downsampled

    Returns
    -------
    down : array, shape (S', R', C')
        the downsampled displacement field, where S' = ceil(S/2), 
        R'= ceil(R/2), C'=ceil(C/2)
    """
    cdef:
        int ns = field.shape[0]
        int nr = field.shape[1]
        int nc = field.shape[2]
        int nns = (ns + 1) // 2
        int nnr = (nr + 1) // 2
        int nnc = (nc + 1) // 2
        int i, j, k, ii, jj, kk
        floating[:, :, :, :] down = np.zeros((nns, nnr, nnc, 3), dtype=np.asarray(field).dtype)
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
    return down


def downsample_scalar_field2D(floating[:, :] field):
    r"""
    Downsamples the input image by a factor of 2. The value at each pixel
    of the resulting image is the average of its surrounding pixels in the
    original image.

    Parameters
    ----------
    field : array, shape (R, C)
        the image to be downsampled

    Returns
    -------
    down : array, shape (R', C')
        the downsampled displacement field, where R'= ceil(R/2), C'=ceil(C/2) 
    """
    cdef:
        int nr = field.shape[0]
        int nc = field.shape[1]
        int nnr = (nr + 1) // 2
        int nnc = (nc + 1) // 2
        int i, j, ii, jj
        floating[:, :] down = np.zeros(shape=(nnr, nnc), dtype=np.asarray(field).dtype)
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
    return down


def downsample_displacement_field2D(floating[:, :, :] field):
    r"""
    Downsamples the input vector field by a factor of 2. The value at each pixel
    of the resulting field is the average of its surrounding pixels in the
    original field.

    Parameters
    ----------
    field : array, shape (R, C)
        the vector field to be downsampled

    Returns
    -------
    down : array, shape (R', C')
        the downsampled displacement field, where R'= ceil(R/2), C'=ceil(C/2), 
    """
    cdef:
        int nr = field.shape[0]
        int nc = field.shape[1]
        int nnr = (nr + 1) // 2
        int nnc = (nc + 1) // 2
        int i, j, ii, jj
        floating[:, :, :] down = np.zeros((nnr, nnc, 2), dtype=np.asarray(field).dtype)
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
    return down


def get_displacement_range(floating[:, :, :, :] d, floating[:, :] affine):
    r"""
    Computes the minimum and maximum values reached by the transformation
    defined by the given displacement field and affine pre-multiplication
    matrix. More precisely, computes max_{x\in L} x+d(A*x), and 
    min_{x\in L} x+d(A*x), where d is the displacement field, A is the affine 
    matrix, the interpolation used is trilinear and the maximum and minimum are 
    taken for each vector component independently.

    Parameters
    ----------
    d : array, shape (S, R, C, 3)
        the displacement field part of the transformation
    affine : array, shape (4, 4)
        the affine pre-multiplication part of the transformation

    Returns
    -------
    minVal : array, shape (3,)
        the minimum value reached at each coordinate
    maxVal : array, shape (3,)
        the maximum value reached at each coordinate
    """
    cdef:
        int nslices = d.shape[0]
        int nrows = d.shape[1]
        int ncols = d.shape[2]
        int i, j, k
        floating dkk, dii, djj
        floating[:] minVal = np.ndarray((3,), dtype=np.asarray(d).dtype)
        floating[:] maxVal = np.ndarray((3,), dtype=np.asarray(d).dtype)
    minVal[...] = d[0, 0, 0, :]
    maxVal[...] = minVal[...]

    with nogil:
        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if(affine != None):
                        dkk = _apply_affine_3d_x0(k, i, j, 1, affine) + d[k, i, j, 0]
                        dii = _apply_affine_3d_x1(k, i, j, 1, affine) + d[k, i, j, 1]
                        djj = _apply_affine_3d_x2(k, i, j, 1, affine) + d[k, i, j, 2]
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


def warp_volume(floating[:, :, :] volume, floating[:, :, :, :] d1,
                floating[:, :] affine_idx_in=None, 
                floating[:, :] affine_idx_out=None, 
                floating[:, :] affine_disp=None,
                int[:] sampling_shape=None):
    r"""
    Deforms the input volume under the transformation T of the from
    T(x) = B*f(A*x), x\in dom(f), where 
    A = affinePre
    B = affinePost
    f = d2
    using trilinear interpolation. If either affine matrix is None, it is
    taken as the identity.

    Parameters
    ----------
    volume : array, shape (S, R, C)
        the input volume to be transformed
    d1 : array, shape (S', R', C', 3)
        the displacement field driving the transformation
    affinePre : array, shape (4, 4)
        the pre-multiplication affine matrix (A, in the model above)
    affinePost : array, shape (4, 4)
        the post-multiplication affine matrix (B, in the model above)

    Returns
    -------
    warped : array, shape (S', R', C')
        the transformed volume
    """
    cdef:
        int nslices = volume.shape[0]
        int nrows = volume.shape[1]
        int ncols = volume.shape[2]
        int nsVol = volume.shape[0]
        int nrVol = volume.shape[1]
        int ncVol = volume.shape[2]
        int i, j, k, inside
        double dkk, dii, djj, dk, di, dj
    if sampling_shape is not None:
        nslices = sampling_shape[0]
        nrows = sampling_shape[1]
        ncols = sampling_shape[2]
    elif d1 is not None:
        nslices = d1.shape[0]
        nrows = d1.shape[1]
        ncols = d1.shape[2]

    cdef floating[:, :, :] warped = np.zeros(shape=(nslices, nrows, ncols), 
                                             dtype=np.asarray(volume).dtype)
    cdef floating[:] tmp = np.zeros(shape=(3,), dtype = np.asarray(volume).dtype)

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
                        inside = interpolate_vector_trilinear(d1, dk, di, dj, tmp)
                        dkk = tmp[0]
                        dii = tmp[1]
                        djj = tmp[2]

                    if not affine_disp is None:
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
                    
                    if not affine_idx_out is None:
                        dkk = dk + _apply_affine_3d_x0(k, i, j, 1, affine_idx_out)
                        dii = di + _apply_affine_3d_x1(k, i, j, 1, affine_idx_out)
                        djj = dj + _apply_affine_3d_x2(k, i, j, 1, affine_idx_out)
                    else:
                        dkk = dk + k
                        dii = di + i
                        djj = dj + j

                    inside = interpolate_scalar_trilinear(volume, dkk, dii, djj, &warped[k,i,j])
    return warped


def warp_volume_affine(floating[:, :, :] volume, int[:] refShape,
                       floating[:, :] affine):
    r"""
    Deforms the input volume under the given affine transformation using 
    trilinear interpolation. The shape of the resulting transformation
    is given by refShape. If the affine matrix is None, it is taken as the 
    identity.

    Parameters
    ----------
    volume : array, shape (S, R, C)
        the input volume to be transformed
    refShape : array, shape (3,)
        the shape of the resulting warped volume
    affine : array, shape (4, 4)
        the affine matrix driving the transformation

    Returns
    -------
    warped : array, shape (S', R', C')
        the transformed volume

    Notes
    -----
    The reason it is necessary to provide the intended shape of the resulting
    warped volume is because the affine transformation is defined on all R^{3}
    but we must sample a finite lattice. Also the resulting shape may not be
    necessarily equal to the input shape, unless we are interested on 
    endomorphisms only and not general diffeomorphisms.
    """
    cdef:
        int nslices = refShape[0]
        int nrows = refShape[1]
        int ncols = refShape[2]
        int nsVol = volume.shape[0]
        int nrVol = volume.shape[1]
        int ncVol = volume.shape[2]
        int i, j, k, ii, jj, kk, inside
        double dkk, dii, djj, tmp0, tmp1
        double alpha, beta, gamma, calpha, cbeta, cgamma
        floating[:, :, :] warped = np.zeros(shape=(nslices, nrows, ncols), 
                                                 dtype=np.asarray(volume).dtype)
    with nogil:

        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if(affine != None):
                        dkk = _apply_affine_3d_x0(k, i, j, 1, affine)
                        dii = _apply_affine_3d_x1(k, i, j, 1, affine)
                        djj = _apply_affine_3d_x2(k, i, j, 1, affine)
                    else:
                        dkk = k
                        dii = i
                        djj = j
                    inside = interpolate_scalar_trilinear(volume, dkk, dii, djj, &warped[k,i,j])
    return warped


def warp_volume_nn(floating[:, :, :] volume, floating[:, :, :, :] d1,
                   floating[:, :] affine_idx_in=None, 
                   floating[:, :] affine_idx_out=None, 
                   floating[:, :] affine_disp=None,
                   int[:] sampling_shape=None):
    r"""
    Deforms the input volume under the transformation T of the from
    T(x) = B*f(A*x), x\in dom(f), where 
    A = affinePre
    B = affinePost
    f = d2
    using nearest neighbor interpolation. If either affine matrix is None, it is
    taken as the identity.

    Parameters
    ----------
    volume : array, shape (S, R, C)
        the input volume to be transformed
    d1 : array, shape (S', R', C', 3)
        the displacement field driving the transformation
    affinePre : array, shape (4, 4)
        the pre-multiplication affine matrix (A, in the model above)
    affinePost : array, shape (4, 4)
        the post-multiplication affine matrix (B, in the model above)

    Returns
    -------
    warped : array, shape (S', R', C')
        the transformed volume
    """
    cdef:
        int nslices = volume.shape[0]
        int nrows = volume.shape[1]
        int ncols = volume.shape[2]
        int nsVol = volume.shape[0]
        int nrVol = volume.shape[1]
        int ncVol = volume.shape[2]
        int i, j, k, inside
        double dkk, dii, djj, dk, di, dj
    if sampling_shape is not None:
        nslices = sampling_shape[0]
        nrows = sampling_shape[1]
        ncols = sampling_shape[2]
    elif d1 is not None:
        nslices = d1.shape[0]
        nrows = d1.shape[1]
        ncols = d1.shape[2]

    cdef floating[:, :, :] warped = np.zeros(shape=(nslices, nrows, ncols), 
                                             dtype=np.asarray(volume).dtype)
    cdef floating[:] tmp = np.zeros(shape=(3,), dtype = np.asarray(volume).dtype)

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
                        inside = interpolate_vector_trilinear(d1, dk, di, dj, tmp)
                        dkk = tmp[0]
                        dii = tmp[1]
                        djj = tmp[2]

                    if not affine_disp is None:
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
                    
                    if not affine_idx_out is None:
                        dkk = dk + _apply_affine_3d_x0(k, i, j, 1, affine_idx_out)
                        dii = di + _apply_affine_3d_x1(k, i, j, 1, affine_idx_out)
                        djj = dj + _apply_affine_3d_x2(k, i, j, 1, affine_idx_out)
                    else:
                        dkk = dk + k
                        dii = di + i
                        djj = dj + j

                    inside = interpolate_scalar_nn_3d(volume, dkk, dii, djj, &warped[k,i,j])
    return warped


def warp_volume_affine_nn(number[:, :, :] volume, int[:] refShape,
                          floating[:, :] affine=None):
    r"""
    Deforms the input volume under the given affine transformation using 
    nearest neighbor interpolation. The shape of the resulting transformation
    is given by refShape. If the affine matrix is None, it is taken as the 
    identity.

    Parameters
    ----------
    volume : array, shape (S, R, C)
        the input volume to be transformed
    refShape : array, shape (3,)
        the shape of the resulting warped volume
    affine : array, shape (4, 4)
        the affine matrix driving the transformation

    Returns
    -------
    warped : array, shape (S', R', C')
        the transformed volume

    Notes
    -----
    The reason it is necessary to provide the intended shape of the resulting
    warped volume is because the affine transformation is defined on all R^{3}
    but we must sample a finite lattice. Also the resulting shape may not be
    necessarily equal to the input shape, unless we are interested on 
    endomorphisms only and not general diffeomorphisms.
    """
    cdef:
        int nslices = refShape[0]
        int nrows = refShape[1]
        int ncols = refShape[2]
        int nsVol = volume.shape[0]
        int nrVol = volume.shape[1]
        int ncVol = volume.shape[2]
        double dkk, dii, djj, tmp0, tmp1
        double alpha, beta, gamma, calpha, cbeta, cgamma
        int k, i, j, kk, ii, jj
        number[:, :, :] warped = np.zeros((nslices, nrows, ncols), 
                                          dtype=np.asarray(volume).dtype)

    with nogil:

        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if(affine != None):
                        dkk = _apply_affine_3d_x0(k, i, j, 1, affine)
                        dii = _apply_affine_3d_x1(k, i, j, 1, affine)
                        djj = _apply_affine_3d_x2(k, i, j, 1, affine)
                    else:
                        dkk = k
                        dii = i
                        djj = j
                    interpolate_scalar_nn_3d(volume, dkk, dii, djj, &warped[k,i,j])
    return warped


def warp_image(floating[:, :] image, floating[:, :, :] d1,
               floating[:,:] affine_idx_in=None,
               floating[:,:] affine_idx_out=None,
               floating[:,:] affine_disp=None,
               int[:] sampling_shape=None):
    r"""
    Deforms the input image under the transformation T of the from
    T(x) = B*f(A*x), x\in dom(f), where 
    A = affinePre
    B = affinePost
    f = d2
    using bilinear interpolation. If either affine matrix is None, it is
    taken as the identity. After simplifying the domain transformation and
    physical transformation products, the final warping is of the form
    warped[i] = image[Tinv*B*A*R*i + Tinv*B*d1[Rinv*A*R*i]]
    where Tinv is the affine transformation bringing physical points to 
    image's discretization, and R, Rinv transform d1's discretization to 
    physical space and physical space to discretization respectively.
    We require affine_idx_in:=Rinv*A*R, affine_idx_out:=Tinv*B*A*R,
    and affine_disp:=Tinv*B

    Parameters
    ----------
    image : array, shape (R, C)
        the input image to be transformed
    d1 : array, shape (R', C', 2)
        the displacement field driving the transformation
    affinePre : array, shape (3, 3)
        the pre-multiplication affine matrix (A, in the model above)
    affinePost : array, shape (3, 3)
        the post-multiplication affine matrix (B, in the model above)

    Returns
    -------
    warped : array, shape (R', C')
        the transformed image
    """
    cdef:
        int nrows = image.shape[0]
        int ncols = image.shape[1]
        int nrVol = image.shape[0]
        int ncVol = image.shape[1]
        int i, j, ii, jj
        double di, dj, dii, djj
    if sampling_shape is not None:
        nrows = sampling_shape[0]
        ncols = sampling_shape[1]
    elif d1 is not None:
        nrows = d1.shape[0]
        ncols = d1.shape[1]
    cdef floating[:, :] warped = np.zeros(shape=(nrows, ncols), 
                                         dtype=np.asarray(image).dtype)
    cdef floating[:] tmp = np.zeros(shape=(2,), dtype = np.asarray(image).dtype)


    with nogil:

        for i in range(nrows):
            for j in range(ncols):
                #Apply inner index premultiplication
                if affine_idx_in is None:
                    dii = d1[i, j, 0]
                    djj = d1[i, j, 1]
                else:
                    di = _apply_affine_2d_x0(
                        i, j, 1, affine_idx_in)
                    dj = _apply_affine_2d_x1(
                        i, j, 1, affine_idx_in)
                    interpolate_vector_bilinear(d1, di, dj, tmp)
                    dii = tmp[0]
                    djj = tmp[1]

                #Apply displacement multiplication 
                if not affine_disp is None:
                    di = _apply_affine_2d_x0(
                        dii, djj, 0, affine_disp)
                    dj = _apply_affine_2d_x1(
                        dii, djj, 0, affine_disp)
                else:
                    di = dii
                    dj = djj

                #Apply outer index multiplization and add the displacements
                if not affine_idx_out is None:
                    dii = di + _apply_affine_2d_x0(i, j, 1, affine_idx_out)
                    djj = dj + _apply_affine_2d_x1(i, j, 1, affine_idx_out)
                else:
                    dii = di + i
                    djj = dj + j

                #Interpolate the input image at the resulting location
                interpolate_scalar_bilinear(image, dii, djj, &warped[i, j])
    return warped


def warp_image_affine(floating[:, :] image, int[:] refShape,
                      floating[:, :] affine=None):
    r"""
    Deforms the input image under the given affine transformation using 
    trilinear interpolation. The shape of the resulting transformation
    is given by refShape. If the affine matrix is None, it is taken as the 
    identity.

    Parameters
    ----------
    image : array, shape (R, C)
        the input image to be transformed
    refShape : array, shape (2,)
        the shape of the resulting warped image
    affine : array, shape (3, 3)
        the affine matrix driving the transformation

    Returns
    -------
    warped : array, shape (R', C')
        the transformed image

    Notes
    -----
    The reason it is necessary to provide the intended shape of the resulting
    warped image is because the affine transformation is defined on all R^{2}
    but we must sample a finite lattice. Also the resulting shape may not be
    necessarily equal to the input shape, unless we are interested on 
    endomorphisms only and not general diffeomorphisms.
    """
    cdef:
        int nrows = refShape[0]
        int ncols = refShape[1]
        int nrVol = image.shape[0]
        int ncVol = image.shape[1]
        int i, j, ii, jj
        double dii, djj, tmp0
        double alpha, beta, calpha, cbeta
        floating[:, :] warped = np.zeros(shape=(nrows, ncols), 
                                         dtype=np.asarray(image).dtype)

    with nogil:

        for i in range(nrows):
            for j in range(ncols):
                if(affine != None):
                    dii = _apply_affine_2d_x0(i, j, 1, affine)
                    djj = _apply_affine_2d_x1(i, j, 1, affine)
                else:
                    dii = i
                    djj = j
                interpolate_scalar_bilinear(image, dii, djj, &warped[i, j])
    return warped


def warp_image_nn(number[:, :] image, floating[:, :, :] d1,
                  floating[:,:] affine_idx_in=None,
                  floating[:,:] affine_idx_out=None,
                  floating[:,:] affine_disp=None,
                  int[:] sampling_shape=None):
    r"""
    Deforms the input image under the transformation T of the from
    T(x) = B*f(A*x), x\in dom(f), where 
    A = affinePre
    B = affinePost
    f = d2
    using neirest neighbor interpolation. If either affine matrix is None, it is
    taken as the identity.

    Parameters
    ----------
    image : array, shape (R, C)
        the input image to be transformed
    d1 : array, shape (R', C', 2)
        the d1 field driving the transformation
    affinePre : array, shape (3, 3)
        the pre-multiplication affine matrix (A, in the model above)
    affinePost : array, shape (3, 3)
        the post-multiplication affine matrix (B, in the model above)

    Returns
    -------
    warped : array, shape (R', C')
        the transformed image
    """
    cdef:
        int nrows = image.shape[0]
        int ncols = image.shape[1]
        int nrVol = image.shape[0]
        int ncVol = image.shape[1]
        int i, j, ii, jj
        double di, dj, dii, djj
    if sampling_shape is not None:
        nrows = sampling_shape[0]
        ncols = sampling_shape[1]
    elif d1 is not None:
        nrows = d1.shape[0]
        ncols = d1.shape[1]
    cdef number[:, :] warped = np.zeros(shape=(nrows, ncols), 
                                         dtype=np.asarray(image).dtype)
    cdef floating[:] tmp = np.zeros(shape=(2,), dtype = np.asarray(image).dtype)


    with nogil:

        for i in range(nrows):
            for j in range(ncols):
                #Apply inner index premultiplication
                if affine_idx_in is None:
                    dii = d1[i, j, 0]
                    djj = d1[i, j, 1]
                else:
                    di = _apply_affine_2d_x0(
                        i, j, 1, affine_idx_in)
                    dj = _apply_affine_2d_x1(
                        i, j, 1, affine_idx_in)
                    interpolate_vector_bilinear(d1, di, dj, tmp)
                    dii = tmp[0]
                    djj = tmp[1]

                #Apply displacement multiplication 
                if not affine_disp is None:
                    di = _apply_affine_2d_x0(
                        dii, djj, 0, affine_disp)
                    dj = _apply_affine_2d_x1(
                        dii, djj, 0, affine_disp)
                else:
                    di = dii
                    dj = djj

                #Apply outer index multiplization and add the displacements
                if not affine_idx_out is None:
                    dii = di + _apply_affine_2d_x0(i, j, 1, affine_idx_out)
                    djj = dj + _apply_affine_2d_x1(i, j, 1, affine_idx_out)
                else:
                    dii = di + i
                    djj = dj + j

                #Interpolate the input image at the resulting location
                interpolate_scalar_nn_2d(image, dii, djj, &warped[i, j])
    return warped


def warp_image_affine_nn(number[:, :] image, int[:] refShape,
                         floating[:, :] affine=None):
    r"""
    Deforms the input image under the given affine transformation using 
    nearest neighbor interpolation. The shape of the resulting transformation
    is given by refShape. If the affine matrix is None, it is taken as the 
    identity.

    Parameters
    ----------
    image : array, shape (R, C)
        the input image to be transformed
    refShape : array, shape (2,)
        the shape of the resulting warped image
    affine : array, shape (3, 3)
        the affine matrix driving the transformation

    Returns
    -------
    warped : array, shape (R', C')
        the transformed image

    Notes
    -----
    The reason it is necessary to provide the intended shape of the resulting
    warped image is because the affine transformation is defined on all R^{2}
    but we must sample a finite lattice. Also the resulting shape may not be
    necessarily equal to the input shape, unless we are interested on 
    endomorphisms only and not general diffeomorphisms.
    """
    cdef:
        int nrows = refShape[0]
        int ncols = refShape[1]
        int nrVol = image.shape[0]
        int ncVol = image.shape[1]
        double dii, djj, tmp0
        double alpha, beta, calpha, cbeta
        int i, j, ii, jj
        number[:, :] warped = np.zeros((nrows, ncols), 
                                       dtype=np.asarray(image).dtype)
    with nogil:

        for i in range(nrows):
            for j in range(ncols):
                if(affine != None):
                    dii = _apply_affine_2d_x0(i, j, 1, affine)
                    djj = _apply_affine_2d_x1(i, j, 1, affine)
                else:
                    dii = i
                    djj = j
                interpolate_scalar_nn_2d(image, dii, djj, &warped[i,j])
    return warped

def warp_2d_stream_line(floating[:, :] streamline, floating[:, :, :] d1,
                        floating[:, :] affinePre=None, 
                        floating[:, :] affinePost=None):
    r"""
    Deforms the input 2d stream line under the transformation T of the from
    T(x) = B*f(A*x), x\in dom(f), where 
    A = affinePre
    B = affinePost
    f = d2
    using bilinear interpolation. If either affine matrix is None, it is
    taken as the identity.

    Parameters
    ----------
    streamline : array, shape (n, 2)
        the input n-point streamline to be transformed
    d1 : array, shape (R', C', 2)
        the displacement field driving the transformation
    affinePre : array, shape (3, 3)
        the pre-multiplication affine matrix (A, in the model above)
    affinePost : array, shape (3, 3)
        the post-multiplication affine matrix (B, in the model above)

    Returns
    -------
    warped : array, shape (n, 2)
        the transformed streamline
    """
    cdef:
        int nr = d1.shape[0]
        int nc = d1.shape[1]
        int n = streamline.shape[0] 
        double i0, j0, dii, djj
        int ii, jj
    for i in range(n):
        if affinePre is not None:
            i0 = _apply_affine_2d_x0(streamline[i, 0], streamline[i, 1], 1, affinePre)
            j0 = _apply_affine_2d_x1(streamline[i, 0], streamline[i, 1], 1, affinePre)
        else:
            i0 = streamline[i,0]
            j0 = streamline[i,1]
        if((i0 < 0) or (j0 < 0) or (i0 > nr - 1) or (j0 > nc - 1)):
            continue
        ii = int(i0)
        jj = int(j0)
        if((ii < 0) or (jj < 0) or (ii >= nr) or (jj >= nc)):
            continue
        calpha = i0 - ii
        cbeta = j0 - jj
        alpha = 1 - calpha
        beta = 1 - cbeta
        #---top-left
        dii= alpha * beta * d1[ii, jj, 0]
        djj= alpha * beta * d1[ii, jj, 1]
        #---top-right
        jj += 1
        if(jj < nc):
            dii += alpha * cbeta * d1[ii, jj, 0]
            djj += alpha * cbeta * d1[ii, jj, 1]
        #---bottom-right
        ii += 1
        if((ii >= 0) and (jj >= 0) and (ii < nr) and (jj < nc)):
            dii += calpha * cbeta * d1[ii, jj, 0]
            djj += calpha * cbeta * d1[ii, jj, 1]
        #---bottom-left
        jj -= 1
        if((ii >= 0) and (jj >= 0) and (ii < nr) and (jj < nc)):
            dii += calpha * beta * d1[ii, jj, 0]
            djj += calpha * beta * d1[ii, jj, 1]
        dii += streamline[i,0]
        djj += streamline[i,1]
        if(affinePost != None):
            streamline[i, 0] = _apply_affine_2d_x0(dii, djj, 1, affinePost)
            streamline[i, 1] = _apply_affine_2d_x1(dii, djj, 1, affinePost)
        else:
            streamline[i, 0], streamline[i, 1] = dii, djj


def expand_displacement_field_3d(floating[:, :, :, :] field, floating[:] factors, int[:] target_shape):
    cdef:
        int tslices = target_shape[0]
        int trows = target_shape[1]
        int tcols = target_shape[2]
        int inside, k, i, j
        double dkk, dii, djj
        floating[:, :, :, :] expanded = np.zeros((tslices, trows, tcols, 3), dtype=np.asarray(field).dtype)

    for k in range(tslices):
        for i in range(trows):
            for j in range(tcols):
                dkk = <double>k*factors[0]
                dii = <double>i*factors[1]
                djj = <double>j*factors[2]
                interpolate_vector_trilinear(field, dkk, dii, djj, expanded[k, i, j])
    return expanded

def expand_displacement_field_2d(floating[:, :, :] field, floating[:] factors, int[:] target_shape):
    cdef:
        int trows = target_shape[0]
        int tcols = target_shape[1]
        int inside, i, j
        double dii, djj
        floating[:, :, :] expanded = np.zeros((trows, tcols, 2), dtype=np.asarray(field).dtype)

    for i in range(trows):
        for j in range(tcols):
            dii = i*factors[0]
            djj = j*factors[1]
            inside = interpolate_vector_bilinear(field, dii, djj, expanded[i, j])
    return expanded


def create_random_displacement_2d(int[:] from_shape, floating[:,:] input_affine, int[:] to_shape, floating[:,:] output_affine):
    r"""
    Creates a random 2D displacement field mapping points of an input discrete domain 
    (with dimensions given by from_shape) to points of an output discrete domain
    (with shape given by to_shape). The affine matrices bringing discrete coordinates
    to physical space are given by input_affine (for the displacement field
    discretization) and output_affine (for the target discretization)

    Returns
    -------
    output : array, shape = from_shape
        the random displacement field in the physical domain
    int_field : array, shape = from_shape
        the assignment of each point in the input grid to the target grid
    """
    cdef:
        int i, j, ri, rj
        double di, dj, dii, djj
        int[:,:,:] int_field = np.ndarray(tuple(from_shape) + (2,), dtype = np.int32)
        floating[:, :, :] output = np.zeros(tuple(from_shape) + (2,), np.asarray(input_affine).dtype)
        int dom_size = from_shape[0]*from_shape[1]

    #compute the actual displacement field in the physical space
    for i in range(from_shape[0]):
        for j in range(from_shape[1]):
            #randomly choose where each input grid point will be mapped to in the target grid
            ri = np.random.randint(0, to_shape[0])
            rj = np.random.randint(0, to_shape[1])
            int_field[i, j, 0] = ri
            int_field[i, j, 1] = rj
            
            #convert the input point to physical coordinates
            if not input_affine is None:
                di = _apply_affine_2d_x0(i, j, 1, input_affine)
                dj = _apply_affine_2d_x1(i, j, 1, input_affine)
            else:
                di = i
                dj = j
            
            #convert the output point to physical coordinates
            if not output_affine is None:
                dii = _apply_affine_2d_x0(ri, rj, 1, output_affine)
                djj = _apply_affine_2d_x1(ri, rj, 1, output_affine)
            else:
                dii = ri
                djj = rj

            #the displacement vector at (i,j) must be the target point minus the
            #original point, both in physical space

            output[i, j, 0] = dii - di
            output[i, j, 1] = djj - dj

    return output, int_field


def create_linear_displacement_field_2d(int[:] shape, 
                                        floating[:,:] input_affine,
                                        floating[:,:] transform):
    r"""
    Creates a 2D displacement field mapping mapping points from the given grid
    shape to themselves after a linear, invertible transformation is applied 
    to them. The resulting displacement field is an invertible endomorphism and 
    may be used to test inversion algorithms.

    Returns
    -------
    output : array, shape = from_shape
        the random displacement field in the physical domain
    """
    cdef:
        int nrows = shape[0]
        int ncols = shape[1]
        int i, j
        double di, dj, dii, djj
        floating[:, :, :] output = np.zeros(tuple(shape) + (2,), np.asarray(input_affine).dtype)

    #compute the actual displacement field in the physical space
    for i in range(nrows):
        for j in range(ncols):
            
            #convert the input point to physical coordinates
            if not input_affine is None:
                di = _apply_affine_2d_x0(i, j, 1, input_affine)
                dj = _apply_affine_2d_x1(i, j, 1, input_affine)
            else:
                di = i
                dj = j

            #transform the point
            
            if not transform is None:
                dii = _apply_affine_2d_x0(di, dj, 1, transform)
                djj = _apply_affine_2d_x1(di, dj, 1, transform)
            else:
                dii = di
                djj = dj

            #the displacement vector at (i,j) must be the target point minus the
            #original point, both in physical space

            output[i, j, 0] = dii - di
            output[i, j, 1] = djj - dj

    return output


def create_random_displacement_3d(int[:] from_shape, floating[:,:] input_affine, int[:] to_shape, floating[:,:] output_affine):
    r"""
    Creates a random 3D displacement field mapping points of an input discrete domain 
    (with dimensions given by from_shape) to points of an output discrete domain
    (with shape given by to_shape). The affine matrices bringing discrete coordinates
    to physical space are given by input_affine (for the displacement field
    discretization) and output_affine (for the target discretization)

    Returns
    -------
    output : array, shape = from_shape
        the random displacement field in the physical domain
    int_field : array, shape = from_shape
        the assignment of each point in the input grid to the target grid
    """
    cdef:
        int i, j, k, ri, rj, rk
        double di, dj, dii, djj
        int[:,:,:,:] int_field = np.ndarray(tuple(from_shape) + (3,), dtype = np.int32)
        floating[:,:,:,:] output = np.zeros(tuple(from_shape) + (3,), np.asarray(input_affine).dtype)
        int dom_size = from_shape[0]*from_shape[1]*from_shape[2]

    #compute the actual displacement field in the physical space
    for k in range(from_shape[0]):
        for i in range(from_shape[1]):
            for j in range(from_shape[2]):
                #randomly choose where each input grid point will be mapped to in the target grid
                rk = np.random.randint(0, to_shape[0])
                ri = np.random.randint(0, to_shape[1])
                rj = np.random.randint(0, to_shape[2])
                int_field[k, i, j, 0] = rk
                int_field[k, i, j, 1] = ri
                int_field[k, i, j, 2] = rj
                
                #convert the input point to physical coordinates
                if not input_affine is None:
                    dk = _apply_affine_3d_x0(k, i, j, 1, input_affine)
                    di = _apply_affine_3d_x1(k, i, j, 1, input_affine)
                    dj = _apply_affine_3d_x2(k, i, j, 1, input_affine)
                else:
                    dk = k
                    di = i
                    dj = j

                #convert the output point to physical coordinates
                if not output_affine is None:
                    dkk = _apply_affine_3d_x0(rk, ri, rj, 1, output_affine)
                    dii = _apply_affine_3d_x1(rk, ri, rj, 1, output_affine)
                    djj = _apply_affine_3d_x2(rk, ri, rj, 1, output_affine)
                else:
                    dkk = rk
                    dii = ri
                    djj = rj

                #the displacement vector at (i,j) must be the target point minus the
                #original point, both in physical space

                output[k, i, j, 0] = dkk - dk
                output[k, i, j, 1] = dii - di
                output[k, i, j, 2] = djj - dj

    return output, int_field

    