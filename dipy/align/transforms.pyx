#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython

cdef extern from "dpy_math.h" nogil:
    double cos(double)
    double sin(double)
    double log(double)

#This dictionary allows us to get the appropriate transform index from a string
transform_type = {'TRANSLATION':TRANSLATION,
                  'ROTATION':ROTATION,
                  'RIGID':RIGID,
                  'SCALING':SCALING,
                  'AFFINE':AFFINE}


def number_of_parameters(int ttype, int dim):
    r""" Number of parameters of the specified transform type

    Parameters
    ----------
    ttype : int
        the type of the transformation (use transform_type dictionary to map
        transformation name to the associated int )
    dim : int
        the domain dimension of the transformation (either 2 or 3)

    Returns
    -------
    n : int
        the number of parameters of the specified transform type
    """
    return _number_of_parameters(ttype, dim)


cdef int _number_of_parameters(int ttype, int dim) nogil:
    if dim == 2:
        if ttype == TRANSLATION:
            return 2
        elif ttype == ROTATION:
            return 1
        elif ttype == RIGID:
            return 3
        elif ttype == SCALING:
            return 1
        elif ttype == AFFINE:
            return 6
    elif dim == 3:
        if ttype == TRANSLATION:
            return 3
        elif ttype == ROTATION:
            return 3
        elif ttype == RIGID:
            return 6
        elif ttype == SCALING:
            return 1
        elif ttype == AFFINE:
            return 12
    return -1


def eval_jacobian_function(int ttype, int dim, double[:] theta, double[:] x,
                           double[:,:] J):
    r""" Compute the Jacobian of a transformation with given parameters at x

    Parameters
    ----------
    ttype : int
        the type of the transformation (use transform_type dictionary to map
        transformation name to the associated int )
    dim : int
        the domain dimension of the transformation (either 2 or 3)
    theta : array, shape (n,)
        the parameters of the transformation at which to evaluate the Jacobian
    x : array, shape (dim,)
        the point at which to evaluate the Jacobian
    J : array, shape (dim, n)
        the destination matrix for the Jacobian
    """
    with nogil:
        get_jacobian_function(ttype, dim)(theta, x, J)


def param_to_matrix(int ttype, int dim, double[:] theta, double[:,:] T):
    r""" Compute the matrix associated to the given transform and parameters

    Parameters
    ----------
    ttype : int
        the type of the transformation (use transform_type dictionary to map
        transformation name to the associated int )
    dim : int
        the domain dimension of the transformation (either 2 or 3)
    theta : array, shape (n,)
        the transformation parameters
    T : array, shape (dim + 1, dim + 1)
        the buffer to write the transform matrix
    """
    with nogil:
        get_param_to_matrix_function(ttype, dim)(theta, T)


def get_identity_parameters(int ttype, int dim, double[:] theta):
    r""" Gets the parameters corresponding to the identity transform

    Parameters
    ----------
    ttype : int
        the type of the transformation (use transform_type dictionary to map
        transformation name to the associated int )
    dim : int
        the domain dimension of the transformation (either 2 or 3)
    theta : array, shape (n,)
        the buffer to write the identity parameters into
    """
    _get_identity_parameters(ttype, dim, theta)


cdef void _get_identity_parameters(int ttype, int dim, double[:] theta) nogil:
    if dim == 2:
        if ttype == TRANSLATION:
            theta[:2] = 0
        elif ttype == ROTATION:
            theta[0] = 0
        elif ttype == RIGID:
            theta[:3] = 0
        elif ttype == SCALING:
            theta[0] = 1
        elif ttype == AFFINE:
            theta[0], theta[1], theta[2] = 1, 0, 0
            theta[3], theta[4], theta[5] = 0, 1, 0
    elif dim == 3:
        if ttype == TRANSLATION:
            theta[:3] = 0
        elif ttype == ROTATION:
            theta[:3] = 0
        elif ttype == RIGID:
            theta[:6] = 0
        elif ttype == SCALING:
            theta[0] = 1
        elif ttype == AFFINE:
            theta[0], theta[1], theta[2], theta[3] = 1, 0, 0, 0
            theta[4], theta[5], theta[6], theta[7] = 0, 1, 0, 0
            theta[8], theta[9], theta[10], theta[11] = 0, 0, 1, 0


cdef jacobian_function get_jacobian_function(int ttype, int dim) nogil:
    r""" Jacobian function corresponding to the given transform and dimension

    Parameters
    ----------
    ttype : int
        the type of the transformation (use transform_type dictionary to map
        transformation name to the associated int )
    dim : int
        the domain dimension of the transformation (either 2 or 3)
    """
    if dim == 2:
        if ttype == TRANSLATION:
            return _translation_jacobian_2d
        elif ttype == ROTATION:
            return _rotation_jacobian_2d
        elif ttype == RIGID:
            return _rigid_jacobian_2d
        elif ttype == SCALING:
            return _scaling_jacobian_2d
        elif ttype == AFFINE:
            return _affine_jacobian_2d
    elif dim == 3:
        if ttype == TRANSLATION:
            return _translation_jacobian_3d
        elif ttype == ROTATION:
            return _rotation_jacobian_3d
        elif ttype == RIGID:
            return _rigid_jacobian_3d
        elif ttype == SCALING:
            return _scaling_jacobian_3d
        elif ttype == AFFINE:
            return _affine_jacobian_3d
    return NULL


cdef param_to_matrix_function get_param_to_matrix_function(int ttype,
                                                           int dim) nogil:
    r""" Param-to-Matrix function of a given transform and dimension

    Parameters
    ----------
    ttype : int
        the type of the transformation (use transform_type dictionary to map
        transformation name to the associated int )
    dim : int
        the domain dimension of the transformation (either 2 or 3)
    """
    if dim == 2:
        if ttype == TRANSLATION:
            return _translation_matrix_2d
        elif ttype == ROTATION:
            return _rotation_matrix_2d
        elif ttype == RIGID:
            return _rigid_matrix_2d
        elif ttype == SCALING:
            return _scaling_matrix_2d
        elif ttype == AFFINE:
            return _affine_matrix_2d
    elif dim == 3:
        if ttype == TRANSLATION:
            return _translation_matrix_3d
        elif ttype == ROTATION:
            return _rotation_matrix_3d
        elif ttype == RIGID:
            return _rigid_matrix_3d
        elif ttype == SCALING:
            return _scaling_matrix_3d
        elif ttype == AFFINE:
            return _affine_matrix_3d
    return NULL


cdef void _translation_matrix_2d(double[:] theta, double[:,:] R) nogil:
    r""" Matrix associated to the 2D translation transform

    Parameters
    ----------
    theta : array, shape(2,)
        the parameters of the 2D translation transform
    R : array, shape(3, 3)
        the buffer in which to write the translation matrix
    """
    R[0,0], R[0,1], R[0, 2] = 1, 0, theta[0]
    R[1,0], R[1,1], R[1, 2] = 0, 1, theta[1]
    R[2,0], R[2,1], R[2, 2] = 0, 0, 1


cdef void _translation_matrix_3d(double[:] theta, double[:,:] R) nogil:
    r""" Matrix associated to the 3D translation transform

    Parameters
    ----------
    theta : array, shape(3,)
        the parameters of the 3D translation transform
    R : array, shape(4, 4)
        the buffer in which to write the translation matrix
    """
    R[0,0], R[0,1], R[0,2], R[0,3] = 1, 0, 0, theta[0]
    R[1,0], R[1,1], R[1,2], R[1,3] = 0, 1, 0, theta[1]
    R[2,0], R[2,1], R[2,2], R[2,3] = 0, 0, 1, theta[2]
    R[3,0], R[3,1], R[3,2], R[3,3] = 0, 0, 0, 1


cdef int _translation_jacobian_2d(double[:] theta, double[:] x,
                                  double[:,:] J) nogil:
    r""" Jacobian matrix of the 2D translation transform
    The transformation is given by:

    T(x) = (T1(x), T2(x)) = (x0 + t0, x1 + t1)

    The derivative w.r.t. t1 and t2 is given by

    T'(x) = [[1, 0], # derivatives of [T1, T2] w.r.t. t0
             [0, 1]] # derivatives of [T1, T2] w.r.t. t1

    Parameters
    ----------
    theta : array, shape(2,)
        the parameters of the 2D translation transform (the Jacobian does not
        depend on the parameters, but we receive the buffer so all Jacobian
        functions receive the same parameters)
    x : array, shape(2,)
        the point at which to compute the Jacobian (the Jacobian does not
        depend on x, but we receive the buffer so all Jacobian functions
        receive the same parameters)
    J : array, shape(2, 2)
        the buffer in which to write the Jacobian
    """
    J[0,0], J[0, 1] = 1.0, 0.0
    J[1,0], J[1, 1] = 0.0, 1.0
    # This Jacobian does not depend on x (it's constant): return 1
    return 1


cdef int _translation_jacobian_3d(double[:] theta, double[:] x,
                                  double[:,:] J) nogil:
    r""" Jacobian matrix of the 3D translation transform
    The transformation is given by:

    T(x) = (T1(x), T2(x), T3(x)) = (x0 + t0, x1 + t1, x2 + t2)

    The derivative w.r.t. t1, t2 and t3 is given by

    T'(x) = [[1, 0, 0], # derivatives of [T1, T2, T3] w.r.t. t0
             [0, 1, 0], # derivatives of [T1, T2, T3] w.r.t. t1
             [0, 0, 1]] # derivatives of [T1, T2, T3] w.r.t. t2
    Parameters
    ----------
    theta : array, shape(3,)
        the parameters of the 3D translation transform (the Jacobian does not
        depend on the parameters, but we receive the buffer so all Jacobian
        functions receive the same parameters)
    x : array, shape(3,)
        the point at which to compute the Jacobian (the Jacobian does not
        depend on x, but we receive the buffer so all Jacobian functions
        receive the same parameters)
    J : array, shape(3, 3)
        the buffer in which to write the Jacobian
    """
    J[0,0], J[0,1], J[0,2] = 1.0, 0.0, 0.0
    J[1,0], J[1,1], J[1,2] = 0.0, 1.0, 0.0
    J[2,0], J[2,1], J[2,2] = 0.0, 0.0, 1.0
    # This Jacobian does not depend on x (it's constant): return 1
    return 1


cdef void _rotation_matrix_2d(double[:] theta, double[:,:] R) nogil:
    r""" Matrix associated to the 2D rotation transform

    Parameters
    ----------
    theta : array, shape(1,)
        the rotation angle
    R : array, shape(3,3)
        the buffer in which to write the matrix
    """
    cdef:
        double ct = cos(theta[0])
        double st = sin(theta[0])
    R[0,0], R[0,1], R[0,2] = ct, -st, 0
    R[1,0], R[1,1], R[1,2] = st, ct, 0
    R[2,0], R[2,1], R[2,2] = 0, 0, 1


cdef int _rotation_jacobian_2d(double[:] theta, double[:] x,
                               double[:,:] J) nogil:
    r''' Jacobian matrix of a 2D rotation transform with parameter theta, at x

    The transformation is given by:

    T(x,y) = (T1(x,y), T2(x,y)) = (x cost - y sint, x sint + y cost)

    The derivatives w.r.t. the rotation angle, t, are:

    T'(x,y) = [-x sint - y cost, # derivative of T1 w.r.t. t
                x cost - y sint] # derivative of T2 w.r.t. t

    Parameters
    ----------
    theta : array, shape(1,)
        the rotation angle
    x : array, shape(2,)
        the point at which to compute the Jacobian
    J : array, shape(2, 1)
        the buffer in which to write the Jacobian
    '''
    cdef:
        double st = sin(theta[0])
        double ct = cos(theta[0])
        double px = x[0], py = x[1]

    J[0, 0] = -px * st - py * ct
    J[1, 0] = px * ct - py * st
    # This Jacobian depends on x (it's not constant): return 0
    return 0


cdef void _rotation_matrix_3d(double[:] theta, double[:,:] R) nogil:
    r""" Matrix associated to the 3D rotation transform

    The matrix is the product of rotation matrices of angles theta[0], theta[1],
    theta[2] around axes x, y, z applied in the following order: y, x, z.
    This order was chosen for consistency with ANTS.

    Parameters
    ----------
    theta : array, shape(3,)
        the rotation angles about each axis:
        theta[0] : rotation angle around x axis
        theta[1] : rotation angle around y axis
        theta[2] : rotation angle around z axis
    R : array, shape(4, 4)
        buffer in which to write the rotation matrix
    """
    cdef:
        double sa = sin(theta[0])
        double ca = cos(theta[0])
        double sb = sin(theta[1])
        double cb = cos(theta[1])
        double sc = sin(theta[2])
        double cc = cos(theta[2])

    R[0,0], R[0,1], R[0,2], R[0, 3] = cc*cb-sc*sa*sb, -sc*ca, cc*sb+sc*sa*cb, 0
    R[1,0], R[1,1], R[1,2], R[1, 3] = sc*cb+cc*sa*sb, cc*ca, sc*sb-cc*sa*cb, 0
    R[2,0], R[2,1], R[2,2], R[2, 3] = -ca*sb, sa, ca*cb, 0
    R[3,0], R[3,1], R[3,2], R[3, 3] = 0, 0, 0, 1


cdef int _rotation_jacobian_3d(double[:] theta, double[:] x,
                               double[:,:] J) nogil:
    r''' Jacobian matrix of a 3D rotation transform with parameters theta, at x

    Parameters
    ----------
    theta : array, shape(3,)
        the rotation angles about the canonical axes
    x : array, shape(3,)
        the point at which to compute the Jacobian
    J : array, shape(3, 3)
        the buffer in which to write the Jacobian
    '''
    cdef:
        double sa = sin(theta[0])
        double ca = cos(theta[0])
        double sb = sin(theta[1])
        double cb = cos(theta[1])
        double sc = sin(theta[2])
        double cc = cos(theta[2])
        double px = x[0], py = x[1], pz = x[2]

    J[0, 0] = ( -sc * ca * sb ) * px + ( sc * sa ) * py + ( sc * ca * cb ) * pz
    J[1, 0] = ( cc * ca * sb ) * px + ( -cc * sa ) * py + ( -cc * ca * cb ) * pz
    J[2, 0] = ( sa * sb ) * px + ( ca ) * py + ( -sa * cb ) * pz

    J[0, 1] = ( -cc * sb - sc * sa * cb ) * px + ( cc * cb - sc * sa * sb ) * pz
    J[1, 1] = ( -sc * sb + cc * sa * cb ) * px + ( sc * cb + cc * sa * sb ) * pz
    J[2, 1] = ( -ca * cb ) * px + ( -ca * sb ) * pz

    J[0, 2] = ( -sc * cb - cc * sa * sb ) * px + ( -cc * ca ) * py + \
              ( -sc * sb + cc * sa * cb ) * pz
    J[1, 2] = ( cc * cb - sc * sa * sb ) * px + ( -sc * ca ) * py + \
              ( cc * sb + sc * sa * cb ) * pz
    J[2, 2] = 0
    # This Jacobian depends on x (it's not constant): return 0
    return 0


cdef void _rigid_matrix_2d(double[:] theta, double[:,:] R) nogil:
    r""" Matrix associated to the 2D rigid transform (rotation + translation)

    Parameters
    ----------
    theta : array, shape(3,)
        the parameters of the 2D rigid transform
        theta[0] : rotation angle
        theta[1] : translation along the x axis
        theta[2] : translation along the y axis
    R : array, shape(3, 3)
        buffer in which to write the rigid matrix
    """
    cdef:
        double ct = cos(theta[0])
        double st = sin(theta[0])
    R[0,0], R[0,1], R[0,2] = ct, -st, theta[1]
    R[1,0], R[1,1], R[1,2] = st, ct, theta[2]
    R[2,0], R[2,1], R[2,2] = 0, 0, 1


cdef int _rigid_jacobian_2d(double[:] theta, double[:] x, double[:,:] J) nogil:
    r''' Jacobian matrix of a 2D rigid transform (rotation + translation)

    The transformation is given by:

    T(x,y) = (T1(x,y), T2(x,y)) = (x cost - y sint + dx, x sint + y cost + dy)

    The derivatives w.r.t. t, dx and dy are:

    T'(x,y) = [-x sint - y cost, 1, 0, # derivative of T1 w.r.t. t, dx, dy
                x cost - y sint, 0, 1] # derivative of T2 w.r.t. t, dx, dy

    Parameters
    ----------
    theta : array, shape(1,)
        the parameters of the 2D rigid transform
        theta[0] : rotation angle (t)
        theta[1] : translation along the x axis (dx)
        theta[2] : translation along the y axis (dy)
    x : array, shape(2,)
        the point at which to compute the Jacobian
    J : array, shape(2, 3)
        the buffer in which to write the Jacobian
    '''
    cdef:
        double st = sin(theta[0])
        double ct = cos(theta[0])
        double px = x[0], py = x[1]

    J[0, 0], J[0, 1], J[0, 2] = -px * st - py * ct, 1, 0
    J[1, 0], J[1, 1], J[1, 2] = px * ct - py * st, 0, 1
    # This Jacobian depends on x (it's not constant): return 0
    return 0


cdef void _rigid_matrix_3d(double[:] theta, double[:,:] R) nogil:
    r""" Matrix associated to the 3D rigid transform (rotation + translation)

    Parameters
    ----------
    theta : array, shape(6,)
        the parameters of the 3D rigid transform
        theta[0] : rotation about the x axis
        theta[1] : rotation about the y axis
        theta[2] : rotation about the z axis
        theta[3] : translation along the x axis
        theta[4] : translation along the y axis
        theta[5] : translation along the z axis
    R : array, shape(4, 4)
        buffer in which to write the rigid matrix
    """
    cdef:
        double sa = sin(theta[0])
        double ca = cos(theta[0])
        double sb = sin(theta[1])
        double cb = cos(theta[1])
        double sc = sin(theta[2])
        double cc = cos(theta[2])
        double dx = theta[3]
        double dy = theta[4]
        double dz = theta[5]

    R[0,0], R[0,1], R[0,2], R[0,3] = cc*cb-sc*sa*sb, -sc*ca, cc*sb+sc*sa*cb, dx
    R[1,0], R[1,1], R[1,2], R[1,3] = sc*cb+cc*sa*sb, cc*ca, sc*sb-cc*sa*cb, dy
    R[2,0], R[2,1], R[2,2], R[2,3] = -ca*sb, sa, ca*cb, dz
    R[3,0], R[3,1], R[3,2], R[3,3] = 0, 0, 0, 1


cdef int _rigid_jacobian_3d(double[:] theta, double[:] x, double[:,:] J) nogil:
    r''' Jacobian matrix of a 3D rigid transform (rotation + translation)

    Parameters
    ----------
    theta : array, shape(6,)
        the parameters of the 3D rigid transform
        theta[0] : rotation about the x axis
        theta[1] : rotation about the y axis
        theta[2] : rotation about the z axis
        theta[3] : translation along the x axis
        theta[4] : translation along the y axis
        theta[5] : translation along the z axis
    x : array, shape(3,)
        the point at which to compute the Jacobian
    J : array, shape(3, 6)
        the buffer in which to write the Jacobian
    '''
    cdef:
        double sa = sin(theta[0])
        double ca = cos(theta[0])
        double sb = sin(theta[1])
        double cb = cos(theta[1])
        double sc = sin(theta[2])
        double cc = cos(theta[2])
        double px = x[0], py = x[1], pz = x[2]

    J[0, 0] = ( -sc * ca * sb ) * px + ( sc * sa ) * py + ( sc * ca * cb ) * pz
    J[1, 0] = ( cc * ca * sb ) * px + ( -cc * sa ) * py + ( -cc * ca * cb ) * pz
    J[2, 0] = ( sa * sb ) * px + ( ca ) * py + ( -sa * cb ) * pz

    J[0, 1] = ( -cc * sb - sc * sa * cb ) * px + ( cc * cb - sc * sa * sb ) * pz
    J[1, 1] = ( -sc * sb + cc * sa * cb ) * px + ( sc * cb + cc * sa * sb ) * pz
    J[2, 1] = ( -ca * cb ) * px + ( -ca * sb ) * pz

    J[0, 2] = ( -sc * cb - cc * sa * sb ) * px + ( -cc * ca ) * py + \
              ( -sc * sb + cc * sa * cb ) * pz
    J[1, 2] = ( cc * cb - sc * sa * sb ) * px + ( -sc * ca ) * py + \
              ( cc * sb + sc * sa * cb ) * pz
    J[2, 2] = 0

    J[0,3:6] = 0
    J[1,3:6] = 0
    J[2,3:6] = 0
    J[0,3], J[1,4], J[2,5] = 1, 1, 1
    # This Jacobian depends on x (it's not constant): return 0
    return 0


cdef void _scaling_matrix_2d(double[:] theta, double[:,:] R) nogil:
    r""" Matrix associated to the 2D (isotropic) scaling transform

    Parameters
    ----------
    theta : array, shape(1,)
        the scale factor
    R : array, shape(3, 3)
        the buffer in which to write the scaling matrix
    """
    R[0,0], R[0,1], R[0, 2] = theta[0], 0, 0
    R[1,0], R[1,1], R[1, 2] = 0, theta[0], 0
    R[2,0], R[2,1], R[2, 2] = 0, 0, 1


cdef void _scaling_matrix_3d(double[:] theta, double[:,:] R) nogil:
    r""" Matrix associated to the 3D (isotropic) scaling transform

    Parameters
    ----------
    theta : array, shape(1,)
        the scale factor
    R : array, shape(4, 4)
        the buffer in which to write the scaling matrix
    """
    R[0,0], R[0,1], R[0,2], R[0,3] = theta[0], 0, 0, 0
    R[1,0], R[1,1], R[1,2], R[1,3] = 0, theta[0], 0, 0
    R[2,0], R[2,1], R[2,2], R[2,3] = 0, 0, theta[0], 0
    R[3,0], R[3,1], R[3,2], R[3,3] = 0, 0, 0, 1


cdef int _scaling_jacobian_2d(double[:] theta, double[:] x,
                              double[:,:] J) nogil:
    r""" Jacobian matrix of the isotropic 2D scale transform
    The transformation is given by:

    T(x) = (s*x0, s*x1)

    The derivative w.r.t. s is T'(x) = [x0, x1]

    Parameters
    ----------
    theta : array, shape(1,)
        the scale factor (the Jacobian does not depend on the scale factor,
        but we receive the buffer to make it consistent with other Jacobian
        functions)
    x : array, shape (2,)
        the point at which to compute the Jacobian
    J : array, shape(2, 1)
        the buffer in which to write the Jacobian
    """
    J[0,0], J[1,0] = x[0], x[1]
    # This Jacobian depends on x (it's not constant): return 0
    return 0


cdef int _scaling_jacobian_3d(double[:] theta, double[:] x,
                              double[:,:] J) nogil:
    r""" Jacobian matrix of the isotropic 3D scale transform
    The transformation is given by:

    T(x) = (s*x0, s*x1, s*x2)

    The derivative w.r.t. s is T'(x) = [x0, x1, x2]

    Parameters
    ----------
    theta : array, shape(1,)
        the scale factor (the Jacobian does not depend on the scale factor,
        but we receive the buffer to make it consistent with other Jacobian
        functions)
    x : array, shape (3,)
        the point at which to compute the Jacobian
    J : array, shape(3, 1)
        the buffer in which to write the Jacobian
    """
    J[0,0], J[1,0], J[2,0]= x[0], x[1], x[2]
    # This Jacobian depends on x (it's not constant): return 0
    return 0


cdef void _affine_matrix_2d(double[:] theta, double[:,:] R) nogil:
    r""" Matrix associated to a general 2D affine transform

    The transformation is given by the matrix:

    A = [[a0, a1, a2],
         [a3, a4, a5],
         [ 0,  0,  1]]

    Parameters
    ----------
    theta : array, shape(6,)
        the parameters of the 2D affine transform
    R : array, shape(3,3)
        the buffer in which to write the matrix
    """
    R[0,0], R[0,1], R[0, 2] = theta[0], theta[1], theta[2]
    R[1,0], R[1,1], R[1, 2] = theta[3], theta[4], theta[5]
    R[2,0], R[2,1], R[2, 2] = 0, 0, 1


cdef void _affine_matrix_3d(double[:] theta, double[:,:] R) nogil:
    r""" Matrix associated to a general 3D affine transform

    The transformation is given by the matrix:

    A = [[a0, a1, a2, a3],
         [a4, a5, a6, a7],
         [a8, a9, a10, a11],
         [ 0,   0,   0,   1]]

    Parameters
    ----------
    theta : array, shape(12,)
        the parameters of the 3D affine transform
    R : array, shape(4,4)
        the buffer in which to write the matrix
    """
    R[0,0], R[0,1], R[0,2], R[0,3] = theta[0], theta[1], theta[2], theta[3]
    R[1,0], R[1,1], R[1,2], R[1,3] = theta[4], theta[5], theta[6], theta[7]
    R[2,0], R[2,1], R[2,2], R[2,3] = theta[8], theta[9], theta[10], theta[11]
    R[3,0], R[3,1], R[3,2], R[3,3] = 0, 0, 0, 1


cdef int _affine_jacobian_2d(double[:] theta, double[:] x, double[:,:] J) nogil:
    r""" Jacobian matrix of the 2D affine transform
    The transformation is given by:

    T(x) = |a0, a1, a2 |   |x0|   | T1(x) |   |a0*x0 + a1*x1 + a2|
           |a3, a4, a5 | * |x1| = | T2(x) | = |a3*x0 + a4*x1 + a5|
                           | 1|

    The derivatives w.r.t. each parameter are given by

    T'(x) = [[x0,  0], #derivatives of [T1, T2] w.r.t a0
             [x1,  0], #derivatives of [T1, T2] w.r.t a1
             [ 1,  0], #derivatives of [T1, T2] w.r.t a2
             [ 0, x0], #derivatives of [T1, T2] w.r.t a3
             [ 0, x1], #derivatives of [T1, T2] w.r.t a4
             [ 0,  1]] #derivatives of [T1, T2, T3] w.r.t a5

    The Jacobian matrix is the transpose of the above matrix.

    Parameters
    ----------
    theta : array, shape(6,)
        the parameters of the 2D affine transform
    x : array, shape (2,)
        the point at which to compute the Jacobian
    J : array, shape(2, 6)
        the buffer in which to write the Jacobian
    """
    J[0,:6] = 0
    J[1,:6] = 0

    J[0, :2] = x[:2]
    J[0, 2] = 1
    J[1, 3:5] = x[:2]
    J[1, 5] = 1
    # This Jacobian depends on x (it's not constant): return 0
    return 0


cdef int _affine_jacobian_3d(double[:] theta, double[:] x, double[:,:] J) nogil:
    r""" Jacobian matrix of the 3D affine transform
    The transformation is given by:

    T(x) = |a0, a1, a2,  a3 |   |x0|   | T1(x) |   |a0*x0 + a1*x1 + a2*x2 + a3|
           |a4, a5, a6,  a7 | * |x1| = | T2(x) | = |a4*x0 + a5*x1 + a6*x2 + a7|
           |a8, a9, a10, a11|   |x2|   | T3(x) |   |a8*x0 + a9*x1 + a10*x2+a11|
                                | 1|

    The derivatives w.r.t. each parameter are given by

    T'(x) = [[x0,  0,  0], #derivatives of [T1, T2, T3] w.r.t a0
             [x1,  0,  0], #derivatives of [T1, T2, T3] w.r.t a1
             [x2,  0,  0], #derivatives of [T1, T2, T3] w.r.t a2
             [ 1,  0,  0], #derivatives of [T1, T2, T3] w.r.t a3
             [ 0, x0,  0], #derivatives of [T1, T2, T3] w.r.t a4
             [ 0, x1,  0], #derivatives of [T1, T2, T3] w.r.t a5
             [ 0, x2,  0], #derivatives of [T1, T2, T3] w.r.t a6
             [ 0,  1,  0], #derivatives of [T1, T2, T3] w.r.t a7
             [ 0,  0, x0], #derivatives of [T1, T2, T3] w.r.t a8
             [ 0,  0, x1], #derivatives of [T1, T2, T3] w.r.t a9
             [ 0,  0, x2], #derivatives of [T1, T2, T3] w.r.t a10
             [ 0,  0,  1]] #derivatives of [T1, T2, T3] w.r.t a11

    The Jacobian matrix is the transpose of the above matrix.

    Parameters
    ----------
    theta : array, shape(12,)
        the parameters of the 3D affine transform
    x : array, shape (3,)
        the point at which to compute the Jacobian
    J : array, shape(3, 12)
        the buffer in which to write the Jacobian
    """
    cdef:
        cnp.npy_intp j

    for j in range(3):
        J[j,:12] = 0
    J[0, :3] = x[:3]
    J[0, 3] = 1
    J[1, 4:7] = x[:3]
    J[1, 7] = 1
    J[2, 8:11] = x[:3]
    J[2, 11] = 1
    # This Jacobian depends on x (it's not constant): return 0
    return 0


cdef void _dot_prod(double[:,:] A, double[:,:] B, double[:,:] C):
    cdef:
        int r = A.shape[0]
        int c = B.shape[1]
        int m = A.shape[1]
        double s
        double[:,:] tmp = np.empty(shape=(r, c), dtype=np.float64)
    with nogil:
        for i in range(r):
            for j in range(c):
                s = 0
                for k in range(m):
                    s += A[i, k] * B[k, j]
                tmp[i, j] = s

        for i in range(r):
            for j in range(c):
                C[i, j] = tmp[i, j]
