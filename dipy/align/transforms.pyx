#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as cnp

cdef extern from "dpy_math.h" nogil:
    double cos(double)
    double sin(double)
    double log(double)

cdef class Transform:
    r""" Base class (contract) for all transforms for affine image registration
    Each transform must define the following (fast, nogil) methods:

    1. _jacobian(theta, x, J): receives a parameter vector theta, a point in
       x, and a matrix J with shape (dim, len(theta)). It must writes in J, the
       Jacobian of the transform with parameters theta evaluated at x.

    2. _get_identity_parameters(theta): receives a vector theta whose length is
       the number of parameters of the transform and sets in theta the values
       that define the identity transform.

    3. _param_to_matrix(theta, T): receives a parameter vector theta, and a
       matrix T of shape (dim + 1, dim + 1) and writes in T the matrix
       representation of the transform with parameters theta

    This base class defines the (slow, convenient) python wrappers for each
    of the above functions, which also do parameter checking and raise
    a ValueError in case the provided parameters are invalid.
    """
    def __cinit__(self):
        r""" Default constructor
        Sets transform dimension and number of parameter to invalid values (-1)
        """
        self.dim = -1
        self.number_of_parameters = -1

    cdef int _jacobian(self, double[:] theta, double[:] x,
                       double[:, :] J) noexcept nogil:
        return -1

    cdef void _get_identity_parameters(self, double[:] theta) noexcept nogil:
        return

    cdef void _param_to_matrix(self, double[:] theta, double[:, :] T) noexcept nogil:
        return

    def jacobian(self, double[:] theta, double[:] x):
        r""" Jacobian function of this transform

        Parameters
        ----------
        theta : array, shape (n,)
            vector containing the n parameters of this transform
        x : array, shape (dim,)
            vector containing the point where the Jacobian must be evaluated

        Returns
        -------
        J : array, shape (dim, n)
            Jacobian matrix of the transform with parameters theta at point x
        """
        n = theta.shape[0]
        if n != self.number_of_parameters:
            raise ValueError("Invalid number of parameters: %d"%(n,))
        m = x.shape[0]
        if m < self.dim:
            raise ValueError("Invalid point dimension: %d"%(m,))
        J = np.zeros((self.dim, n))
        ret = self._jacobian(theta, x, J)
        return np.asarray(J)

    def get_identity_parameters(self):
        r""" Parameter values corresponding to the identity transform

        Returns
        -------
        theta : array, shape (n,)
            the n parameter values corresponding to the identity transform
        """
        if self.number_of_parameters < 0:
            raise ValueError("Invalid transform.")
        theta = np.zeros(self.number_of_parameters)
        self._get_identity_parameters(theta)
        return np.asarray(theta)

    def param_to_matrix(self, double[:] theta):
        r""" Matrix representation of this transform with the given parameters

        Parameters
        ----------
        theta : array, shape (n,)
            the parameter values of the transform

        Returns
        -------
        T : array, shape (dim + 1, dim + 1)
            the matrix representation of this transform with parameters theta
        """
        n = len(theta)
        if n != self.number_of_parameters:
            raise ValueError("Invalid number of parameters: %d"%(n,))
        T = np.eye(self.dim + 1)
        self._param_to_matrix(theta, T)
        return np.asarray(T)

    def get_number_of_parameters(self):
        return self.number_of_parameters

    def get_dim(self):
        return self.dim


cdef class TranslationTransform2D(Transform):
    def __init__(self):
        r""" Translation transform in 2D
        """
        self.dim = 2
        self.number_of_parameters = 2

    cdef int _jacobian(self, double[:] theta, double[:] x,
                       double[:, :] J) noexcept nogil:
        r""" Jacobian matrix of the 2D translation transform
        The transformation is given by:

        T(x) = (T1(x), T2(x)) = (x0 + t0, x1 + t1)

        The derivative w.r.t. t1 and t2 is given by

        T'(x) = [[1, 0], # derivatives of [T1, T2] w.r.t. t0
                 [0, 1]] # derivatives of [T1, T2] w.r.t. t1

        Parameters
        ----------
        theta : array, shape (2,)
            the parameters of the 2D translation transform (the Jacobian does
            not depend on the parameters, but we receive the buffer so all
            Jacobian functions receive the same parameters)
        x : array, shape (2,)
            the point at which to compute the Jacobian (the Jacobian does not
            depend on x, but we receive the buffer so all Jacobian functions
            receive the same parameters)
        J : array, shape (2, 2)
            the buffer in which to write the Jacobian

        Returns
        -------
        is_constant : int
            always returns 1, indicating that the Jacobian is constant
            (independent of x)
        """
        J[0, 0], J[0, 1] = 1.0, 0.0
        J[1, 0], J[1, 1] = 0.0, 1.0
        # This Jacobian does not depend on x (it's constant): return 1
        return 1

    cdef void _get_identity_parameters(self, double[:] theta) noexcept nogil:
        r""" Parameter values corresponding to the identity
        Sets in theta the parameter values corresponding to the identity
        transform

        Parameters
        ----------
        theta : array, shape (2,)
            buffer to write the parameters of the 2D translation transform
        """
        theta[:2] = 0

    cdef void _param_to_matrix(self, double[:] theta, double[:, :] R) noexcept nogil:
        r""" Matrix associated with the 2D translation transform

        Parameters
        ----------
        theta : array, shape (2,)
            the parameters of the 2D translation transform
        R : array, shape (3, 3)
            the buffer in which to write the translation matrix
        """
        R[0, 0], R[0, 1], R[0, 2] = 1, 0, theta[0]
        R[1, 0], R[1, 1], R[1, 2] = 0, 1, theta[1]
        R[2, 0], R[2, 1], R[2, 2] = 0, 0, 1


cdef class TranslationTransform3D(Transform):
    def __init__(self):
        r""" Translation transform in 3D
        """
        self.dim = 3
        self.number_of_parameters = 3

    cdef int _jacobian(self, double[:] theta, double[:] x,
                       double[:, :] J) noexcept nogil:
        r""" Jacobian matrix of the 3D translation transform
        The transformation is given by:

        T(x) = (T1(x), T2(x), T3(x)) = (x0 + t0, x1 + t1, x2 + t2)

        The derivative w.r.t. t1, t2 and t3 is given by

        T'(x) = [[1, 0, 0], # derivatives of [T1, T2, T3] w.r.t. t0
                 [0, 1, 0], # derivatives of [T1, T2, T3] w.r.t. t1
                 [0, 0, 1]] # derivatives of [T1, T2, T3] w.r.t. t2
        Parameters
        ----------
        theta : array, shape (3,)
            the parameters of the 3D translation transform (the Jacobian does
            not depend on the parameters, but we receive the buffer so all
            Jacobian functions receive the same parameters)
        x : array, shape (3,)
            the point at which to compute the Jacobian (the Jacobian does not
            depend on x, but we receive the buffer so all Jacobian functions
            receive the same parameters)
        J : array, shape (3, 3)
            the buffer in which to write the Jacobian

        Returns
        -------
        is_constant : int
            always returns 1, indicating that the Jacobian is constant
            (independent of x)
        """
        J[0, 0], J[0, 1], J[0, 2] = 1.0, 0.0, 0.0
        J[1, 0], J[1, 1], J[1, 2] = 0.0, 1.0, 0.0
        J[2, 0], J[2, 1], J[2, 2] = 0.0, 0.0, 1.0
        # This Jacobian does not depend on x (it's constant): return 1
        return 1

    cdef void _get_identity_parameters(self, double[:] theta) noexcept nogil:
        r""" Parameter values corresponding to the identity
        Sets in theta the parameter values corresponding to the identity
        transform

        Parameters
        ----------
        theta : array, shape (3,)
            buffer to write the parameters of the 3D translation transform
        """
        theta[:3] = 0

    cdef void _param_to_matrix(self, double[:] theta, double[:, :] R) noexcept nogil:
        r""" Matrix associated with the 3D translation transform

        Parameters
        ----------
        theta : array, shape (3,)
            the parameters of the 3D translation transform
        R : array, shape (4, 4)
            the buffer in which to write the translation matrix
        """
        R[0, 0], R[0, 1], R[0, 2], R[0, 3] = 1, 0, 0, theta[0]
        R[1, 0], R[1, 1], R[1, 2], R[1, 3] = 0, 1, 0, theta[1]
        R[2, 0], R[2, 1], R[2, 2], R[2, 3] = 0, 0, 1, theta[2]
        R[3, 0], R[3, 1], R[3, 2], R[3, 3] = 0, 0, 0, 1


cdef class RotationTransform2D(Transform):
    def __init__(self):
        r""" Rotation transform in 2D
        """
        self.dim = 2
        self.number_of_parameters = 1

    cdef int _jacobian(self, double[:] theta, double[:] x,
                       double[:, :] J) noexcept nogil:
        r""" Jacobian matrix of a 2D rotation with parameter theta, at x

        The transformation is given by:

        T(x,y) = (T1(x,y), T2(x,y)) = (x cost - y sint, x sint + y cost)

        The derivatives w.r.t. the rotation angle, t, are:

        T'(x,y) = [-x sint - y cost, # derivative of T1 w.r.t. t
                    x cost - y sint] # derivative of T2 w.r.t. t

        Parameters
        ----------
        theta : array, shape (1,)
            the rotation angle
        x : array, shape (2,)
            the point at which to compute the Jacobian
        J : array, shape (2, 1)
            the buffer in which to write the Jacobian

        Returns
        -------
        is_constant : int
            always returns 0, indicating that the Jacobian is not
            constant (it depends on the value of x)
        """
        cdef:
            double st = sin(theta[0])
            double ct = cos(theta[0])
            double px = x[0], py = x[1]

        J[0, 0] = -px * st - py * ct
        J[1, 0] = px * ct - py * st
        # This Jacobian depends on x (it's not constant): return 0
        return 0

    cdef void _get_identity_parameters(self, double[:] theta) noexcept nogil:
        r""" Parameter values corresponding to the identity
        Sets in theta the parameter values corresponding to the identity
        transform

        Parameters
        ----------
        theta : array, shape (1,)
            buffer to write the parameters of the 2D rotation transform
        """
        theta[0] = 0

    cdef void _param_to_matrix(self, double[:] theta, double[:, :] R) noexcept nogil:
        r""" Matrix associated with the 2D rotation transform

        Parameters
        ----------
        theta : array, shape (1,)
            the rotation angle
        R : array, shape (3,3)
            the buffer in which to write the matrix
        """
        cdef:
            double ct = cos(theta[0])
            double st = sin(theta[0])
        R[0, 0], R[0, 1], R[0, 2] = ct, -st, 0
        R[1, 0], R[1, 1], R[1, 2] = st, ct, 0
        R[2, 0], R[2, 1], R[2, 2] = 0, 0, 1


cdef class RotationTransform3D(Transform):
    def __init__(self):
        r""" Rotation transform in 3D
        """
        self.dim = 3
        self.number_of_parameters = 3

    cdef int _jacobian(self, double[:] theta, double[:] x,
                       double[:, :] J) noexcept nogil:
        r""" Jacobian matrix of a 3D rotation with parameters theta, at x

        Parameters
        ----------
        theta : array, shape (3,)
            the rotation angles about the canonical axes
        x : array, shape (3,)
            the point at which to compute the Jacobian
        J : array, shape (3, 3)
            the buffer in which to write the Jacobian

        Returns
        -------
        is_constant : int
            always returns 0, indicating that the Jacobian is not
            constant (it depends on the value of x)
        """
        cdef:
            double sa = sin(theta[0])
            double ca = cos(theta[0])
            double sb = sin(theta[1])
            double cb = cos(theta[1])
            double sc = sin(theta[2])
            double cc = cos(theta[2])
            double px = x[0], py = x[1], z = x[2]

        J[0, 0] = (-sc * ca * sb) * px + (sc * sa) * py + (sc * ca * cb) * z
        J[1, 0] = (cc * ca * sb) * px + (-cc * sa) * py + (-cc * ca * cb) * z
        J[2, 0] = (sa * sb) * px + ca * py + (-sa * cb) * z

        J[0, 1] = (-cc * sb - sc * sa * cb) * px + (cc * cb - sc * sa * sb) * z
        J[1, 1] = (-sc * sb + cc * sa * cb) * px + (sc * cb + cc * sa * sb) * z
        J[2, 1] = (-ca * cb) * px + (-ca * sb) * z

        J[0, 2] = (-sc * cb - cc * sa * sb) * px + (-cc * ca) * py + \
                  (-sc * sb + cc * sa * cb) * z
        J[1, 2] = (cc * cb - sc * sa * sb) * px + (-sc * ca) * py + \
                  (cc * sb + sc * sa * cb) * z
        J[2, 2] = 0
        # This Jacobian depends on x (it's not constant): return 0
        return 0

    cdef void _get_identity_parameters(self, double[:] theta) noexcept nogil:
        r""" Parameter values corresponding to the identity
        Sets in theta the parameter values corresponding to the identity
        transform

        Parameters
        ----------
        theta : array, shape (3,)
            buffer to write the parameters of the 3D rotation transform
        """
        theta[:3] = 0

    cdef void _param_to_matrix(self, double[:] theta, double[:, :] R) noexcept nogil:
        r""" Matrix associated with the 3D rotation transform

        The matrix is the product of rotation matrices of angles theta[0],
        theta[1], theta[2] around axes x, y, z applied in the following
        order: y, x, z. This order was chosen for consistency with ANTS.

        Parameters
        ----------
        theta : array, shape (3,)
            the rotation angles about each axis:
            theta[0] : rotation angle around x axis
            theta[1] : rotation angle around y axis
            theta[2] : rotation angle around z axis
        R : array, shape (4, 4)
            buffer in which to write the rotation matrix
        """
        cdef:
            double sa = sin(theta[0])
            double ca = cos(theta[0])
            double sb = sin(theta[1])
            double cb = cos(theta[1])
            double sc = sin(theta[2])
            double cc = cos(theta[2])

        R[0,0], R[0,1], R[0,2] = cc*cb-sc*sa*sb, -sc*ca, cc*sb+sc*sa*cb
        R[1,0], R[1,1], R[1,2] = sc*cb+cc*sa*sb, cc*ca, sc*sb-cc*sa*cb
        R[2,0], R[2,1], R[2,2] = -ca*sb, sa, ca*cb
        R[3,0], R[3,1], R[3,2] = 0, 0, 0
        R[0, 3] = 0
        R[1, 3] = 0
        R[2, 3] = 0
        R[3, 3] = 1


cdef class RigidTransform2D(Transform):
    def __init__(self):
        r""" Rigid transform in 2D (rotation + translation)
        The parameter vector theta of length 3 is interpreted as follows:
        theta[0] : rotation angle
        theta[1] : translation along the x axis
        theta[2] : translation along the y axis
        """
        self.dim = 2
        self.number_of_parameters = 3

    cdef int _jacobian(self, double[:] theta, double[:] x,
                       double[:, :] J) noexcept nogil:
        r""" Jacobian matrix of a 2D rigid transform (rotation + translation)

        The transformation is given by:

        T(x,y) = (T1(x,y), T2(x,y)) =
                 (x cost - y sint + dx, x sint + y cost + dy)

        The derivatives w.r.t. t, dx and dy are:

        T'(x,y) = [-x sint - y cost, 1, 0, # derivative of T1 w.r.t. t, dx, dy
                    x cost - y sint, 0, 1] # derivative of T2 w.r.t. t, dx, dy

        Parameters
        ----------
        theta : array, shape (3,)
            the parameters of the 2D rigid transform
            theta[0] : rotation angle (t)
            theta[1] : translation along the x axis (dx)
            theta[2] : translation along the y axis (dy)
        x : array, shape (2,)
            the point at which to compute the Jacobian
        J : array, shape (2, 3)
            the buffer in which to write the Jacobian

        Returns
        -------
        is_constant : int
            always returns 0, indicating that the Jacobian is not
            constant (it depends on the value of x)
        """
        cdef:
            double st = sin(theta[0])
            double ct = cos(theta[0])
            double px = x[0], py = x[1]

        J[0, 0], J[0, 1], J[0, 2] = -px * st - py * ct, 1, 0
        J[1, 0], J[1, 1], J[1, 2] = px * ct - py * st, 0, 1
        # This Jacobian depends on x (it's not constant): return 0
        return 0

    cdef void _get_identity_parameters(self, double[:] theta) noexcept nogil:
        r""" Parameter values corresponding to the identity
        Sets in theta the parameter values corresponding to the identity
        transform

        Parameters
        ----------
        theta : array, shape (3,)
            buffer to write the parameters of the 2D rigid transform
            theta[0] : rotation angle
            theta[1] : translation along the x axis
            theta[2] : translation along the y axis
        """
        theta[:3] = 0

    cdef void _param_to_matrix(self, double[:] theta, double[:, :] R) noexcept nogil:
        r""" Matrix associated with the 2D rigid transform

        Parameters
        ----------
        theta : array, shape (3,)
            the parameters of the 2D rigid transform
            theta[0] : rotation angle
            theta[1] : translation along the x axis
            theta[2] : translation along the y axis
        R : array, shape (3, 3)
            buffer in which to write the rigid matrix
        """
        cdef:
            double ct = cos(theta[0])
            double st = sin(theta[0])
        R[0, 0], R[0, 1], R[0, 2] = ct, -st, theta[1]
        R[1, 0], R[1, 1], R[1, 2] = st, ct, theta[2]
        R[2, 0], R[2, 1], R[2, 2] = 0, 0, 1


cdef class RigidTransform3D(Transform):
    def __init__(self):
        r""" Rigid transform in 3D (rotation + translation)
        The parameter vector theta of length 6 is interpreted as follows:
        theta[0] : rotation about the x axis
        theta[1] : rotation about the y axis
        theta[2] : rotation about the z axis
        theta[3] : translation along the x axis
        theta[4] : translation along the y axis
        theta[5] : translation along the z axis
        """
        self.dim = 3
        self.number_of_parameters = 6

    cdef int _jacobian(self, double[:] theta, double[:] x,
                       double[:, :] J) noexcept nogil:
        r""" Jacobian matrix of a 3D rigid transform (rotation + translation)

        Parameters
        ----------
        theta : array, shape (6,)
            the parameters of the 3D rigid transform
            theta[0] : rotation about the x axis
            theta[1] : rotation about the y axis
            theta[2] : rotation about the z axis
            theta[3] : translation along the x axis
            theta[4] : translation along the y axis
            theta[5] : translation along the z axis
        x : array, shape (3,)
            the point at which to compute the Jacobian
        J : array, shape (3, 6)
            the buffer in which to write the Jacobian

        Returns
        -------
        is_constant : int
            always returns 0, indicating that the Jacobian is not
            constant (it depends on the value of x)
        """
        cdef:
            double sa = sin(theta[0])
            double ca = cos(theta[0])
            double sb = sin(theta[1])
            double cb = cos(theta[1])
            double sc = sin(theta[2])
            double cc = cos(theta[2])
            double px = x[0], py = x[1], pz = x[2]

        J[0, 0] = (-sc * ca * sb) * px + (sc * sa) * py + (sc * ca * cb) * pz
        J[1, 0] = (cc * ca * sb) * px + (-cc * sa) * py + (-cc * ca * cb) * pz
        J[2, 0] = (sa * sb) * px + ca * py + (-sa * cb) * pz

        J[0, 1] = (-cc * sb - sc * sa * cb) * px + (cc * cb - sc * sa * sb) * pz
        J[1, 1] = (-sc * sb + cc * sa * cb) * px + (sc * cb + cc * sa * sb) * pz
        J[2, 1] = (-ca * cb) * px + (-ca * sb) * pz

        J[0, 2] = (-sc * cb - cc * sa * sb) * px + (-cc * ca) * py + \
                  (-sc * sb + cc * sa * cb) * pz
        J[1, 2] = (cc * cb - sc * sa * sb) * px + (-sc * ca) * py + \
                  (cc * sb + sc * sa * cb) * pz
        J[2, 2] = 0

        J[0, 3:6] = 0
        J[1, 3:6] = 0
        J[2, 3:6] = 0
        J[0, 3], J[1, 4], J[2, 5] = 1, 1, 1
        # This Jacobian depends on x (it's not constant): return 0
        return 0

    cdef void _get_identity_parameters(self, double[:] theta) noexcept nogil:
        r""" Parameter values corresponding to the identity
        Sets in theta the parameter values corresponding to the identity
        transform

        Parameters
        ----------
        theta : array, shape (6,)
            buffer to write the parameters of the 3D rigid transform
            theta[0] : rotation about the x axis
            theta[1] : rotation about the y axis
            theta[2] : rotation about the z axis
            theta[3] : translation along the x axis
            theta[4] : translation along the y axis
            theta[5] : translation along the z axis
        """
        theta[:6] = 0

    cdef void _param_to_matrix(self, double[:] theta, double[:, :] R) noexcept nogil:
        r""" Matrix associated with the 3D rigid transform

        Parameters
        ----------
        theta : array, shape (6,)
            the parameters of the 3D rigid transform
            theta[0] : rotation about the x axis
            theta[1] : rotation about the y axis
            theta[2] : rotation about the z axis
            theta[3] : translation along the x axis
            theta[4] : translation along the y axis
            theta[5] : translation along the z axis
        R : array, shape (4, 4)
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

        R[0,0], R[0,1], R[0,2] = cc*cb-sc*sa*sb, -sc*ca, cc*sb+sc*sa*cb
        R[1,0], R[1,1], R[1,2] = sc*cb+cc*sa*sb, cc*ca, sc*sb-cc*sa*cb
        R[2,0], R[2,1], R[2,2] = -ca*sb, sa, ca*cb
        R[3,0], R[3,1], R[3,2] = 0, 0, 0
        R[0,3] = dx
        R[1,3] = dy
        R[2,3] = dz
        R[3,3] = 1


cdef class RigidIsoScalingTransform2D(Transform):
    def __init__(self):
        """ Rigid isoscaling transform in 2D.

        (rotation + translation + scaling)

        The parameter vector theta of length 4 is interpreted as follows:
        theta[0] : rotation angle
        theta[1] : translation along the x axis
        theta[2] : translation along the y axis
        theta[3] : isotropic scaling
        """
        self.dim = 2
        self.number_of_parameters = 4

    cdef int _jacobian(self, double[:] theta, double[:] x,
                       double[:, :] J) noexcept nogil:
        r""" Jacobian matrix of a 2D rigid isoscaling transform.

        The transformation (rotation + translation + isoscaling) is given by:

        T(x,y) = (T1(x,y), T2(x,y)) =
            (sx * x * cost -  sx * y * sint + dx, sy * x * sint + sy * y * cost + dy)

        Parameters
        ----------
        theta : array, shape (4,)
            the parameters of the 2D rigid isoscaling transform
            theta[0] : rotation angle (t)
            theta[1] : translation along the x axis (dx)
            theta[2] : translation along the y axis (dy)
            theta[3] : isotropic scaling (s)
        x : array, shape (2,)
            the point at which to compute the Jacobian
        J : array, shape (2, 4)
            the buffer in which to write the Jacobian

        Returns
        -------
        is_constant : int
            always returns 0, indicating that the Jacobian is not
            constant (it depends on the value of x)
        """
        cdef:
            double st = sin(theta[0])
            double ct = cos(theta[0])
            double px = x[0], py = x[1]
            double scale = theta[3]

        J[0, 0], J[0, 1], J[0, 2] = -px * scale * st - py * scale * ct, 1, 0
        J[1, 0], J[1, 1], J[1, 2] = px * scale * ct - py * scale * st, 0, 1

        J[0, 3] = px * ct - py * st
        J[1, 3] = px * st + py * ct
        # This Jacobian depends on x (it's not constant): return 0
        return 0

    cdef void _get_identity_parameters(self, double[:] theta) noexcept nogil:
        """ Parameter values corresponding to the identity
        Sets in theta the parameter values corresponding to the identity
        transform

        Parameters
        ----------
        theta : array, shape (4,)
            buffer to write the parameters of the 2D isoscaling rigid transform
            theta[0] : rotation angle
            theta[1] : translation along the x axis
            theta[2] : translation along the y axis
            theta[3] : isotropic scaling
        """
        theta[:3] = 0
        theta[3] = 1

    cdef void _param_to_matrix(self, double[:] theta, double[:, :] R) noexcept nogil:
        r""" Matrix associated with the 2D rigid isoscaling transform

        Parameters
        ----------
        theta : array, shape (4,)
            the parameters of the 2D rigid isoscaling transform
            theta[0] : rotation angle
            theta[1] : translation along the x axis
            theta[2] : translation along the y axis
            theta[3] : isotropic scaling
        R : array, shape (3, 3)
            buffer in which to write the rigid isoscaling matrix
        """
        cdef:
            double ct = cos(theta[0])
            double st = sin(theta[0])
        R[0, 0], R[0, 1], R[0, 2] = ct*theta[3], -st*theta[3], theta[1]
        R[1, 0], R[1, 1], R[1, 2] = st*theta[3], ct*theta[3], theta[2]
        R[2, 0], R[2, 1], R[2, 2] = 0, 0, 1


cdef class RigidIsoScalingTransform3D(Transform):
    def __init__(self):
        """Rigid isoscaling transform in 3D.

        (rotation + translation + scaling)
        The parameter vector theta of length 7 is interpreted as follows:
        theta[0] : rotation about the x axis
        theta[1] : rotation about the y axis
        theta[2] : rotation about the z axis
        theta[3] : translation along the x axis
        theta[4] : translation along the y axis
        theta[5] : translation along the z axis
        theta[6] : isotropic scaling

        """
        self.dim = 3
        self.number_of_parameters = 7

    cdef int _jacobian(self, double[:] theta, double[:] x,
                       double[:, :] J) noexcept nogil:
        r""" Jacobian matrix of a 3D isoscaling rigid transform.

        (rotation + translation + scaling)

        Parameters
        ----------
        theta : array, shape (7,)
            the parameters of the 3D rigid isoscaling transform
            theta[0] : rotation about the x axis
            theta[1] : rotation about the y axis
            theta[2] : rotation about the z axis
            theta[3] : translation along the x axis
            theta[4] : translation along the y axis
            theta[5] : translation along the z axis
            theta[6] : isotropic scaling
        x : array, shape (3,)
            the point at which to compute the Jacobian
        J : array, shape (3, 7)
            the buffer in which to write the Jacobian

        Returns
        -------
        is_constant : int
            always returns 0, indicating that the Jacobian is not
            constant (it depends on the value of x)
        """
        cdef:
            double sa = sin(theta[0])
            double ca = cos(theta[0])
            double sb = sin(theta[1])
            double cb = cos(theta[1])
            double sc = sin(theta[2])
            double cc = cos(theta[2])
            double px = x[0], py = x[1], pz = x[2]
            double scale = theta[6]

        J[0, 0] = (-sc * ca * sb) * px * scale + (sc * sa) * py * scale + \
                  (sc * ca * cb) * pz * scale
        J[1, 0] = (cc * ca * sb) * px * scale + (-cc * sa) * py * scale + \
                  (-cc * ca * cb) * pz  * scale
        J[2, 0] = (sa * sb) * px * scale + ca * py * scale + \
                  (-sa * cb) * pz * scale

        J[0, 1] = (-cc * sb - sc * sa * cb) * px * scale + \
                  (cc * cb - sc * sa * sb) * pz * scale
        J[1, 1] = (-sc * sb + cc * sa * cb) * px * scale + \
                  (sc * cb + cc * sa * sb) * pz * scale
        J[2, 1] = (-ca * cb) * px * scale + (-ca * sb) * pz * scale

        J[0, 2] = (-sc * cb - cc * sa * sb) * px * scale + \
                  (-cc * ca) * py  * scale + \
                  (-sc * sb + cc * sa * cb) * pz * scale
        J[1, 2] = (cc * cb - sc * sa * sb) * px * scale + \
                  (-sc * ca) * py * scale + \
                  (cc * sb + sc * sa * cb) * pz * scale
        J[2, 2] = 0

        J[0, 3:6] = 0
        J[1, 3:6] = 0
        J[2, 3:6] = 0
        J[0, 3], J[1, 4], J[2, 5] = 1, 1, 1
        J[0, 6] = (cc*cb-sc*sa*sb) * px - sc*ca*py + (cc*sb+sc*sa*cb) * pz
        J[1, 6] = (sc*cb+cc*sa*sb) * px + cc*ca*py + (sc*sb-cc*sa*cb) * pz
        J[2, 6] = -ca*sb*px + sa*py + ca*cb*pz
        # This Jacobian depends on x (it's not constant): return 0
        return 0

    cdef void _get_identity_parameters(self, double[:] theta) noexcept nogil:
        """ Parameter values corresponding to the identity

        Sets in theta the parameter values corresponding to the identity
        transform

        Parameters
        ----------
        theta : array, shape (7,)
            buffer to write the parameters of the 3D rigid isoscaling transform
            theta[0] : rotation about the x axis
            theta[1] : rotation about the y axis
            theta[2] : rotation about the z axis
            theta[3] : translation along the x axis
            theta[4] : translation along the y axis
            theta[5] : translation along the z axis
            theta[6] : isotropic scaling

        """
        theta[:6] = 0
        theta[6] = 1

    cdef void _param_to_matrix(self, double[:] theta, double[:, :] R) noexcept nogil:
        """ Matrix associated with the 3D rigid isoscaling transform

        Parameters
        ----------
        theta : array, shape (7,)
            the parameters of the 3D rigid isoscaling transform
            theta[0] : rotation about the x axis
            theta[1] : rotation about the y axis
            theta[2] : rotation about the z axis
            theta[3] : translation along the x axis
            theta[4] : translation along the y axis
            theta[5] : translation along the z axis
            theta[6] : isotropic scaling
        R : array, shape (4, 4)
            buffer in which to write the rigid isoscaling matrix
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
            double sxyz = theta[6]

        R[0,0], R[0,1], R[0,2] = (cc*cb-sc*sa*sb)*sxyz, -sc*ca*sxyz, (cc*sb+sc*sa*cb)*sxyz
        R[1,0], R[1,1], R[1,2] = (sc*cb+cc*sa*sb)*sxyz, cc*ca*sxyz, (sc*sb-cc*sa*cb)*sxyz
        R[2,0], R[2,1], R[2,2] = -ca*sb*sxyz, sa*sxyz, ca*cb*sxyz
        R[3,0], R[3,1], R[3,2] = 0, 0, 0
        R[0,3] = dx
        R[1,3] = dy
        R[2,3] = dz
        R[3,3] = 1


cdef class RigidScalingTransform2D(Transform):
    def __init__(self):
        """ Rigid scaling transform in 2D.

        (rotation + translation + scaling)

        The parameter vector theta of length 5 is interpreted as follows:
        theta[0] : rotation angle
        theta[1] : translation along the x axis
        theta[2] : translation along the y axis
        theta[3] : scaling along the x axis
        theta[4] : scaling along the y axis
        """
        self.dim = 2
        self.number_of_parameters = 5

    cdef int _jacobian(self, double[:] theta, double[:] x,
                       double[:, :] J) noexcept nogil:
        r""" Jacobian matrix of a 2D rigid scaling transform.

        The transformation (rotation + translation + scaling) is given by:

        T(x,y) = (T1(x,y), T2(x,y)) =
            (sx * x * cost -  sx * y * sint + dx, sy * x * sint + sy * y * cost + dy)

        Parameters
        ----------
        theta : array, shape (5,)
            the parameters of the 2D rigid isoscaling transform
            theta[0] : rotation angle (t)
            theta[1] : translation along the x axis (dx)
            theta[2] : translation along the y axis (dy)
            theta[3] : scaling along the x axis
            theta[4] : scaling along the y axis
        x : array, shape (2,)
            the point at which to compute the Jacobian
        J : array, shape (2, 5)
            the buffer in which to write the Jacobian

        Returns
        -------
        is_constant : int
            always returns 0, indicating that the Jacobian is not
            constant (it depends on the value of x)
        """
        cdef:
            double st = sin(theta[0])
            double ct = cos(theta[0])
            double px = x[0], py = x[1]
            double fx = theta[3]
            double fy = theta[4]

        J[0, 0], J[0, 1], J[0, 2] = -px * fx * st - py * fx * ct, 1, 0
        J[1, 0], J[1, 1], J[1, 2] = px * fy * ct - py * fy * st, 0, 1

        J[0, 3], J[0, 4] = px * ct - py * st, 0
        J[1, 3], J[1, 4] = 0, px * st + py * ct
        # This Jacobian depends on x (it's not constant): return 0
        return 0

    cdef void _get_identity_parameters(self, double[:] theta) noexcept nogil:
        """ Parameter values corresponding to the identity
        Sets in theta the parameter values corresponding to the identity
        transform

        Parameters
        ----------
        theta : array, shape (5,)
            buffer to write the parameters of the 2D scaling rigid transform
            theta[0] : rotation angle
            theta[1] : translation along the x axis
            theta[2] : translation along the y axis
            theta[3] : scaling along the x axis
            theta[4] : scaling along the y axis
        """
        theta[:3] = 0
        theta[3], theta[4] = 1, 1

    cdef void _param_to_matrix(self, double[:] theta, double[:, :] R) noexcept nogil:
        r""" Matrix associated with the 2D rigid scaling transform

        Parameters
        ----------
        theta : array, shape (5,)
            the parameters of the 2D rigid scaling transform
            theta[0] : rotation angle
            theta[1] : translation along the x axis
            theta[2] : translation along the y axis
            theta[3] : scaling along the x axis
            theta[4] : scaling along the y axis
        R : array, shape (3, 3)
            buffer in which to write the rigid scaling matrix
        """
        cdef:
            double ct = cos(theta[0])
            double st = sin(theta[0])
        R[0, 0], R[0, 1], R[0, 2] = ct*theta[3], -st*theta[3], theta[1]
        R[1, 0], R[1, 1], R[1, 2] = st*theta[4], ct*theta[4], theta[2]
        R[2, 0], R[2, 1], R[2, 2] = 0, 0, 1


cdef class RigidScalingTransform3D(Transform):
    def __init__(self):
        """Rigid Scaling transform in 3D (rotation + translation + scaling).

        The parameter vector theta of length 9 is interpreted as follows:
        theta[0] : rotation about the x axis
        theta[1] : rotation about the y axis
        theta[2] : rotation about the z axis
        theta[3] : translation along the x axis
        theta[4] : translation along the y axis
        theta[5] : translation along the z axis
        theta[6] : scaling in the x axis
        theta[7] : scaling in the y axis
        theta[8] : scaling in the z axis

        """
        self.dim = 3
        self.number_of_parameters = 9

    cdef int _jacobian(self, double[:] theta, double[:] x,
                       double[:, :] J) noexcept nogil:
        """Jacobian matrix of a 3D rigid scaling transform.

        (rotation + translation + scaling)

        Parameters
        ----------
        theta : array, shape (9,)
            the parameters of the 3D rigid transform
            theta[0] : rotation about the x axis
            theta[1] : rotation about the y axis
            theta[2] : rotation about the z axis
            theta[3] : translation along the x axis
            theta[4] : translation along the y axis
            theta[5] : translation along the z axis
            theta[6] : scaling in the x axis
            theta[7] : scaling in the y axis
            theta[8] : scaling in the z axis
        x : array, shape (3,)
            the point at which to compute the Jacobian
        J : array, shape (3, 9)
            the buffer in which to write the Jacobian

        Returns
        -------
        is_constant : int
            always returns 0, indicating that the Jacobian is not
            constant (it depends on the value of x)
        """
        cdef:
            double sa = sin(theta[0])
            double ca = cos(theta[0])
            double sb = sin(theta[1])
            double cb = cos(theta[1])
            double sc = sin(theta[2])
            double cc = cos(theta[2])
            double px = x[0], py = x[1], pz = x[2]
            double fx = theta[6]
            double fy = theta[7]
            double fz = theta[8]

        J[0, 0] = (-sc * ca * sb) * px * fx + (sc * sa) * py * fx + \
                  (sc * ca * cb) * pz * fx
        J[1, 0] = (cc * ca * sb) * px * fy + (-cc * sa) * py * fy + \
                  (-cc * ca * cb) * pz * fy
        J[2, 0] = (sa * sb) * px * fz + ca * py * fz + (-sa * cb) * pz * fz

        J[0, 1] = (-cc * sb - sc * sa * cb) * px * fx + \
                  (cc * cb - sc * sa * sb) * pz * fx
        J[1, 1] = (-sc * sb + cc * sa * cb) * px * fy + \
                  (sc * cb + cc * sa * sb) * pz * fy
        J[2, 1] = (-ca * cb) * px * fz + (-ca * sb) * pz * fz

        J[0, 2] = (-sc * cb - cc * sa * sb) * px * fx + (-cc * ca) * py * fx + \
                  (-sc * sb + cc * sa * cb) * pz * fx
        J[1, 2] = (cc * cb - sc * sa * sb) * px * fy + (-sc * ca) * py * fy + \
                  (cc * sb + sc * sa * cb) * pz * fy
        J[2, 2] = 0

        J[0, 3:6] = 0
        J[1, 3:6] = 0
        J[2, 3:6] = 0
        J[0, 3], J[1, 4], J[2, 5] = 1, 1, 1
        J[0, 6] = (cc*cb-sc*sa*sb) * px - sc*ca*py + (cc*sb+sc*sa*cb) * pz
        J[1, 6], J[2, 6] = 0, 0
        J[1, 7] = (sc*cb+cc*sa*sb) * px + cc*ca*py + (sc*sb-cc*sa*cb) * pz
        J[0, 7], J[2, 7] = 0, 0
        J[0, 8], J[1, 8] = 0, 0
        J[2, 8] = -ca*sb*px + sa*py + ca*cb*pz
        # This Jacobian depends on x (it's not constant): return 0
        return 0

    cdef void _get_identity_parameters(self, double[:] theta) noexcept nogil:
        r""" Parameter values corresponding to the identity
        Sets in theta the parameter values corresponding to the identity
        transform

        Parameters
        ----------
        theta : array, shape (9,)
            buffer to write the parameters of the 3D rigid scaling transform
            theta[0] : rotation about the x axis
            theta[1] : rotation about the y axis
            theta[2] : rotation about the z axis
            theta[3] : translation along the x axis
            theta[4] : translation along the y axis
            theta[5] : translation along the z axis
            theta[6] : scaling in the x axis
            theta[7] : scaling in the y axis
            theta[8] : scaling in the z axis
        """
        theta[:6] = 0
        theta[6:9] = 1

    cdef void _param_to_matrix(self, double[:] theta, double[:, :] R) noexcept nogil:
        r""" Matrix associated with the 3D rigid scaling transform

        Parameters
        ----------
        theta : array, shape (9,)
            the parameters of the 3D rigid scaling transform
            theta[0] : rotation about the x axis
            theta[1] : rotation about the y axis
            theta[2] : rotation about the z axis
            theta[3] : translation along the x axis
            theta[4] : translation along the y axis
            theta[5] : translation along the z axis
            theta[6] : scaling in the x axis
            theta[7] : scaling in the y axis
            theta[8] : scaling in the z axis
        R : array, shape (4, 4)
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
            double fx = theta[6]
            double fy = theta[7]
            double fz = theta[8]

        R[0,0], R[0,1], R[0,2] = (cc*cb-sc*sa*sb)*fx, -sc*ca*fx, (cc*sb+sc*sa*cb)*fx
        R[1,0], R[1,1], R[1,2] = (sc*cb+cc*sa*sb)*fy, cc*ca*fy, (sc*sb-cc*sa*cb)*fy
        R[2,0], R[2,1], R[2,2] = -ca*sb*fz, sa*fz, ca*cb*fz
        R[3,0], R[3,1], R[3,2] = 0, 0, 0
        R[0,3] = dx
        R[1,3] = dy
        R[2,3] = dz
        R[3,3] = 1


cdef class ScalingTransform2D(Transform):
    def __init__(self):
        r""" Scaling transform in 2D
        """
        self.dim = 2
        self.number_of_parameters = 1

    cdef int _jacobian(self, double[:] theta, double[:] x,
                       double[:, :] J) noexcept nogil:
        r""" Jacobian matrix of the isotropic 2D scale transform
        The transformation is given by:

        T(x) = (s*x0, s*x1)

        The derivative w.r.t. s is T'(x) = [x0, x1]

        Parameters
        ----------
        theta : array, shape (1,)
            the scale factor (the Jacobian does not depend on the scale factor,
            but we receive the buffer to make it consistent with other Jacobian
            functions)
        x : array, shape (2,)
            the point at which to compute the Jacobian
        J : array, shape (2, 1)
            the buffer in which to write the Jacobian

        Returns
        -------
        is_constant : int
            always returns 0, indicating that the Jacobian is not
            constant (it depends on the value of x)
        """
        J[0, 0], J[1, 0] = x[0], x[1]
        # This Jacobian depends on x (it's not constant): return 0
        return 0

    cdef void _get_identity_parameters(self, double[:] theta) noexcept nogil:
        r""" Parameter values corresponding to the identity
        Sets in theta the parameter values corresponding to the identity
        transform

        Parameters
        ----------
        theta : array, shape (1,)
            buffer to write the parameters of the 2D scale transform
        """
        theta[0] = 1

    cdef void _param_to_matrix(self, double[:] theta, double[:, :] R) noexcept nogil:
        r""" Matrix associated with the 2D (isotropic) scaling transform

        Parameters
        ----------
        theta : array, shape (1,)
            the scale factor
        R : array, shape (3, 3)
            the buffer in which to write the scaling matrix
        """
        R[0, 0], R[0, 1], R[0, 2] = theta[0], 0, 0
        R[1, 0], R[1, 1], R[1, 2] = 0, theta[0], 0
        R[2, 0], R[2, 1], R[2, 2] = 0, 0, 1


cdef class ScalingTransform3D(Transform):
    def __init__(self):
        r""" Scaling transform in 3D
        """
        self.dim = 3
        self.number_of_parameters = 1

    cdef int _jacobian(self, double[:] theta, double[:] x,
                       double[:, :] J) noexcept nogil:
        r""" Jacobian matrix of the isotropic 3D scale transform
        The transformation is given by:

        T(x) = (s*x0, s*x1, s*x2)

        The derivative w.r.t. s is T'(x) = [x0, x1, x2]

        Parameters
        ----------
        theta : array, shape (1,)
            the scale factor (the Jacobian does not depend on the scale factor,
            but we receive the buffer to make it consistent with other Jacobian
            functions)
        x : array, shape (3,)
            the point at which to compute the Jacobian
        J : array, shape (3, 1)
            the buffer in which to write the Jacobian

        Returns
        -------
        is_constant : int
            always returns 0, indicating that the Jacobian is not
            constant (it depends on the value of x)
        """
        J[0, 0], J[1, 0], J[2, 0]= x[0], x[1], x[2]
        # This Jacobian depends on x (it's not constant): return 0
        return 0

    cdef void _get_identity_parameters(self, double[:] theta) noexcept nogil:
        r""" Parameter values corresponding to the identity
        Sets in theta the parameter values corresponding to the identity
        transform

        Parameters
        ----------
        theta : array, shape (1,)
            buffer to write the parameters of the 3D scale transform
        """
        theta[0] = 1

    cdef void _param_to_matrix(self, double[:] theta, double[:, :] R) noexcept nogil:
        r""" Matrix associated with the 3D (isotropic) scaling transform

        Parameters
        ----------
        theta : array, shape (1,)
            the scale factor
        R : array, shape (4, 4)
            the buffer in which to write the scaling matrix
        """
        R[0, 0], R[0, 1], R[0, 2], R[0, 3] = theta[0], 0, 0, 0
        R[1, 0], R[1, 1], R[1, 2], R[1, 3] = 0, theta[0], 0, 0
        R[2, 0], R[2, 1], R[2, 2], R[2, 3] = 0, 0, theta[0], 0
        R[3, 0], R[3, 1], R[3, 2], R[3, 3] = 0, 0, 0, 1


cdef class AffineTransform2D(Transform):
    def __init__(self):
        r""" Affine transform in 2D
        """
        self.dim = 2
        self.number_of_parameters = 6

    cdef int _jacobian(self, double[:] theta, double[:] x,
                       double[:, :] J) noexcept nogil:
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
        theta : array, shape (6,)
            the parameters of the 2D affine transform
        x : array, shape (2,)
            the point at which to compute the Jacobian
        J : array, shape (2, 6)
            the buffer in which to write the Jacobian

        Returns
        -------
        is_constant : int
            always returns 0, indicating that the Jacobian is not
            constant (it depends on the value of x)
        """
        J[0, :6] = 0
        J[1, :6] = 0

        J[0, :2] = x[:2]
        J[0, 2] = 1
        J[1, 3:5] = x[:2]
        J[1, 5] = 1
        # This Jacobian depends on x (it's not constant): return 0
        return 0

    cdef void _get_identity_parameters(self, double[:] theta) noexcept nogil:
        r""" Parameter values corresponding to the identity
        Sets in theta the parameter values corresponding to the identity
        transform

        Parameters
        ----------
        theta : array, shape (6,)
            buffer to write the parameters of the 2D affine transform
        """
        theta[0], theta[1], theta[2] = 1, 0, 0
        theta[3], theta[4], theta[5] = 0, 1, 0

    cdef void _param_to_matrix(self, double[:] theta, double[:, :] R) noexcept nogil:
        r""" Matrix associated with a general 2D affine transform

        The transformation is given by the matrix:

        A = [[a0, a1, a2],
             [a3, a4, a5],
             [ 0,  0,  1]]

        Parameters
        ----------
        theta : array, shape (6,)
            the parameters of the 2D affine transform
        R : array, shape (3,3)
            the buffer in which to write the matrix
        """
        R[0, 0], R[0, 1], R[0, 2] = theta[0], theta[1], theta[2]
        R[1, 0], R[1, 1], R[1, 2] = theta[3], theta[4], theta[5]
        R[2, 0], R[2, 1], R[2, 2] = 0, 0, 1


cdef class AffineTransform3D(Transform):
    def __init__(self):
        r""" Affine transform in 3D
        """
        self.dim = 3
        self.number_of_parameters = 12

    cdef int _jacobian(self, double[:] theta, double[:] x,
                       double[:, :] J) noexcept nogil:
        r""" Jacobian matrix of the 3D affine transform
        The transformation is given by:

        T(x)= |a0, a1, a2,  a3 |  |x0|  | T1(x) |  |a0*x0 + a1*x1 + a2*x2 + a3|
              |a4, a5, a6,  a7 |* |x1|= | T2(x) |= |a4*x0 + a5*x1 + a6*x2 + a7|
              |a8, a9, a10, a11|  |x2|  | T3(x) |  |a8*x0 + a9*x1 + a10*x2+a11|
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
        theta : array, shape (12,)
            the parameters of the 3D affine transform
        x : array, shape (3,)
            the point at which to compute the Jacobian
        J : array, shape (3, 12)
            the buffer in which to write the Jacobian

        Returns
        -------
        is_constant : int
            always returns 0, indicating that the Jacobian is not
            constant (it depends on the value of x)
        """
        cdef:
            cnp.npy_intp j

        for j in range(3):
            J[j, :12] = 0
        J[0, :3] = x[:3]
        J[0, 3] = 1
        J[1, 4:7] = x[:3]
        J[1, 7] = 1
        J[2, 8:11] = x[:3]
        J[2, 11] = 1
        # This Jacobian depends on x (it's not constant): return 0
        return 0

    cdef void _get_identity_parameters(self, double[:] theta) noexcept nogil:
        r""" Parameter values corresponding to the identity
        Sets in theta the parameter values corresponding to the identity
        transform

        Parameters
        ----------
        theta : array, shape (12,)
            buffer to write the parameters of the 3D affine transform
        """
        theta[0], theta[1], theta[2], theta[3] = 1, 0, 0, 0
        theta[4], theta[5], theta[6], theta[7] = 0, 1, 0, 0
        theta[8], theta[9], theta[10], theta[11] = 0, 0, 1, 0

    cdef void _param_to_matrix(self, double[:] theta, double[:, :] R) noexcept nogil:
        r""" Matrix associated with a general 3D affine transform

        The transformation is given by the matrix:

        A = [[a0, a1, a2, a3],
             [a4, a5, a6, a7],
             [a8, a9, a10, a11],
             [ 0,   0,   0,   1]]

        Parameters
        ----------
        theta : array, shape (12,)
            the parameters of the 3D affine transform
        R : array, shape (4,4)
            the buffer in which to write the matrix
        """
        R[0, 0], R[0, 1], R[0, 2] = theta[0], theta[1], theta[2]
        R[1, 0], R[1, 1], R[1, 2] = theta[4], theta[5], theta[6]
        R[2, 0], R[2, 1], R[2, 2] = theta[8], theta[9], theta[10]
        R[3, 0], R[3, 1], R[3, 2] = 0, 0, 0
        R[0, 3] = theta[3]
        R[1, 3] = theta[7]
        R[2, 3] = theta[11]
        R[3, 3] = 1


regtransforms = dict()
regtransforms [('TRANSLATION', 2)] = TranslationTransform2D()
regtransforms [('TRANSLATION', 3)] = TranslationTransform3D()
regtransforms [('ROTATION', 2)] = RotationTransform2D()
regtransforms [('ROTATION', 3)] = RotationTransform3D()
regtransforms [('RIGID', 2)] = RigidTransform2D()
regtransforms [('RIGID', 3)] = RigidTransform3D()
regtransforms [('SCALING', 2)] = ScalingTransform2D()
regtransforms [('SCALING', 3)] = ScalingTransform3D()
regtransforms [('AFFINE', 2)] = AffineTransform2D()
regtransforms [('AFFINE', 3)] = AffineTransform3D()
regtransforms [('RIGIDSCALING', 2)] = RigidScalingTransform2D()
regtransforms [('RIGIDSCALING', 3)] = RigidScalingTransform3D()
regtransforms [('RIGIDISOSCALING', 2)] = RigidIsoScalingTransform2D()
regtransforms [('RIGIDISOSCALING', 3)] = RigidIsoScalingTransform3D()
