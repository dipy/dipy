#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

cdef inline double _apply_affine_3d_x0(double x0, double x1, double x2,
                                       double h, double[:, :] aff) nogil:
    r"""Multiplies aff by (x0, x1, x2, h), returns the 1st element of product

    Returns the first component of the product of the homogeneous matrix aff by
    (x0, x1, x2, h)
    """
    return aff[0, 0] * x0 + aff[0, 1] * x1 + aff[0, 2] * x2 + h*aff[0, 3]


cdef inline double _apply_affine_3d_x1(double x0, double x1, double x2,
                                       double h, double[:, :] aff) nogil:
    r"""Multiplies aff by (x0, x1, x2, h), returns the 2nd element of product

    Returns the first component of the product of the homogeneous matrix aff by
    (x0, x1, x2, h)
    """
    return aff[1, 0] * x0 + aff[1, 1] * x1 + aff[1, 2] * x2 + h*aff[1, 3]


cdef inline double _apply_affine_3d_x2(double x0, double x1, double x2,
                                       double h, double[:, :] aff) nogil:
    r"""Multiplies aff by (x0, x1, x2, h), returns the 3d element of product

    Returns the first component of the product of the homogeneous matrix aff by
    (x0, x1, x2, h)
    """
    return aff[2, 0] * x0 + aff[2, 1] * x1 + aff[2, 2] * x2 + h*aff[2, 3]


cdef inline double _apply_affine_2d_x0(double x0, double x1, double h,
                                       double[:, :] aff) nogil:
    r"""Multiplies aff by (x0, x1, h), returns the 1st element of product
    Returns the first component of the product of the homogeneous matrix aff by
    (x0, x1, h)
    """
    return aff[0, 0] * x0 + aff[0, 1] * x1 + h*aff[0, 2]


cdef inline double _apply_affine_2d_x1(double x0, double x1, double h,
                                       double[:, :] aff) nogil:
    r"""Multiplies aff by (x0, x1, h), returns the 2nd element of product

    Returns the first component of the product of the homogeneous matrix aff by
    (x0, x1, h)
    """
    return aff[1, 0] * x0 + aff[1, 1] * x1 + h*aff[1, 2]
