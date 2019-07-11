
cimport numpy as cnp
from dipy.align.fused_types cimport floating, number

cpdef trilinear_interpolate4d(
    double[:, :, :, :] data,
    double[:] point,
    cnp.ndarray out=*)

cdef int trilinear_interpolate4d_c(
    double[:, :, :, :] data,
    double* point,
    double[:] result) nogil

cdef int _interpolate_vector_2d(floating[:, :, :] field, double dii,
                                double djj, floating *out) nogil
cdef int _interpolate_scalar_2d(floating[:, :] image, double dii,
                                double djj, floating *out) nogil
cdef int _interpolate_scalar_nn_2d(number[:, :] image, double dii,
                                   double djj, number *out) nogil
cdef int _interpolate_scalar_nn_3d(number[:, :, :] volume, double dkk,
                                   double dii, double djj,
                                   number *out) nogil
cdef int _interpolate_scalar_3d(floating[:, :, :] volume,
                                double dkk, double dii, double djj,
                                floating *out) nogil
cdef int _interpolate_vector_3d(floating[:, :, :, :] field, double dkk,
                                double dii, double djj,
                                floating* out) nogil
cdef void _trilinear_interpolation_iso(double *X,
                                       double *W,
                                       cnp.npy_intp *IN) nogil
cdef cnp.npy_intp offset(cnp.npy_intp *indices,
                         cnp.npy_intp *strides,
                         int lenind,
                         int typesize) nogil
