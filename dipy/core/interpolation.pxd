
cimport numpy as cnp
from dipy.align.fused_types cimport floating, number


cdef int trilinear_interpolate4d_c(double[:, :, :, :] data,
                                   double* point,
                                   double* result) noexcept nogil

cdef int _interpolate_vector_2d(floating[:, :, :] field, double dii,
                                double djj, floating* out) noexcept nogil
cdef int _interpolate_scalar_2d(floating[:, :] image, double dii,
                                double djj, floating* out) noexcept nogil
cdef int _interpolate_scalar_nn_2d(number[:, :] image, double dii,
                                   double djj, number *out) noexcept nogil
cdef int _interpolate_scalar_nn_3d(number[:, :, :] volume, double dkk,
                                   double dii, double djj,
                                   number* out) noexcept nogil
cdef int _interpolate_scalar_3d(floating[:, :, :] volume,
                                double dkk, double dii, double djj,
                                floating* out) noexcept nogil
cdef int _interpolate_vector_3d(floating[:, :, :, :] field, double dkk,
                                double dii, double djj,
                                floating* out) noexcept nogil
cdef void _trilinear_interpolation_iso(double* X,
                                       double* W,
                                       cnp.npy_intp* IN) noexcept nogil
cdef cnp.npy_intp offset(cnp.npy_intp* indices,
                         cnp.npy_intp* strides,
                         int lenind,
                         int typesize) noexcept nogil
