cdef class Transform:
    cdef:
        int number_of_parameters
        int dim
    cdef int _jacobian(self, double[:] theta, double[:] x, double[:, :] J) noexcept nogil
    cdef void _get_identity_parameters(self, double[:] theta) noexcept nogil
    cdef void _param_to_matrix(self, double[:] theta, double[:, :] T) noexcept nogil
