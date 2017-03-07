# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

ctypedef float[:, :] float2d
ctypedef double[:, :] double2d

ctypedef fused Streamline:
    float2d
    double2d


cdef double c_length(Streamline streamline) nogil

cdef void c_arclengths(Streamline streamline, double * out) nogil

cdef void c_set_number_of_points(Streamline streamline, Streamline out) nogil
