# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False


from dipy.align.fused_types cimport floating


cdef inline floating f_max(floating a, floating b) noexcept nogil:
    """Return the maximum of a and b."""
    return a if a >= b else b


cdef inline floating f_min(floating a, floating b) noexcept nogil:
    """Return the minimum of a and b."""
    return a if a <= b else b


cdef floating f_array_min(floating* arr, int n) noexcept nogil
cdef floating f_array_max(floating* arr, int n) noexcept nogil
