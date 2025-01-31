# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False


from dipy.align.fused_types cimport floating


cdef floating f_array_min(floating* arr, int n) noexcept nogil:
    """Return the minimum value of an array."""
    cdef int i
    cdef double min_val = arr[0]
    for i in range(1, n):
        if arr[i] < min_val:
            min_val = arr[i]
    return min_val


cdef floating f_array_max(floating* arr, int n) noexcept nogil:
    """Compute the maximum value of an array."""
    cdef int i
    cdef double max_val = arr[0]
    for i in range(1, n):
        if arr[i] > max_val:
            max_val = arr[i]
    return max_val