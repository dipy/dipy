# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

cdef int where_to_insert(
        np.float_t* arr,
        np.float_t number,
        int size) nogil:
    cdef:
        int idx
        np.float_t current
    for idx in range(size - 1, -1, -1):
        current = arr[idx]
        if number >= current:
            return idx + 1

    return 0


cdef void cumsum(
        np.float_t* arr_in,
        np.float_t* arr_out,
        int N) nogil:
    cdef:
        int i = 0
        np.float_t sum = 0
    for i in range(N):
        sum += arr_in[i]
        arr_out[i] = sum


cdef void copy_point(
        double * a,
        double * b) nogil:
    cdef:
        int i = 0
    for i in range(3):
        b[i] = a[i]


cdef void scalar_muliplication_point(
        double * a,
        double scalar) nogil:
    cdef:
        int i = 0
    for i in range(3):
        a[i] *= scalar
