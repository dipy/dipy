# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

cdef int where_to_insert(cnp.float_t* arr, cnp.float_t number, int size) nogil:
    cdef:
        int idx
        cnp.float_t current
    for idx in range(size - 1, -1, -1):
        current = arr[idx]
        if number >= current:
            return idx + 1

    return 0


cdef void cumsum(cnp.float_t* arr_in, cnp.float_t* arr_out, int N) nogil:
    cdef:
        int i = 0
        cnp.float_t csum = 0
    for i in range(N):
        csum += arr_in[i]
        arr_out[i] = csum


cdef void copy_point(double * a, double * b) nogil:
    cdef:
        int i = 0
    for i in range(3):
        b[i] = a[i]


cdef void scalar_muliplication_point(double * a, double scalar) nogil:
    cdef:
        int i = 0
    for i in range(3):
        a[i] *= scalar


cdef double norm(double * v) nogil:
    """Compute the vector norm.

    Parameters
    ----------
    v : double[3]
        input vector.

    """
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


cdef double dot(double * v1, double * v2) nogil:
    """Compute vectors dot product.

    Parameters
    ----------
    v1 : double[3]
        input vector 1.
    v2 : double[3]
        input vector 2.

    Returns
    -------
    _ : double
        dot product of input vectors.
    """
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


cdef void normalize(double * v) nogil:
    """Normalize the vector.

    Parameters
    ----------
    v : double[3]
        input vector

    Notes
    -----
    Overwrites the first argument.

    """
    cdef double scale = 1.0 / norm(v)
    v[0] = v[0] * scale
    v[1] = v[1] * scale
    v[2] = v[2] * scale


cdef void cross(double * out, double * v1, double * v2) nogil:
    """Compute vectors cross product.

    Parameters
    ----------
    out : double[3]
        output vector.
    v1 : double[3]
        input vector 1.
    v2 : double[3]
        input vector 2.

    Notes
    -----
    Overwrites the first argument.

    """
    out[0] = v1[1] * v2[2] - v1[2] * v2[1]
    out[1] = v1[2] * v2[0] - v1[0] * v2[2]
    out[2] = v1[0] * v2[1] - v1[1] * v2[0]


cdef void random_vector(double * out) nogil:
    """Generate a unit random vector

    Parameters
    ----------
    out : double[3]
        output vector

    Notes
    -----
    Overwrites the input
    """
    out[0] = 2.0 * random() - 1.0
    out[1] = 2.0 * random() - 1.0
    out[2] = 2.0 * random() - 1.0
    normalize(out)


cdef void random_perpendicular_vector(double * out, double * v) nogil:
    """Generate a random perpendicular vector

    Parameters
    ----------
    out : double[3]
        output vector

    v : double[3]
        input vector

    Notes
    -----
    Overwrites the first argument
    """
    cdef double[3] tmp

    random_vector(tmp)
    cross(out, v, tmp)
    normalize(out)


cpdef (double, double) random_point_within_circle(double r) nogil:
    """Generate a random point within a circle

    Parameters
    ----------
    r : double
        The radius of the circle

    Returns
    -------
    x : double
        x coordinate of the random point

    y : double
        y coordinate of the random point

    """
    cdef double x = 1.0
    cdef double y = 1.0

    while (x * x + y * y) > 1:
        x = 2.0 * random() - 1.0
        y = 2.0 * random() - 1.0
    return (r * x, r * y)


cpdef double random() nogil:
    """Sample a random number between 0 and 1.

    Returns
    -------
    _ : double
        random number.
    """
    return rand() / float(RAND_MAX)


cpdef void seed(cnp.npy_uint32 s) nogil:
    """Set the random seed of stdlib.

    Parameters
    ----------
    s : cnp.npy_uint32
        random seed.
    """
    srand(s)
