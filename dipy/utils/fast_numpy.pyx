# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
from libc.stdio cimport printf
cimport numpy as cnp


cdef void take(
    double* odf,
    cnp.npy_intp* indices,
    cnp.npy_intp n_indices,
    double* values_out) noexcept nogil:
    """
    Mimic np.take(odf, indices) in Cython using pointers.

    Parameters
    ----------
    odf : double*
        Pointer to the input array from which values are taken.
    indices : npy_intp*
        Pointer to the array of indices specifying which elements to take.
    n_indices : npy_intp
        Number of indices to process.
    values_out : double*
        Pointer to the output array where the selected values will be stored.
    """
    cdef int i
    for i in range(n_indices):
        values_out[i] = odf[indices[i]]


cdef int where_to_insert(cnp.float_t* arr, cnp.float_t number, int size) noexcept nogil:
    cdef:
        int idx
        cnp.float_t current
    for idx in range(size - 1, -1, -1):
        current = arr[idx]
        if number >= current:
            return idx + 1

    return 0


cdef void cumsum(cnp.float_t* arr_in, cnp.float_t* arr_out, int N) noexcept nogil:
    cdef:
        cnp.npy_intp i = 0
        cnp.float_t csum = 0
    for i in range(N):
        csum += arr_in[i]
        arr_out[i] = csum


cdef void copy_point(double * a, double * b) noexcept nogil:
    cdef:
        cnp.npy_intp i = 0
    for i in range(3):
        b[i] = a[i]


cdef void scalar_muliplication_point(double * a, double scalar) noexcept nogil:
    cdef:
        cnp.npy_intp i = 0
    for i in range(3):
        a[i] *= scalar


cdef double norm(double * v) noexcept nogil:
    """Compute the vector norm.

    Parameters
    ----------
    v : double[3]
        input vector.

    """
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


cdef double dot(double * v1, double * v2) noexcept nogil:
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


cdef void normalize(double * v) noexcept nogil:
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


cdef void cross(double * out, double * v1, double * v2) noexcept nogil:
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


cdef void random_vector(double * out, RNGState* rng=NULL) noexcept nogil:
    """Generate a unit random vector

    Parameters
    ----------
    out : double[3]
        output vector

    Notes
    -----
    Overwrites the input
    """
    if rng == NULL:
        out[0] = 2.0 * random() - 1.0
        out[1] = 2.0 * random() - 1.0
        out[2] = 2.0 * random() - 1.0
    else:
        out[0] = 2.0 * random_float(rng) - 1.0
        out[1] = 2.0 * random_float(rng) - 1.0
        out[2] = 2.0 * random_float(rng) - 1.0
    normalize(out)


cdef void random_perpendicular_vector(double * out, double * v, RNGState* rng=NULL) noexcept nogil:
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

    random_vector(tmp, rng)
    cross(out, v, tmp)
    normalize(out)


cdef (double, double) random_point_within_circle(double r, RNGState* rng=NULL) noexcept nogil:
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

    if rng == NULL:
        while (x * x + y * y) > 1:
            x = 2.0 * random() - 1.0
            y = 2.0 * random() - 1.0
    else:
        while (x * x + y * y) > 1:
            x = 2.0 * random_float(rng) - 1.0
            y = 2.0 * random_float(rng) - 1.0
    return (r * x, r * y)


cpdef double random() noexcept nogil:
    """Sample a random number between 0 and 1.

    Returns
    -------
    _ : double
        random number.
    """
    return rand() / float(RAND_MAX)


cpdef void seed(cnp.npy_uint32 s) noexcept nogil:
    """Set the random seed of stdlib.

    Parameters
    ----------
    s : int
        random seed.
    """
    srand(s)


cdef void print_c_array_pointer(double* arr, int size) noexcept nogil:
    cdef int i
    for i in range(size):
        printf("%f, ", arr[i])
    printf("\n\n\n")


cdef void seed_rng(RNGState* rng_state, cnp.npy_uint64 seed) noexcept nogil:
    """
    Seed the RNG state (thread-safe).

    Parameters
    ----------
    rng_state : RNGState*
        The RNG state.
    seed : int
        The seed value.

    """
    rng_state.state = 0
    rng_state.inc = (seed << 1) | 1
    next_rng(rng_state)  # Advance the state
    rng_state.state += seed
    next_rng(rng_state)


cdef cnp.npy_uint32 next_rng(RNGState* rng_state) noexcept nogil:
    """
    Generate the next random number (thread-safe, PCG algorithm).

    Parameters
    ----------
    rng_state : RNGState*
        The RNG state.

    Returns
    -------
    _ : int
        The next random number.
    """
    cdef cnp.npy_uint64 oldstate = rng_state.state
    rng_state.state = oldstate * 6364136223846793005ULL + rng_state.inc
    cdef cnp.npy_uint32 xorshifted = ((oldstate >> 18) ^ oldstate) >> 27
    cdef cnp.npy_uint32 rot = oldstate >> 59
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31))


cdef double random_float(RNGState* rng_state) noexcept nogil:
    """
    Generate a random float in [0, 1) (thread-safe).

    Parameters
    ----------
    rng_state : RNGState*
        The RNG state.

    Returns
    -------
    _ : float
        The random float.
    """
    return next_rng(rng_state) / <double>0xffffffff
