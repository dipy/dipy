# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: cdivision=True

from dipy.utils.deprecator import deprecate_with_version

cimport numpy as cnp


cdef void take(
    double* odf,
    cnp.npy_intp* indices,
    cnp.npy_intp n_indices,
    double* values_out,
) noexcept nogil:
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
    cdef int lo = 0, hi = size, mid
    while lo < hi:
        mid = (lo + hi) >> 1
        if number >= arr[mid]:
            lo = mid + 1
        else:
            hi = mid
    return lo


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


cdef void scalar_multiplication_point(double * a, double scalar) noexcept nogil:
    cdef:
        cnp.npy_intp i = 0
    for i in range(3):
        a[i] *= scalar


# Python-level deprecated wrapper (emits DeprecationWarning)
@deprecate_with_version(
    "scalar_muliplication_point is a deprecated spelling. "
    "Use scalar_multiplication_point instead.",
    since="1.13",
    until="2.0",
)
def scalar_muliplication_point(double[::1] point, double scalar):
    """Deprecated. Use :func:`scalar_multiplication_point` instead."""
    scalar_multiplication_point(&point[0], scalar)


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


cdef void random_perpendicular_vector(
    double * out, double * v, RNGState* rng=NULL
) noexcept nogil:
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


cdef (double, double) random_point_within_circle(
    double r, RNGState* rng=NULL
) noexcept nogil:
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


cdef void seed_rng(RNGState* rng_state, cnp.npy_uint64 seed) noexcept nogil:
    """
    Seed the RNG state with the PCG32 generator (thread-safe).

    Parameters
    ----------
    rng_state : RNGState*
        The RNG state.
    seed : int
        The seed value.

    """
    rng_state.kind = RNG_PCG32
    rng_state.state = 0
    rng_state.inc = (seed << 1) | 1
    next_rng(rng_state)  # Advance the state
    rng_state.state += seed
    next_rng(rng_state)


cdef inline cnp.npy_uint32 _float_bits(double x) noexcept nogil:
    cdef float f = <float>x
    return (<cnp.npy_uint32*>&f)[0]


cdef void seed_philox_streamline(RNGState* rng_state, cnp.npy_uint64 seed,
                                 double* position,
                                 double* direction) noexcept nogil:
    """
    Seed the Philox4x32-10 stream of a streamline (thread-safe).

    This is the generator of every tracker, on the CPU and on the GPU
    backends of :mod:`dipy.tracking.accel`. The key is ``seed``; the stream
    is a Philox hash of the float32 bits of the seed position (voxel
    coordinates) and initial direction, so a streamline draws the same
    numbers on every backend, whatever its index, the chunking or the
    number of threads. The kernels implement the same function.

    Parameters
    ----------
    rng_state : RNGState*
        The RNG state.
    seed : int
        The random seed (Philox key).
    position : double*
        Seed position (3 values).
    direction : double*
        Initial direction (3 values).

    """
    cdef:
        cnp.npy_uint32 ctr[4]
        cnp.npy_uint32 key[2]
        cnp.npy_uint32 a[4]
        cnp.npy_uint32 b[4]
        int i

    key[0] = <cnp.npy_uint32>seed
    key[1] = <cnp.npy_uint32>(seed >> 32)
    for i in range(3):
        ctr[i] = _float_bits(position[i])
    ctr[3] = 0
    philox4x32_10(ctr, key, a)
    for i in range(3):
        ctr[i] = _float_bits(direction[i])
    ctr[3] = a[0]
    key[0] = a[1]
    key[1] = a[2]
    philox4x32_10(ctr, key, b)

    rng_state.kind = RNG_PHILOX
    rng_state.key[0] = <cnp.npy_uint32>seed
    rng_state.key[1] = <cnp.npy_uint32>(seed >> 32)
    rng_state.ctr[0] = 0
    rng_state.ctr[1] = b[0]
    rng_state.ctr[2] = b[1]
    rng_state.ctr[3] = b[2]
    rng_state.idx = 4  # the first block is generated on the first draw


cdef void philox4x32_10(cnp.npy_uint32* ctr, cnp.npy_uint32* key,
                        cnp.npy_uint32* out) noexcept nogil:
    """Philox4x32-10 block cipher (Salmon et al., SC '11).

    ``out`` receives the 4 random words of counter ``ctr`` and key ``key``.
    """
    cdef:
        cnp.npy_uint32 c0 = ctr[0], c1 = ctr[1], c2 = ctr[2], c3 = ctr[3]
        cnp.npy_uint32 k0 = key[0], k1 = key[1]
        cnp.npy_uint64 p0, p1
        int r

    for r in range(10):
        if r > 0:
            k0 = k0 + <cnp.npy_uint32>0x9E3779B9
            k1 = k1 + <cnp.npy_uint32>0xBB67AE85
        p0 = <cnp.npy_uint64>0xD2511F53 * c0
        p1 = <cnp.npy_uint64>0xCD9E8D57 * c2
        c0 = <cnp.npy_uint32>(p1 >> 32) ^ c1 ^ k0
        c1 = <cnp.npy_uint32>p1
        c2 = <cnp.npy_uint32>(p0 >> 32) ^ c3 ^ k1
        c3 = <cnp.npy_uint32>p0
    out[0] = c0
    out[1] = c1
    out[2] = c2
    out[3] = c3


cdef cnp.npy_uint32 next_rng(RNGState* rng_state) noexcept nogil:
    """
    Generate the next random number (thread-safe).

    Uses PCG32, or Philox4x32-10 if the state was seeded with
    ``seed_philox_streamline``.

    Parameters
    ----------
    rng_state : RNGState*
        The RNG state.

    Returns
    -------
    _ : int
        The next random number.
    """
    cdef cnp.npy_uint64 oldstate
    cdef cnp.npy_uint32 xorshifted, rot
    if rng_state.kind == RNG_PHILOX:
        if rng_state.idx == 4:
            philox4x32_10(rng_state.ctr, rng_state.key, rng_state.out)
            rng_state.ctr[0] = rng_state.ctr[0] + 1
            rng_state.idx = 0
        rng_state.idx = rng_state.idx + 1
        return rng_state.out[rng_state.idx - 1]

    oldstate = rng_state.state
    rng_state.state = oldstate * 6364136223846793005ULL + rng_state.inc
    xorshifted = ((oldstate >> 18) ^ oldstate) >> 27
    rot = oldstate >> 59
    return (xorshifted >> rot) | (xorshifted << ((32 - rot) & 31))


cdef double random_float(RNGState* rng_state) noexcept nogil:
    """
    Generate a random float in [0, 1) (thread-safe).

    With Philox, the float has 24 random bits so that it is exactly the value
    drawn, in float32, by the GPU trackers.

    Parameters
    ----------
    rng_state : RNGState*
        The RNG state.

    Returns
    -------
    _ : float
        The random float.
    """
    if rng_state.kind == RNG_PHILOX:
        return (next_rng(rng_state) >> 8) * (1.0 / 16777216.0)
    return next_rng(rng_state) * (1.0 / 4294967296.0)


def philox4x32_10_block(ctr, key):
    """Philox4x32-10 block cipher, as used by the probabilistic trackers.

    Parameters
    ----------
    ctr : sequence of 4 int
        Counter (uint32 words).
    key : sequence of 2 int
        Key (uint32 words).

    Returns
    -------
    out : list of 4 int
        Output uint32 words.
    """
    cdef cnp.npy_uint32 c[4]
    cdef cnp.npy_uint32 k[2]
    cdef cnp.npy_uint32 o[4]
    cdef int i
    for i in range(4):
        c[i] = ctr[i]
    for i in range(2):
        k[i] = key[i]
    philox4x32_10(c, k, o)
    return [o[0], o[1], o[2], o[3]]
