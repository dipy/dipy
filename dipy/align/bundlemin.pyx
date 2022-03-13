#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as cnp

cimport safe_openmp as openmp
from safe_openmp cimport have_openmp

from cython.parallel import prange
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt

from dipy.utils.omp import determine_num_threads
from dipy.utils.omp cimport set_num_threads, restore_default_num_threads

cdef cnp.dtype f64_dt = np.dtype(np.float64)


cdef double min_direct_flip_dist(double *a,double *b,
                                 cnp.npy_intp rows) nogil:
    r""" Minimum of direct and flip average (MDF) distance [Garyfallidis12]
    between two streamlines.

    Parameters
    ----------
    a : double pointer
        first streamline
    b : double pointer
        second streamline
    rows : number of points of the streamline
        both tracks need to have the same number of points

    Returns
    -------
    out : double
        minimum of direct and flipped average distances

    References
    ----------
    .. [Garyfallidis12] Garyfallidis E. et al., QuickBundles a method for
                        tractography simplification, Frontiers in Neuroscience,
                        vol 6, no 175, 2012.
    """

    cdef:
        cnp.npy_intp i=0, j=0
        double sub=0, subf=0, distf=0, dist=0, tmprow=0, tmprowf=0


    for i in range(rows):
        tmprow = 0
        tmprowf = 0
        for j in range(3):
            sub = a[i * 3 + j] - b[i * 3 + j]
            subf = a[i * 3 + j] - b[(rows - 1 - i) * 3 + j]
            tmprow += sub * sub
            tmprowf += subf * subf
        dist += sqrt(tmprow)
        distf += sqrt(tmprowf)

    dist = dist / <double>rows
    distf = distf / <double>rows

    if dist <= distf:
        return dist
    return distf


def _bundle_minimum_distance_matrix(double [:, ::1] static,
                                    double [:, ::1] moving,
                                    cnp.npy_intp static_size,
                                    cnp.npy_intp moving_size,
                                    cnp.npy_intp rows,
                                    double [:, ::1] D,
                                    num_threads=None):
    """ MDF-based pairwise distance optimization function

    We minimize the distance between moving streamlines of the same number of
    points as they align with the static streamlines.

    Parameters
    ----------
    static: array
        Static streamlines
    moving: array
        Moving streamlines
    static_size : int
        Number of static streamlines
    moving_size : int
        Number of moving streamlines
    rows : int
        Number of points per streamline
    D : 2D array
        Distance matrix
    num_threads : int, optional
        Number of threads to be used for OpenMP parallelization. If None
        (default) the value of OMP_NUM_THREADS environment variable is used
        if it is set, otherwise all available threads are used. If < 0 the
        maximal number of threads minus |num_threads + 1| is used (enter -1 to
        use as many threads as possible). 0 raises an error.

    Returns
    -------
    cost : double
    """

    cdef:
        cnp.npy_intp i=0, j=0, mov_i=0, mov_j=0
        int threads_to_use = -1

    threads_to_use = determine_num_threads(num_threads)
    set_num_threads(threads_to_use)

    with nogil:

        for i in prange(static_size):
            for j in prange(moving_size):

                D[i, j] = min_direct_flip_dist(&static[i * rows, 0],
                                               &moving[j * rows, 0],
                                               rows)

    if num_threads is not None:
        restore_default_num_threads()

    return np.asarray(D)


def _bundle_minimum_distance(double [:, ::1] static,
                             double [:, ::1] moving,
                             cnp.npy_intp static_size,
                             cnp.npy_intp moving_size,
                             cnp.npy_intp rows,
                             num_threads=None):
    """ MDF-based pairwise distance optimization function

    We minimize the distance between moving streamlines of the same number of
    points as they align with the static streamlines.

    Parameters
    ----------
    static : array
        Static streamlines
    moving : array
        Moving streamlines
    static_size : int
        Number of static streamlines
    moving_size : int
        Number of moving streamlines
    rows : int
        Number of points per streamline
    num_threads : int, optional
        Number of threads to be used for OpenMP parallelization. If None
        (default) the value of OMP_NUM_THREADS environment variable is used
        if it is set, otherwise all available threads are used. If < 0 the
        maximal number of threads minus |num_threads + 1| is used (enter -1 to
        use as many threads as possible). 0 raises an error.

    Returns
    -------
    cost : double

    Notes
    -----
    The difference with ``_bundle_minimum_distance_matrix`` is that it does not
    save the full distance matrix and therefore needs much less memory.
    """

    cdef:
        cnp.npy_intp i=0, j=0
        double sum_i=0, sum_j=0, tmp=0
        double inf = np.finfo('f8').max
        double dist=0
        double * min_j
        double * min_i
        openmp.omp_lock_t lock
        int threads_to_use = -1

    threads_to_use = determine_num_threads(num_threads)
    set_num_threads(threads_to_use)

    with nogil:

        if have_openmp:
            openmp.omp_init_lock(&lock)

        min_j = <double *> malloc(static_size * sizeof(double))
        min_i = <double *> malloc(moving_size * sizeof(double))

        for i in range(static_size):
            min_j[i] = inf

        for j in range(moving_size):
            min_i[j] = inf

        for i in prange(static_size):

            for j in range(moving_size):

                tmp = min_direct_flip_dist(&static[i * rows, 0],
                                       &moving[j * rows, 0], rows)

                if have_openmp:
                    openmp.omp_set_lock(&lock)
                if tmp < min_j[i]:
                    min_j[i] = tmp

                if tmp < min_i[j]:
                    min_i[j] = tmp
                if have_openmp:
                    openmp.omp_unset_lock(&lock)

        if have_openmp:
            openmp.omp_destroy_lock(&lock)

        for i in range(static_size):
            sum_i += min_j[i]

        for j in range(moving_size):
            sum_j += min_i[j]

        free(min_j)
        free(min_i)

        dist = (sum_i / <double>static_size + sum_j / <double>moving_size)

        dist = 0.25 * dist * dist

    if num_threads is not None:
        restore_default_num_threads()

    return dist



def _bundle_minimum_distance_asymmetric(double [:, ::1] static,
                                        double [:, ::1] moving,
                                        cnp.npy_intp static_size,
                                        cnp.npy_intp moving_size,
                                        cnp.npy_intp rows):
    """ MDF-based pairwise distance optimization function

    We minimize the distance between moving streamlines of the same number of
    points as they align with the static streamlines.

    Parameters
    ----------
    static : array
        Static streamlines
    moving : array
        Moving streamlines
    static_size : int
        Number of static streamlines
    moving_size : int
        Number of moving streamlines
    rows : int
        Number of points per streamline

    Returns
    -------
    cost : double

    Notes
    -----
    The difference with ``_bundle_minimum_distance`` is that we sum the
    minimum values only for the static. Therefore, this is an asymmetric
    distance metric. This means that we are weighting only one direction of the
    registration. Not both directions. This can be very useful when we want
    to register a big set of bundles to a small set of bundles.
    See [Wanyan17]_.

    References
    ----------
    .. [Wanyan17] Wanyan and Garyfallidis, Important new insights for the
    reduction of false positives in tractograms emerge from streamline-based
    registration and pruning, International Society for Magnetic Resonance in
    Medicine, Honolulu, Hawai, 2017.

    """

    cdef:
        cnp.npy_intp i=0, j=0
        double sum_i=0, sum_j=0, tmp=0
        double inf = np.finfo('f8').max
        double dist=0
        double * min_j
        openmp.omp_lock_t lock

    with nogil:

        if have_openmp:
            openmp.omp_init_lock(&lock)

        min_j = <double *> malloc(static_size * sizeof(double))

        for sz_i in range(static_size):
            min_j[sz_i] = inf

        for i in prange(static_size):

            for j in range(moving_size):

                tmp = min_direct_flip_dist(&static[i * rows, 0],
                                           &moving[j * rows, 0], rows)

                if have_openmp:
                    openmp.omp_set_lock(&lock)
                if tmp < min_j[i]:
                    min_j[i] = tmp

                if have_openmp:
                    openmp.omp_unset_lock(&lock)

        if have_openmp:
            openmp.omp_destroy_lock(&lock)

        for i in range(static_size):
            sum_i += min_j[i]

        free(min_j)

        dist = sum_i / <double>static_size

    return dist


def distance_matrix_mdf(streamlines_a, streamlines_b):
    r""" Minimum direct flipped distance matrix between two streamline sets

    All streamlines need to have the same number of points

    Parameters
    ----------
    streamlines_a : sequence
       of streamlines as arrays, [(N, 3) .. (N, 3)]
    streamlines_b : sequence
       of streamlines as arrays, [(N, 3) .. (N, 3)]

    Returns
    -------
    DM : array, shape (len(streamlines_a), len(streamlines_b))
        distance matrix
    """
    cdef:
        cnp.npy_intp i, j, lentA, lentB
    # preprocess tracks
    cdef:
        cnp.npy_intp longest_track_len = 0, track_len
        longest_track_lenA, longest_track_lenB
        cnp.ndarray[object, ndim=1] tracksA64
        cnp.ndarray[object, ndim=1] tracksB64
        cnp.ndarray[cnp.double_t, ndim=2] DM

    lentA = len(streamlines_a)
    lentB = len(streamlines_b)
    tracksA64 = np.zeros((lentA,), dtype=object)
    tracksB64 = np.zeros((lentB,), dtype=object)
    DM = np.zeros((lentA,lentB), dtype=np.double)
    if streamlines_a[0].shape[0] != streamlines_b[0].shape[0]:
        msg = 'Streamlines should have the same number of points as required'
        msg += 'by the MDF distance'
        raise ValueError(msg)
    # process tracks to predictable memory layout
    for i in range(lentA):
        tracksA64[i] = np.ascontiguousarray(streamlines_a[i], dtype=f64_dt)
    for i in range(lentB):
        tracksB64[i] = np.ascontiguousarray(streamlines_b[i], dtype=f64_dt)
    # preallocate buffer array for track distance calculations
    cdef:
        cnp.float64_t *t1_ptr
        cnp.float64_t *t2_ptr
        cnp.float64_t *min_buffer
    # cycle over tracks
    cdef:
        cnp.ndarray [cnp.float64_t, ndim=2] t1, t2
        cnp.npy_intp t1_len, t2_len
        double d[2]
    t_len = tracksA64[0].shape[0]

    for i from 0 <= i < lentA:
        t1 = tracksA64[i]
        t1_ptr = <cnp.float64_t *> cnp.PyArray_DATA(t1)
        for j from 0 <= j < lentB:
            t2 = tracksB64[j]
            t2_ptr = <cnp.float64_t *> cnp.PyArray_DATA(t2)

            DM[i, j] = min_direct_flip_dist(t1_ptr, t2_ptr,t_len)

    return DM
