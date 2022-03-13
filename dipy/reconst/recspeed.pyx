# Emacs should think this is a -*- python -*- file
""" Optimized routines for creating voxel diffusion models
"""

# cython: profile=True
# cython: embedsignature=True

cimport cython

import numpy as np
cimport numpy as cnp

cdef extern from "dpy_math.h" nogil:
    double floor(double x)
    double fabs(double x)
    double cos(double x)
    double sin(double x)
    float acos(float x )
    double sqrt(double x)
    double DPY_PI


# initialize numpy runtime
cnp.import_array()

#numpy pointers
cdef inline float* asfp(cnp.ndarray pt):
    return <float *> cnp.PyArray_DATA(pt)

cdef inline double* asdp(cnp.ndarray pt):
    return <double *> cnp.PyArray_DATA(pt)


@cython.boundscheck(False)
@cython.wraparound(False)
def remove_similar_vertices(
    cython.floating[:, :] vertices,
    double theta,
    bint return_mapping=False,
    bint return_index=False):
    """Remove vertices that are less than `theta` degrees from any other

    Returns vertices that are at least theta degrees from any other vertex.
    Vertex v and -v are considered the same so if v and -v are both in
    `vertices` only one is kept. Also if v and w are both in vertices, w must
    be separated by theta degrees from both v and -v to be unique.

    Parameters
    ----------
    vertices : (N, 3) ndarray
        N unit vectors.
    theta : float
        The minimum separation between vertices in degrees.
    return_mapping : {False, True}, optional
        If True, return `mapping` as well as `vertices` and maybe `indices`
        (see below).
    return_indices : {False, True}, optional
        If True, return `indices` as well as `vertices` and maybe `mapping`
        (see below).

    Returns
    -------
    unique_vertices : (M, 3) ndarray
        Vertices sufficiently separated from one another.
    mapping : (N,) ndarray
        For each element ``vertices[i]`` ($i \in 0..N-1$), the index $j$ to a
        vertex in `unique_vertices` that is less than `theta` degrees from
        ``vertices[i]``.  Only returned if `return_mapping` is True.
    indices : (N,) ndarray
        `indices` gives the reverse of `mapping`.  For each element
        ``unique_vertices[j]`` ($j \in 0..M-1$), the index $i$ to a vertex in
        `vertices` that is less than `theta` degrees from
        ``unique_vertices[j]``.  If there is more than one element of
        `vertices` that is less than theta degrees from `unique_vertices[j]`,
        return the first (lowest index) matching value.  Only return if
        `return_indices` is True.

    """
    if vertices.shape[1] != 3:
        raise ValueError('Vertices should be 2D with second dim length 3')
    cdef:
        cnp.ndarray[cnp.float_t, ndim=2, mode='c'] unique_vertices
        cnp.ndarray[cnp.uint16_t, ndim=1, mode='c'] mapping
        cnp.ndarray[cnp.uint16_t, ndim=1, mode='c'] index
        char pass_all
        # Variable has to be large enough for all valid sizes of vertices
        cnp.npy_int32 i, j
        cnp.npy_int32 n_unique = 0
        # Large enough for all possible sizes of vertices
        cnp.npy_intp n = vertices.shape[0]
        double a, b, c, sim
        double cos_similarity = cos(DPY_PI/180 * theta)
    if n >= 2**16:  # constrained by input data type
        raise ValueError("too many vertices")
    unique_vertices = np.empty((n, 3), dtype=float)
    if return_mapping:
        mapping = np.empty(n, dtype=np.uint16)
    if return_index:
        index = np.empty(n, dtype=np.uint16)

    for i in range(n):
        pass_all = 1
        a = vertices[i, 0]
        b = vertices[i, 1]
        c = vertices[i, 2]
        # Check all other accepted vertices for similarity to this one
        for j in range(n_unique):
            sim = fabs(a * unique_vertices[j, 0] +
                       b * unique_vertices[j, 1] +
                       c * unique_vertices[j, 2])
            if sim > cos_similarity:  # too similar, drop
                pass_all = 0
                if return_mapping:
                    mapping[i] = j
                # This point unique_vertices[j] already has an entry in index,
                # so we do not need to update.
                break
        if pass_all:  # none similar, keep
            unique_vertices[n_unique, 0] = a
            unique_vertices[n_unique, 1] = b
            unique_vertices[n_unique, 2] = c
            if return_mapping:
                mapping[i] = n_unique
            if return_index:
                index[n_unique] = i
            n_unique += 1

    verts = unique_vertices[:n_unique].copy()
    if not return_mapping and not return_index:
        return verts
    out = [verts]
    if return_mapping:
        out.append(mapping)
    if return_index:
        out.append(index[:n_unique].copy())
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def search_descending(cython.floating[::1] a, double relative_threshold):
    """`i` in descending array `a` so `a[i] < a[0] * relative_threshold`

    Call ``T = a[0] * relative_threshold``. Return value `i` will be the
    smallest index in the descending array `a` such that ``a[i] < T``.
    Equivalently, `i` will be the largest index such that ``all(a[:i] >= T)``.
    If all values in `a` are >= T, return the length of array `a`.

    Parameters
    ----------
    a : ndarray, ndim=1, c-contiguous
        Array to be searched.  We assume `a` is in descending order.
    relative_threshold : float
        Applied threshold will be ``T`` with ``T = a[0] * relative_threshold``.

    Returns
    -------
    i : np.intp
        If ``T = a[0] * relative_threshold`` then `i` will be the largest index
        such that ``all(a[:i] >= T)``.  If all values in `a` are >= T then
        `i` will be `len(a)`.

    Examples
    --------
    >>> a = np.arange(10, 0, -1, dtype=float)
    >>> a
    array([ 10.,   9.,   8.,   7.,   6.,   5.,   4.,   3.,   2.,   1.])
    >>> search_descending(a, 0.5)
    6
    >>> a < 10 * 0.5
    array([False, False, False, False, False, False,  True,  True,  True,  True], dtype=bool)
    >>> search_descending(a, 1)
    1
    >>> search_descending(a, 2)
    0
    >>> search_descending(a, 0)
    10

    """
    if a.shape[0] == 0:
        return 0

    cdef:
        cnp.npy_intp left = 0
        cnp.npy_intp right = a.shape[0]
        cnp.npy_intp mid
        double threshold = relative_threshold * a[0]

    while left != right:
        mid = (left + right) // 2
        if a[mid] >= threshold:
            left = mid + 1
        else:
            right = mid
    return left


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.profile(True)
def local_maxima(double[:] odf, cnp.uint16_t[:, :] edges):
    """Local maxima of a function evaluated on a discrete set of points.

    If a function is evaluated on some set of points where each pair of
    neighboring points is an edge in edges, find the local maxima.

    Parameters
    ----------
    odf : array, 1d, dtype=double
        The function evaluated on a set of discrete points.
    edges : array (N, 2)
        The set of neighbor relations between the points. Every edge, ie
        `edges[i, :]`, is a pair of neighboring points.

    Returns
    -------
    peak_values : ndarray
        Value of odf at a maximum point. Peak values is sorted in descending
        order.
    peak_indices : ndarray
        Indices of maximum points. Sorted in the same order as `peak_values` so
        `odf[peak_indices[i]] == peak_values[i]`.

    Notes
    -----
    A point is a local maximum if it is > at least one neighbor and >= all
    neighbors. If no points meet the above criteria, 1 maximum is returned such
    that `odf[maximum] == max(odf)`.

    See Also
    --------
    dipy.core.sphere

    """
    cdef:
        cnp.ndarray[cnp.npy_intp] wpeak
    wpeak = np.zeros((odf.shape[0],), dtype=np.intp)
    count = _compare_neighbors(odf, edges, &wpeak[0])
    if count == -1:
        raise IndexError("Values in edges must be < len(odf)")
    elif count == -2:
        raise ValueError("odf can not have nans")
    indices = wpeak[:count].copy()
    # Get peak values return
    values = np.take(odf, indices)
    # Sort both values and indices
    _cosort(values, indices)
    return values, indices


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _cosort(double[::1] A, cnp.npy_intp[::1] B) nogil:
    """Sorts `A` in-place and applies the same reordering to `B`"""
    cdef:
        cnp.npy_intp n = A.shape[0]
        cnp.npy_intp hole
        double insert_A
        long insert_B

    for i in range(1, n):
        insert_A = A[i]
        insert_B = B[i]
        hole = i
        while hole > 0 and insert_A > A[hole -1]:
            A[hole] = A[hole - 1]
            B[hole] = B[hole - 1]
            hole -= 1
        A[hole] = insert_A
        B[hole] = insert_B


@cython.wraparound(False)
@cython.boundscheck(False)
cdef long _compare_neighbors(double[:] odf, cnp.uint16_t[:, :] edges,
                             cnp.npy_intp *wpeak_ptr) nogil:
    """Compares every pair of points in edges

    Parameters
    ----------
    odf : array of double
        values of points on sphere.
    edges : array of uint16
        neighbor relationships on sphere. Every set of neighbors on the sphere
        should be an edge.
    wpeak_ptr : pointer
        pointer to a block of memory which will be updated with the result of
        the comparisons. This block of memory must be large enough to hold
        len(odf) longs. The first `count` elements of wpeak will be updated
        with the indices of the peaks.

    Returns
    -------
    count : long
        Number of maxima in odf. A value < 0 indicates an error:
            -1 : value in edges too large, >= than len(odf)
            -2 : odf contains nans
    """
    cdef:
        cnp.npy_intp lenedges = edges.shape[0]
        cnp.npy_intp lenodf = odf.shape[0]
        cnp.npy_intp i
        cnp.uint16_t find0, find1
        double odf0, odf1
        long count = 0

    for i in range(lenedges):

        find0 = edges[i, 0]
        find1 = edges[i, 1]
        if find0 >= lenodf or find1 >= lenodf:
            count = -1
            break
        odf0 = odf[find0]
        odf1 = odf[find1]

        """
        Here `wpeak_ptr` is used as an indicator array that can take one of
        three values.  If `wpeak_ptr[i]` is:
        * -1 : point i of the sphere is smaller than at least one neighbor.
        *  0 : point i is equal to all its neighbors.
        *  1 : point i is > at least one neighbor and >= all its neighbors.

        Each iteration of the loop is a comparison between neighboring points
        (the two point of an edge). At each iteration we update wpeak_ptr in the
        following way::

            wpeak_ptr[smaller_point] = -1
            if wpeak_ptr[larger_point] == 0:
                wpeak_ptr[larger_point] = 1

        If the two points are equal, wpeak is left unchanged.
        """
        if odf0 < odf1:
            wpeak_ptr[find0] = -1
            wpeak_ptr[find1] |= 1
        elif odf0 > odf1:
            wpeak_ptr[find0] |= 1
            wpeak_ptr[find1] = -1
        elif (odf0 != odf0) or (odf1 != odf1):
            count = -2
            break

    if count < 0:
        return count

    # Count the number of peaks and use first count elements of wpeak_ptr to
    # hold indices of those peaks
    for i in range(lenodf):
        if wpeak_ptr[i] > 0:
            wpeak_ptr[count] = i
            count += 1

    return count


@cython.boundscheck(False)
@cython.wraparound(False)
def le_to_odf(cnp.ndarray[double, ndim=1] odf,
                 cnp.ndarray[double, ndim=1] LEs,
                 cnp.ndarray[double, ndim=1] radius,
                 int odfn,
                 int radiusn,
                 int anglesn):
    """odf for interpolated Laplacian normalized signal
    """
    cdef int m, i, j

    with nogil:
        for m in range(odfn):
            for i in range(radiusn):
                for j in range(anglesn):
                    odf[m]=odf[m]-LEs[(m*radiusn+i)*anglesn+j]*radius[i]

    return


@cython.boundscheck(False)
@cython.wraparound(False)
def sum_on_blocks_1d(cnp.ndarray[double, ndim=1] arr,
    cnp.ndarray[long, ndim=1] blocks,
    cnp.ndarray[double, ndim=1] out,int outn):
    """Summations on blocks of 1d array
    """
    cdef:
        int m,i,j
        double sum

    with nogil:
        j=0
        for m in range(outn):
            sum=0
            for i in range(j,j+blocks[m]):
                sum+=arr[i]
            out[m]=sum
            j+=blocks[m]
    return


def argmax_from_adj(vals, vertex_inds, adj_inds):
    """Indices of local maxima from `vals` given adjacent points

    Parameters
    ----------
    vals : (N,) array, dtype np.float64
       values at all vertices referred to in either of `vertex_inds` or
       `adj_inds`'
    vertex_inds : (V,) array
       indices into `vals` giving vertices that may be local maxima.
    adj_inds : sequence
       For every vertex in ``vertex_inds``, the indices (into `vals`) of
       the neighboring points

    Returns
    -------
    inds : (M,) array
       Indices into `vals` giving local maxima of vals, given topology
       from `adj_inds`, and restrictions from `vertex_inds`.  Inds are
       returned sorted by value at that index - i.e. smallest value (at
       index) first.
    """
    cvals, cvertinds = proc_reco_args(vals, vertex_inds)
    cadj_counts, cadj_inds = adj_to_countarrs(adj_inds)
    return argmax_from_countarrs(cvals, cvertinds, cadj_counts, cadj_inds)


def proc_reco_args(vals, vertinds):
    vals = np.ascontiguousarray(vals.astype(float))
    vertinds = np.ascontiguousarray(vertinds.astype(np.uint32))
    return vals, vertinds


def adj_to_countarrs(adj_inds):
    """Convert adjacency sequence to counts and flattened indices

    We use this to provide expected input to ``argmax_from_countarrs``

    Parameters
    ----------
    adj_indices : sequence
       length V sequence of sequences, where sequence ``i`` contains the
       neighbors of a particular vertex.

    Returns
    -------
    counts : (V,) array
       Number of neighbors for each vertex
    adj_inds : (n,) array
       flat array containing `adj_indices` unrolled as a vector
    """
    counts = []
    all_inds = []
    for verts in adj_inds:
        v = list(verts)
        all_inds += v
        counts.append(len(v))
    adj_inds = np.array(all_inds, dtype=np.uint32)
    counts = np.array(counts, dtype=np.uint32)
    return counts, adj_inds


# prefetch argsort for small speedup
cdef object argsort = np.argsort


def argmax_from_countarrs(cnp.ndarray vals,
                          cnp.ndarray vertinds,
                          cnp.ndarray adj_counts,
                          cnp.ndarray adj_inds):
    """Indices of local maxima from `vals` from count, array neighbors

    Parameters
    ----------
    vals : (N,) array, dtype float
       values at all vertices referred to in either of `vertex_inds` or
       `adj_inds`'
    vertinds : (V,) array, dtype uint32
       indices into `vals` giving vertices that may be local maxima.
    adj_counts : (V,) array, dtype uint32
       For every vertex ``i`` in ``vertex_inds``, the number of
       neighbors for vertex ``i``
    adj_inds : (P,) array, dtype uint32
       Indices for neighbors for each point.  ``P=sum(adj_counts)``

    Returns
    -------
    inds : (M,) array
       Indices into `vals` giving local maxima of vals, given topology
       from `adj_counts` and `adj_inds`, and restrictions from
       `vertex_inds`.  Inds are returned sorted by value at that index -
       i.e. smallest value (at index) first.
    """
    cdef:
        cnp.ndarray[cnp.float64_t, ndim=1] cvals = vals
        cnp.ndarray[cnp.uint32_t, ndim=1] cvertinds = vertinds
        cnp.ndarray[cnp.uint32_t, ndim=1] cadj_counts = adj_counts
        cnp.ndarray[cnp.uint32_t, ndim=1] cadj_inds = adj_inds
        # temporary arrays for storing maxes
        cnp.ndarray[cnp.float64_t, ndim=1] maxes = vals.copy()
        cnp.ndarray[cnp.uint32_t, ndim=1] maxinds = vertinds.copy()
        cnp.npy_intp i, j, V, C, n_maxes=0, adj_size, adj_pos=0
        int is_max
        cnp.float64_t *vals_ptr
        double val
        cnp.uint32_t vert_ind, ind
        cnp.uint32_t *vertinds_ptr
        cnp.uint32_t *counts_ptr
        cnp.uint32_t *adj_ptr
        cnp.uint32_t vals_size, vert_size
    if not (cnp.PyArray_ISCONTIGUOUS(cvals) and
            cnp.PyArray_ISCONTIGUOUS(cvertinds) and
            cnp.PyArray_ISCONTIGUOUS(cadj_counts) and
            cnp.PyArray_ISCONTIGUOUS(cadj_inds)):
        raise ValueError('Need contiguous arrays as input')
    vals_size = cnp.PyArray_DIM(cvals, 0)
    vals_ptr = <cnp.float64_t *> cnp.PyArray_DATA(cvals)
    vertinds_ptr = <cnp.uint32_t *> cnp.PyArray_DATA(cvertinds)
    adj_ptr = <cnp.uint32_t *> cnp.PyArray_DATA(cadj_inds)
    counts_ptr = <cnp.uint32_t *> cnp.PyArray_DATA(cadj_counts)
    V = cnp.PyArray_DIM(cadj_counts, 0)
    adj_size = cnp.PyArray_DIM(cadj_inds, 0)
    if cnp.PyArray_DIM(cvertinds, 0) < V:
        raise ValueError('Too few indices for adj arrays')
    for i in range(V):
        vert_ind = vertinds_ptr[i]
        if vert_ind >= vals_size:
            raise IndexError('Overshoot on vals')
        val = vals_ptr[vert_ind]
        C = counts_ptr[i]
        # check for overshoot
        adj_pos += C
        if adj_pos > adj_size:
            raise IndexError('Overshoot on adj_inds array')
        is_max = 1
        for j in range(C):
            ind = adj_ptr[j]
            if ind >= vals_size:
                raise IndexError('Overshoot on vals')
            if val <= vals_ptr[ind]:
                is_max = 0
                break
        if is_max:
            maxinds[n_maxes] = vert_ind
            maxes[n_maxes] = val
            n_maxes +=1
        adj_ptr += C
    if n_maxes == 0:
        return np.array([])
    # fancy indexing always produces a copy
    return maxinds[argsort(maxes[:n_maxes])]
