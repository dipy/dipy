# Emacs should think this is a -*- python -*- file
""" Optimized routines for creating voxel diffusion models
"""

# cython: profile=True
# cython: embedsignature=True

cimport cython

import numpy as np
cimport numpy as cnp

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

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
    return <float *>pt.data

cdef inline double* asdp(cnp.ndarray pt):
    return <double *>pt.data


cdef void splitoffset(float *offset, size_t *index, size_t shape) nogil:
    """Splits a global offset into an integer index and a relative offset"""
    offset[0] -= .5
    if offset[0] <= 0:
        index[0] = 0
        offset[0] = 0.
    elif offset[0] >= (shape - 1):
        index[0] = shape - 2
        offset[0] = 1.
    else:
        index[0] = <size_t> offset[0]
        offset[0] = offset[0] - index[0]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def trilinear_interp(cnp.ndarray[cnp.float32_t, ndim=4, mode='strided'] data,
                     cnp.ndarray[cnp.float_t, ndim=1, mode='strided'] index,
                     cnp.ndarray[cnp.float_t, ndim=1, mode='c'] voxel_size):
    """Interpolates data at index

    Interpolates data from a 4d volume, first 3 dimensions are x, y, z the
    last dimension holds data.
    """
    cdef:
        float x = index[0] / voxel_size[0]
        float y = index[1] / voxel_size[1]
        float z = index[2] / voxel_size[2]
        float weight
        size_t x_ind, y_ind, z_ind, ii, jj, kk, LL
        size_t last_d = data.shape[3]
        bint bounds_check
        cnp.ndarray[cnp.float32_t, ndim=1, mode='c'] result
    bounds_check = (x < 0 or y < 0 or z < 0 or
                    x > data.shape[0] or
                    y > data.shape[1] or
                    z > data.shape[2])
    if bounds_check:
        raise IndexError

    splitoffset(&x, &x_ind, data.shape[0])
    splitoffset(&y, &y_ind, data.shape[1])
    splitoffset(&z, &z_ind, data.shape[2])

    result = np.zeros(last_d, dtype='float32')
    for ii from 0 <= ii <= 1:
        for jj from 0 <= jj <= 1:
            for kk from 0 <= kk <= 1:
                weight = wght(ii, x)*wght(jj, y)*wght(kk, z)
                for LL from 0 <= LL < last_d:
                    result[LL] += data[x_ind+ii,y_ind+jj,z_ind+kk,LL]*weight
    return result


@cython.profile(False)
cdef float wght(int i, float r) nogil:
    if i:
        return r
    else:
        return 1.-r


@cython.boundscheck(False)
@cython.wraparound(False)
def remove_similar_vertices(cnp.ndarray[cnp.float_t, ndim=2, mode='strided'] vertices,
                            double theta,
                            bint return_mapping=False,
                            bint return_index=False):
    """remove_similar_vertices(vertices, theta)

    Returns vertices that are separated by at least theta degrees from all
    other vertices. Vertex v and -v are considered the same so if v and -v are
    both in `vertices` only one is kept. Also if v and w are both in vertices,
    w must be separated by theta degrees from both v and -v to be unique.

    Parameters
    ----------
    vertices : (N, 3) ndarray
        N unit vectors
    theta : float
        The minimum separation between vertices in degrees.

    Returns
    -------
    unique_vertices : (M, 3) ndarray
        Vertices sufficiently separated from one another.
    mapping : (N,) ndarray
        Indices into unique_vertices. For each vertex in `vertices` the index
        of a vertex in `unique_vertices` that is less than theta degrees away.

    """
    if vertices.shape[1] != 3:
        raise ValueError()
    cdef:
        cnp.ndarray[cnp.float_t, ndim=2, mode='c'] unique_vertices
        cnp.ndarray[cnp.uint16_t, ndim=1, mode='c'] mapping
        cnp.ndarray[cnp.uint16_t, ndim=1, mode='c'] index
        char pass_all
        size_t i, j
        size_t count = 0
        size_t n = vertices.shape[0]
        double a, b, c, sim
        double cos_similarity = cos(DPY_PI/180 * theta)
    if n > 2**16:
        raise ValueError("too many vertices")
    unique_vertices = np.empty((n, 3), dtype=np.float)
    if return_mapping:
        mapping = np.empty(n, dtype=np.uint16)
    else:
        mapping = None
    if return_index:
        index = np.empty(n, dtype=np.uint16)
    else:
        index = None

    for i in range(n):
        pass_all = 1
        a = vertices[i, 0]
        b = vertices[i, 1]
        c = vertices[i, 2]
        for j in range(count):
            sim = fabs(a * unique_vertices[j, 0] +
                       b * unique_vertices[j, 1] +
                       c * unique_vertices[j, 2])
            if sim > cos_similarity:
                pass_all = 0
                if return_mapping:
                    mapping[i] = j
                break
        if pass_all:
            unique_vertices[count, 0] = a
            unique_vertices[count, 1] = b
            unique_vertices[count, 2] = c
            if return_mapping:
                mapping[i] = count
            if return_index:
                index[count] = i
            count += 1

    if return_mapping and return_index:
        return unique_vertices[:count].copy(), mapping, index[:count].copy()
    elif return_mapping:
        return unique_vertices[:count].copy(), mapping
    elif return_index:
        return unique_vertices[:count].copy(), index[:count].copy()
    else:
        return unique_vertices[:count].copy()


@cython.boundscheck(False)
@cython.wraparound(False)
def search_descending(cnp.ndarray[cnp.float_t, ndim=1, mode='c'] a,
                       double relative_threshould):
    """Searches a descending array for the first element smaller than some
    threshold

    Parameters
    ----------
    a : ndarray, ndim=1, c-contiguous
        Array to be searched.
    relative_threshold : float
        Threshold relative to `a[0]`.

    Returns
    -------
    i : int
        The greatest index such that ``all(a[:i] >= relative_threshold *
        a[0])``.

    Note
    ----
    This function will never return 0, 1 is returned if ``a[0] <
    relative_threshold * a[0]`` or if ``len(a) == 0``.

    """
    if a.shape[0] == 0:
        return 1

    cdef:
        size_t left = 1
        size_t right = a.shape[0]
        size_t mid
        double threshold = relative_threshould * a[0]

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
def local_maxima(cnp.ndarray odf, cnp.ndarray edges):
    """local_maxima(odf, edges)
    Finds the local maxima of a function evaluated on a discrete set of points.

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

    Note
    ----
    A point is a local maximum if it is > at least one neighbor and >= all
    neighbors. If no points meets the above criteria, 1 maximum is returned
    such that `odf[maximum] == max(odf)`.

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
    values = odf.take(indices)
    # Sort both values and indices
    _cosort(values, indices)
    return values, indices


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _cosort(double[::1] A, cnp.npy_intp[::1] B) nogil:
    """Sorts A inplace and applies the same reording to B"""
    cdef:
        size_t n = A.shape[0]
        size_t hole
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
    odf :
        values of points on sphere.
    edges :
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
        size_t lenedges = edges.shape[0]
        size_t lenodf = odf.shape[0]
        size_t i
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

    # If count == 0, all values of odf are equal, and point 0 is returned as a
    # peak to satisfy the requirement that peak_values[0] == max(odf).
    if count == 0:
        count = 1
        wpeak_ptr[0] = 0

    return count


@cython.boundscheck(False)
@cython.wraparound(False)
def le_to_odf(cnp.ndarray[double, ndim=1] odf, \
                 cnp.ndarray[double, ndim=1] LEs,\
                 cnp.ndarray[double, ndim=1] radius,\
                 int odfn,\
                 int radiusn,\
                 int anglesn):
    """ Expecting interpolated laplacian normalized signal and  then calculates the odf for that.
    """    
    cdef int m,i,j    
    
    with nogil:    
        for m in range(odfn):
            for i in range(radiusn):
                for j in range(anglesn):
                    odf[m]=odf[m]-LEs[(m*radiusn+i)*anglesn+j]*radius[i]

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_on_blocks_1d(cnp.ndarray[double, ndim=1] arr,\
    cnp.ndarray[long, ndim=1] blocks,\
    cnp.ndarray[double, ndim=1] out,int outn):
    """ Summations on blocks of 1d array
    
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
    """ Indices of local maxima from `vals` given adjacent points

    Parameters
    ------------
    vals : (N,) array, dtype np.float64
       values at all vertices referred to in either of `vertex_inds` or
       `adj_inds`'
    vertex_inds : (V,) array
       indices into `vals` giving vertices that may be local maxima.
    adj_inds : sequence
       For every vertex in ``vertex_inds``, the indices (into `vals`) of
       the neighboring points

    Returns
    ---------
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
    vals = np.ascontiguousarray(vals.astype(np.float))
    vertinds = np.ascontiguousarray(vertinds.astype(np.uint32))
    return vals, vertinds


def adj_to_countarrs(adj_inds):
    """ Convert adjacency sequence to counts and flattened indices

    We use this to provide expected input to ``argmax_from_countarrs``

    Parameters
    ------------
    adj_indices : sequence
       length V sequence of sequences, where sequence ``i`` contains the
       neighbors of a particular vertex.

    Returns
    ---------
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
    """ Indices of local maxima from `vals` from count, array neighbors

    Parameters
    ------------
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
    ---------
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
        cnp.uint32_t vert_ind, *vertinds_ptr, *counts_ptr, *adj_ptr, ind
        cnp.uint32_t vals_size, vert_size
    if not (cnp.PyArray_ISCONTIGUOUS(cvals) and
            cnp.PyArray_ISCONTIGUOUS(cvertinds) and
            cnp.PyArray_ISCONTIGUOUS(cadj_counts) and
            cnp.PyArray_ISCONTIGUOUS(cadj_inds)):
        raise ValueError('Need contiguous arrays as input')
    vals_size = cvals.shape[0]
    vals_ptr = <cnp.float64_t *>cvals.data
    vertinds_ptr = <cnp.uint32_t *>cvertinds.data
    adj_ptr = <cnp.uint32_t *>cadj_inds.data
    counts_ptr = <cnp.uint32_t *>cadj_counts.data
    V = cadj_counts.shape[0]
    adj_size = cadj_inds.shape[0]
    if cvertinds.shape[0] < V:
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

