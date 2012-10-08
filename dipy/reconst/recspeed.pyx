# Emacs should think this is a -*- python -*- file
""" Optimized routines for creating voxel diffusion models
"""

# cython: profile=True
# cython: embedsignature=True

cimport cython

import numpy as np
cimport numpy as cnp



cdef extern from "math.h" nogil:
    double floor(double x)
    double fabs(double x)
    double log2(double x)
    double cos(double x)
    double sin(double x)
    float acos(float x )
    bint isnan(double x)
    double sqrt(double x)


DEF PI=3.1415926535897931


# initialize numpy runtime
cnp.import_array()

#numpy pointers
cdef inline float* asfp(cnp.ndarray pt):
    return <float *>pt.data

cdef inline double* asdp(cnp.ndarray pt):
    return <double *>pt.data

@cython.wraparound(False)
def trilinear_interp(cnp.ndarray[cnp.float_t, ndim=4] data, 
                     cnp.ndarray[cnp.float_t, ndim=1] index,
                     cnp.ndarray[cnp.float_t, ndim=1] voxel_size):
    """Interpolates data at index

    Interpolates data from a 4d volume, first 3 dimensions are x, y, z the
    last dimension holds data.
    """
    cdef:
        double x = index[0] / voxel_size[0] - .5
        double y = index[1] / voxel_size[1] - .5
        double z = index[2] / voxel_size[2] - .5
        double weight
        int x_ind = <int> floor(x)
        int y_ind = <int> floor(y)
        int z_ind = <int> floor(z)
        int ii, jj, kk, LL
        int last_d = data.shape[3]
        cnp.ndarray[cnp.float_t, ndim=1] result=np.zeros(last_d)
    x = x % 1
    y = y % 1
    z = z % 1

    for ii from 0 <= ii <= 1:
        for jj from 0 <= jj <= 1:
            for kk from 0 <= kk <= 1:
                weight = wght(ii, x)*wght(jj, y)*wght(kk, z)
                for LL from 0 <= LL < last_d:
                    result[LL] += data[x_ind+ii,y_ind+jj,z_ind+kk,LL]*weight
    return result

cdef double wght(int i, double r) nogil:
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
        double cos_similarity = cos(PI/180 * theta)
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


#@cython.boundscheck(False)
@cython.wraparound(False)
def local_maxima(cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] codf,
                 cnp.ndarray[cnp.uint16_t, ndim=2, mode='c'] cedges):
    """Given a function, odf, and neighbor pairs, edges, finds the local maxima

    If a function is evaluated on some set of points where each pair of
    neighboring points is in edges, the function compares each pair of
    neighbors and returns the value and location of each point that is >= all
    its neighbors.

    Parameters
    ----------
    odf : array_like
        The odf of some function evaluated at some set of points
    edges : array_like (N, 2)
        every edges(i,:) is a pair of neighboring points

    Returns
    -------
    peaks : ndarray
        odf at local maximums, orders the peaks in descending order
    inds : ndarray
        location of local maximums, indexes to odf array so that
        odf[inds[i]] == peaks[i]

    Note
    ----
    Comparing on edges might be faster then comparing on faces if edges does
    not contain repeated entries. Additionally in the event that some function
    is symmetric in some way, that symmetry can be exploited to further reduce
    the domain of the search and the number of input edges. This is done in the
    create_half_unit_sphere function of dipy.core.triangle_subdivide for
    functions with antipodal symmetry.

    See Also
    --------
    create_half_unit_sphere

    """

    cdef:
        cnp.ndarray[cnp.uint8_t, ndim=1, mode='c'] cpeak = np.ones(len(codf),
                                                                   'uint8')
        int i=0
        int lenedges = len(cedges)
        int find0, find1
        double odf0, odf1

    for i from 0 <= i < lenedges:

        find0 = cedges[i,0]
        find1 = cedges[i,1]
        odf0 = codf[find0]
        odf1 = codf[find1]

        if odf0 > odf1:
            cpeak[find1] = 0
        elif odf0 < odf1:
            cpeak[find0] = 0
        elif (odf0 != odf0) or (odf1 != odf1):
            raise ValueError("odf cannot have nans")

    peakidx = cpeak.nonzero()[0]
    peakvalues = codf[peakidx]
    order = peakvalues.argsort()[::-1]
    return peakvalues[order], peakidx[order]


@cython.boundscheck(False)
@cython.wraparound(False)
def pdf_to_odf(cnp.ndarray[double, ndim=1] odf, \
                 cnp.ndarray[double, ndim=1] PrIs,\
                 cnp.ndarray[double, ndim=1] radius,\
                 int odfn,int radiusn):
    """ Expecting interpolated pdf then calculates the odf for that pdf
    """    
    cdef int m,i
    
    with nogil:
        for m in range(odfn):
            for i in range(radiusn):
                odf[m]=odf[m]+PrIs[m*radiusn+i]*radius[i]*radius[i]

    return

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

