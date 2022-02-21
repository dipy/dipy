 # A type of -*- python -*- file
""" Optimized track distances, similarities and distanch clustering algorithms
"""

# cython: profile=True
# cython: embedsignature=True

cimport cython

from libc.stdlib cimport calloc, realloc, free

import numpy as np
from warnings import warn
cimport numpy as cnp


cdef extern from "dpy_math.h" nogil:
    double floor(double x)
    float sqrt(float x)
    float fabs(float x)
    float acos(float x )
    bint dpy_isnan(double x)
    double dpy_log2(double x)


#@cython.boundscheck(False)
#@cython.wraparound(False)

DEF biggest_double = 1.79769e+308 #np.finfo('f8').max
DEF biggest_float = 3.4028235e+38 #np.finfo('f4').max

cdef inline cnp.ndarray[cnp.float32_t, ndim=1] as_float_3vec(object vec):
    """ Utility function to convert object to 3D float vector """
    return np.squeeze(np.asarray(vec, dtype=np.float32))


cdef inline float* asfp(cnp.ndarray pt):
    return <float *> cnp.PyArray_DATA(pt)


def normalized_3vec(vec):
    """ Return normalized 3D vector

    Vector divided by Euclidean (L2) norm

    Parameters
    ----------
    vec : array-like shape (3,)

    Returns
    -------
    vec_out : array shape (3,)
    """
    cdef cnp.ndarray[cnp.float32_t, ndim=1] vec_in = as_float_3vec(vec)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] vec_out = np.zeros((3,), np.float32)
    cnormalized_3vec(<float *> cnp.PyArray_DATA(vec_in), <float*> cnp.PyArray_DATA(vec_out))
    return vec_out


def norm_3vec(vec):
    """ Euclidean (L2) norm of length 3 vector

    Parameters
    ----------
    vec : array-like shape (3,)

    Returns
    -------
    norm : float
       Euclidean norm
    """
    cdef cnp.ndarray[cnp.float32_t, ndim=1] vec_in = as_float_3vec(vec)
    return cnorm_3vec(<float *> cnp.PyArray_DATA(vec_in))


cdef inline float cnorm_3vec(float *vec):
    """ Calculate Euclidean norm of input vector

    Parameters
    ----------
    vec : float *
       length 3 float vector

    Returns
    -------
    norm : float
       Euclidean norm
    """
    cdef float v0, v1, v2
    v0 = vec[0]
    v1 = vec[1]
    v2 = vec[2]
    return sqrt(v0 * v0 + v1*v1 + v2*v2)


cdef inline void cnormalized_3vec(float *vec_in, float *vec_out):
    """ Calculate and fill normalized 3D vector

    Parameters
    ----------
    vec_in : float *
       Length 3 vector to normalize
    vec_out : float *
       Memory into which to write normalized length 3 vector

    """
    cdef float norm = cnorm_3vec(vec_in)
    cdef int i
    for i in range(3):
        vec_out[i] = vec_in[i] / norm


def inner_3vecs(vec1, vec2):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec1 = as_float_3vec(vec1)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec2 = as_float_3vec(vec2)
    return cinner_3vecs(<float *> cnp.PyArray_DATA(fvec1), <float*> cnp.PyArray_DATA(fvec2))


cdef inline float cinner_3vecs(float *vec1, float *vec2) nogil:
    cdef int i
    cdef float ip = 0
    for i from 0<=i<3:
        ip += vec1[i]*vec2[i]
    return ip


def sub_3vecs(vec1, vec2):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec1 = as_float_3vec(vec1)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec2 = as_float_3vec(vec2)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] vec_out = np.zeros((3,), np.float32)
    csub_3vecs(<float *> cnp.PyArray_DATA(fvec1),
               <float *> cnp.PyArray_DATA(fvec2),
               <float *> cnp.PyArray_DATA(vec_out))
    return vec_out


cdef inline void csub_3vecs(float *vec1, float *vec2, float *vec_out) nogil:
    cdef int i
    for i from 0<=i<3:
        vec_out[i] = vec1[i]-vec2[i]


def add_3vecs(vec1, vec2):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec1 = as_float_3vec(vec1)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec2 = as_float_3vec(vec2)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] vec_out = np.zeros((3,), np.float32)
    cadd_3vecs(<float *> cnp.PyArray_DATA(fvec1),
               <float *> cnp.PyArray_DATA(fvec2),
               <float *> cnp.PyArray_DATA(vec_out))
    return vec_out


cdef inline void cadd_3vecs(float *vec1, float *vec2, float *vec_out) nogil:
    cdef int i
    for i from 0<=i<3:
        vec_out[i] = vec1[i]+vec2[i]

def mul_3vecs(vec1, vec2):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec1 = as_float_3vec(vec1)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec2 = as_float_3vec(vec2)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] vec_out = np.zeros((3,), np.float32)
    cmul_3vecs(<float *> cnp.PyArray_DATA(fvec1),
               <float *> cnp.PyArray_DATA(fvec2),
               <float *> cnp.PyArray_DATA(vec_out))
    return vec_out

cdef inline void cmul_3vecs(float *vec1, float *vec2, float *vec_out) nogil:
    cdef int i
    for i from 0<=i<3:
        vec_out[i] = vec1[i]*vec2[i]

def mul_3vec(a, vec):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec = as_float_3vec(vec)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] vec_out = np.zeros((3,), np.float32)
    cmul_3vec(a, <float *> cnp.PyArray_DATA(fvec),
              <float *> cnp.PyArray_DATA(vec_out))
    return vec_out

cdef inline void cmul_3vec(float a, float *vec, float *vec_out) nogil:
    cdef int i
    for i from 0<=i<3:
        vec_out[i] = a*vec[i]


# float 32 dtype for casting
cdef cnp.dtype f32_dt = np.dtype(np.float32)


def cut_plane(tracks, ref):
    """ Extract divergence vectors and points of intersection
    between planes normal to the reference fiber and other tracks

    Parameters
    ----------
    tracks : sequence
        of tracks as arrays, shape (N1,3) .. (Nm,3)
    ref : array, shape (N,3)
        reference track

    Returns
    -------
    hits : sequence
       list of points and rcds (radial coefficient of divergence)

    Notes
    -----
    The orthogonality relationship
    ``np.inner(hits[p][q][0:3]-ref[p+1],ref[p+2]-ref[r][p+1])`` will hold
    throughout for every point q in the hits plane at point (p+1) on the
    reference track.

    Examples
    --------
    >>> refx = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]],dtype='float32')
    >>> bundlex = [np.array([[0.5,1,0],[1.5,2,0],[2.5,3,0]],dtype='float32')]
    >>> res = cut_plane(bundlex,refx)
    >>> len(res)
    2
    >>> print(res[0])
    [[ 1.          1.5         0.          0.70710683  0.        ]]
    >>> print(res[1])
    [[ 2.          2.5         0.          0.70710677  0.        ]]
    """
    cdef:
        cnp.npy_intp n_hits, hit_no, max_hit_len
        float alpha,beta,lrq,rcd,lhp,ld
        cnp.ndarray[cnp.float32_t, ndim=2] ref32
        cnp.ndarray[cnp.float32_t, ndim=2] track
        object hits
        cnp.ndarray[cnp.float32_t, ndim=1] one_hit
        float *hit_ptr
        cnp.ndarray[cnp.float32_t, ndim=2] hit_arr
        object Hit=[]
    # make reference fiber usable type
    ref32 = np.ascontiguousarray(ref, f32_dt)
    # convert all the tracks to something we can work with.  Get track
    # lengths
    cdef:
        cnp.npy_intp N_tracks=len(tracks)
        cnp.ndarray[cnp.uint64_t, ndim=1] track_lengths
        cnp.npy_intp t_no, N_track
    cdef object tracks32 = []
    track_lengths = np.empty((N_tracks,), dtype=np.uint64)
    for t_no in range(N_tracks):
        track = np.ascontiguousarray(tracks[t_no], f32_dt)
        track_lengths[t_no] = cnp.PyArray_DIM(track, 0)
        tracks32.append(track)
    # set up loop across reference fiber points
    cdef:
        cnp.npy_intp N_ref = cnp.PyArray_DIM(ref32, 0)
        cnp.npy_intp p_no, q_no
        float *this_ref_p
        float *next_ref_p
        float *this_trk_p
        float *next_trk_p
        float along[3]
        float normal[3]
        float qMp[3]
        float rMp[3]
        float rMq[3]
        float pMq[3]
        float hit[3]
        float hitMp[3]
        float *delta
    normal[:] = [0, 0, 0]
    # List used for storage of hits.  We will fill this with lots of
    # small numpy arrays, and reuse them over the reference track point
    # loops.
    max_hit_len = 0
    hits = []
    # for every point along the reference track
    next_ref_p = asfp(ref32[0])
    for p_no in range(N_ref-1):
        # extract point to point vector into `along`
        this_ref_p = next_ref_p
        next_ref_p = asfp(ref32[p_no+1])
        csub_3vecs(next_ref_p, this_ref_p, along)
        # normalize
        cnormalized_3vec(along, normal)
        # initialize index for hits
        hit_no = 0
        # for every track
        for t_no in range(N_tracks):
            track=tracks32[t_no]
            N_track = track_lengths[t_no]
            # for every point on the track
            next_trk_p = asfp(track[0])
            for q_no in range(N_track-1):
                # p = ref32[p_no]
                # q = track[q_no]
                # r = track[q_no+1]
                # float* versions of above: p == this_ref_p
                this_trk_p = next_trk_p # q
                next_trk_p = asfp(track[q_no+1]) # r
                #if np.inner(normal,q-p)*np.inner(normal,r-p) <= 0:
                csub_3vecs(this_trk_p, this_ref_p, qMp) # q-p
                csub_3vecs(next_trk_p, this_ref_p, rMp) # r-p
                if (cinner_3vecs(normal, qMp) * cinner_3vecs(normal, rMp)) <=0:
                    #if np.inner((r-q),normal) != 0:
                    csub_3vecs(next_trk_p, this_trk_p, rMq)
                    beta = cinner_3vecs(rMq, normal)
                    if beta !=0:
                        #alpha = np.inner((p-q),normal)/np.inner((r-q),normal)
                        csub_3vecs(this_ref_p, this_trk_p, pMq)
                        alpha = (cinner_3vecs(pMq, normal) /
                                  cinner_3vecs(rMq, normal))
                        if alpha < 1:
                            # hit = q+alpha*(r-q)
                            hit[0] = this_trk_p[0]+alpha*rMq[0]
                            hit[1] = this_trk_p[1]+alpha*rMq[1]
                            hit[2] = this_trk_p[2]+alpha*rMq[2]
                            # h-p
                            csub_3vecs(hit, this_ref_p, hitMp)
                            # |h-p|
                            lhp = cnorm_3vec(hitMp)
                            delta = rMq # just renaming
                            # |r-q| == |delta|
                            ld = cnorm_3vec(delta)
                            """ # Summary of stuff in comments
                            # divergence =((r-q)-inner(r-q,normal)*normal)/|r-q|
                            div[0] = (rMq[0]-beta*normal[0]) / ld
                            div[1] = (rMq[1]-beta*normal[1]) / ld
                            div[2] = (rMq[2]-beta*normal[2]) / ld
                            # radial coefficient of divergence d.(h-p)/|h-p|
                            """
                            # radial divergence
                            # np.inner(delta, (hit-p)) / (ld * lhp)
                            if lhp > 0:
                                rcd = fabs(cinner_3vecs(delta, hitMp)
                                           / (ld*lhp))
                            else:
                                rcd=0
                            # hit data into array
                            if hit_no >= max_hit_len:
                                one_hit = np.empty((5,), dtype=f32_dt)
                                hits.append(one_hit)
                            else:
                                one_hit = hits[hit_no]
                            hit_ptr = <float *> cnp.PyArray_DATA(one_hit)
                            hit_ptr[0] = hit[0]
                            hit_ptr[1] = hit[1]
                            hit_ptr[2] = hit[2]
                            hit_ptr[3] = rcd
                            hit_ptr[4] = t_no
                            hit_no += 1
        # convert hits list to hits array
        n_hits = hit_no
        if n_hits > max_hit_len:
            max_hit_len = n_hits
        hit_arr = np.empty((n_hits,5), dtype=f32_dt)
        for hit_no in range(n_hits):
            hit_arr[hit_no] = hits[hit_no]
        Hit.append(hit_arr)
        #Div.append(divs[1:])
    return Hit[1:]





def most_similar_track_mam(tracks,metric='avg'):
    """ Find the most similar track in a bundle
    using distances calculated from Zhang et. al 2008.

    Parameters
    ----------
    tracks : sequence
       of tracks as arrays, shape (N1,3) .. (Nm,3)
    metric : str
       'avg', 'min', 'max'

    Returns
    -------
    si : int
       index of the most similar track in tracks. This can be used as a
       reference track for a bundle.
    s : array, shape (len(tracks),)
        similarities between tracks[si] and the rest of the tracks in
        the bundle

    Notes
    -----
    A vague description of this function is given below:

    for (i,j) in tracks_combinations_of_2:

        calculate the mean_closest_distance from i to j  (mcd_i)
        calculate the mean_closest_distance from j to i  (mcd_j)

        if 'avg':
            s holds the average similarities
        if 'min':
            s holds the minimum similarities
        if 'max':
            s holds the maximum similarities

    si holds the index of the track with min {avg,min,max} average metric
    """
    cdef:
        cnp.npy_intp i, j, lent
        int metric_type
    if metric=='avg':
        metric_type = 0
    elif metric == 'min':
        metric_type = 1
    elif metric == 'max':
        metric_type = 2
    else:
        raise ValueError('Metric should be one of avg, min, max')
    # preprocess tracks
    cdef:
        cnp.npy_intp longest_track_len = 0, track_len
        cnp.ndarray[object, ndim=1] tracks32
    lent = len(tracks)
    tracks32 = np.zeros((lent,), dtype=object)
    # process tracks to predictable memory layout, find longest track
    for i in range(lent):
        tracks32[i] = np.ascontiguousarray(tracks[i], dtype=f32_dt)
        track_len = tracks32[i].shape[0]
        if track_len > longest_track_len:
            longest_track_len = track_len
    # buffer for distances of found track to other tracks
    cdef:
        cnp.ndarray[cnp.double_t, ndim=1] track2others
    track2others = np.zeros((lent,), dtype=np.double)
    # use this buffer also for working space containing summed distances
    # of candidate track to all other tracks
    cdef cnp.double_t *sum_track2others = <cnp.double_t *> cnp.PyArray_DATA(track2others)
    # preallocate buffer array for track distance calculations
    cdef:
        cnp.ndarray [cnp.float32_t, ndim=1] distances_buffer
        cnp.float32_t *t1_ptr
        cnp.float32_t *t2_ptr
        cnp.float32_t *min_buffer
        cnp.float32_t distance
    distances_buffer = np.zeros((longest_track_len*2,), dtype=np.float32)
    min_buffer = <cnp.float32_t *> cnp.PyArray_DATA(distances_buffer)
    # cycle over tracks
    cdef:
        cnp.ndarray [cnp.float32_t, ndim=2] t1, t2
        cnp.npy_intp t1_len, t2_len
    for i from 0 <= i < lent-1:
        t1 = tracks32[i]
        t1_len = cnp.PyArray_DIM(t1, 0)
        t1_ptr = <cnp.float32_t *> cnp.PyArray_DATA(t1)
        for j from i+1 <= j < lent:
            t2 = tracks32[j]
            t2_len = cnp.PyArray_DIM(t2, 0)
            t2_ptr = <cnp.float32_t *> cnp.PyArray_DATA(t2)
            distance = czhang(t1_len, t1_ptr, t2_len, t2_ptr, min_buffer, metric_type)
            # get metric
            sum_track2others[i]+=distance
            sum_track2others[j]+=distance
    # find track with smallest summed metric with other tracks
    cdef double mn = sum_track2others[0]
    cdef cnp.npy_intp si = 0
    for i in range(lent):
        if sum_track2others[i] < mn:
            si = i
            mn = sum_track2others[i]
    # recalculate distance of this track from the others
    t1 = tracks32[si]
    t1_len = cnp.PyArray_DIM(t1, 0)
    t1_ptr = <cnp.float32_t *> cnp.PyArray_DATA(t1)
    for j from 0 <= j < lent:
        t2 = tracks32[j]
        t2_len = cnp.PyArray_DIM(t2, 0)
        t2_ptr = <cnp.float32_t *> cnp.PyArray_DATA(t2)
        track2others[j] = czhang(t1_len, t1_ptr, t2_len, t2_ptr, min_buffer, metric_type)
    return si, track2others

@cython.boundscheck(False)
@cython.wraparound(False)
def bundles_distances_mam(tracksA, tracksB, metric='avg'):
    """ Calculate distances between list of tracks A and list of tracks B

    Parameters
    ----------
    tracksA : sequence
       of tracks as arrays, shape (N1,3) .. (Nm,3)
    tracksB : sequence
       of tracks as arrays, shape (N1,3) .. (Nm,3)
    metric : str
       'avg', 'min', 'max'

    Returns
    -------
    DM : array, shape (len(tracksA), len(tracksB))
        distances between tracksA and tracksB according to metric

    See Also
    --------
    dipy.tracking.streamline.set_number_of_points

    """
    cdef:
        cnp.npy_intp i, j, lentA, lentB
        int metric_type
    if metric=='avg':
        metric_type = 0
    elif metric == 'min':
        metric_type = 1
    elif metric == 'max':
        metric_type = 2
    else:
        raise ValueError('Metric should be one of avg, min, max')
    # preprocess tracks
    cdef:
        cnp.npy_intp longest_track_len = 0, track_len
        cnp.npy_intp longest_track_lenA = 0, longest_track_lenB = 0
        cnp.ndarray[object, ndim=1] tracksA32
        cnp.ndarray[object, ndim=1] tracksB32
        cnp.ndarray[cnp.double_t, ndim=2] DM
    lentA = len(tracksA)
    lentB = len(tracksB)
    # for performance issue, we just check the first streamline
    if len(tracksA[0]) != len(tracksB[0]):
        w_s = "Streamlines do not have the same number of points. "
        w_s += "All streamlines need to have the same number of points. "
        w_s += "Use dipy.tracking.streamline.set_number_of_points to adjust "
        w_s += "your streamlines"
        warn(w_s)
    tracksA32 = np.zeros((lentA,), dtype=object)
    tracksB32 = np.zeros((lentB,), dtype=object)
    DM = np.zeros((lentA,lentB), dtype=np.double)
    # process tracks to predictable memory layout, find longest track
    for i in range(lentA):
        tracksA32[i] = np.ascontiguousarray(tracksA[i], dtype=f32_dt)
        track_len = tracksA32[i].shape[0]
        if track_len > longest_track_lenA:
            longest_track_lenA = track_len
    for i in range(lentB):
        tracksB32[i] = np.ascontiguousarray(tracksB[i], dtype=f32_dt)
        track_len = tracksB32[i].shape[0]
        if track_len > longest_track_lenB:
            longest_track_lenB = track_len
    if longest_track_lenB > longest_track_lenA:
        longest_track_lenA = longest_track_lenB
    # preallocate buffer array for track distance calculations
    cdef:
        cnp.ndarray [cnp.float32_t, ndim=1] distances_buffer
        cnp.float32_t *t1_ptr
        cnp.float32_t *t2_ptr
        cnp.float32_t *min_buffer
    distances_buffer = np.zeros((longest_track_lenA*2,), dtype=np.float32)
    min_buffer = <cnp.float32_t *> cnp.PyArray_DATA(distances_buffer)
    # cycle over tracks
    cdef:
        cnp.ndarray [cnp.float32_t, ndim=2] t1, t2
        cnp.npy_intp t1_len, t2_len
    for i from 0 <= i < lentA:
        t1 = tracksA32[i]
        t1_len = cnp.PyArray_DIM(t1, 0)
        t1_ptr = <cnp.float32_t *> cnp.PyArray_DATA(t1)
        for j from 0 <= j < lentB:
            t2 = tracksB32[j]
            t2_len = cnp.PyArray_DIM(t2, 0)
            t2_ptr = <cnp.float32_t *> cnp.PyArray_DATA(t2)
            DM[i,j] = czhang(t1_len, t1_ptr, t2_len, t2_ptr, min_buffer, metric_type)

    return DM


@cython.boundscheck(False)
@cython.wraparound(False)
def bundles_distances_mdf(tracksA, tracksB):
    """ Calculate distances between list of tracks A and list of tracks B

    All tracks need to have the same number of points

    Parameters
    ----------
    tracksA : sequence
       of tracks as arrays, [(N,3) .. (N,3)]
    tracksB : sequence
       of tracks as arrays, [(N,3) .. (N,3)]

    Returns
    -------
    DM : array, shape (len(tracksA), len(tracksB))
        distances between tracksA and tracksB according to metric

    See Also
    --------
    dipy.tracking.streamline.set_number_of_points

    """
    cdef:
        cnp.npy_intp i, j, lentA, lentB
    # preprocess tracks
    cdef:
        cnp.npy_intp longest_track_len = 0, track_len
        longest_track_lenA, longest_track_lenB
        cnp.ndarray[object, ndim=1] tracksA32
        cnp.ndarray[object, ndim=1] tracksB32
        cnp.ndarray[cnp.double_t, ndim=2] DM
    lentA = len(tracksA)
    lentB = len(tracksB)
    # for performance issue, we just check the first streamline
    if len(tracksA[0]) != len(tracksB[0]):
        w_s = "Streamlines do not have the same number of points. "
        w_s += "All streamlines need to have the same number of points. "
        w_s += "Use dipy.tracking.streamline.set_number_of_points to adjust "
        w_s += "your streamlines"
        warn(w_s)
    tracksA32 = np.zeros((lentA,), dtype=object)
    tracksB32 = np.zeros((lentB,), dtype=object)
    DM = np.zeros((lentA,lentB), dtype=np.double)
    # process tracks to predictable memory layout
    for i in range(lentA):
        tracksA32[i] = np.ascontiguousarray(tracksA[i], dtype=f32_dt)
    for i in range(lentB):
        tracksB32[i] = np.ascontiguousarray(tracksB[i], dtype=f32_dt)
    # preallocate buffer array for track distance calculations
    cdef:
        cnp.float32_t *t1_ptr
        cnp.float32_t *t2_ptr
        cnp.float32_t *min_buffer
    # cycle over tracks
    cdef:
        cnp.ndarray [cnp.float32_t, ndim=2] t1, t2
        cnp.npy_intp t1_len, t2_len
        float d[2]
    t_len = tracksA32[0].shape[0]

    for i from 0 <= i < lentA:
        t1 = tracksA32[i]
        #t1_len = t1.shape[0]
        t1_ptr = <cnp.float32_t *> cnp.PyArray_DATA(t1)
        for j from 0 <= j < lentB:
            t2 = tracksB32[j]
            #t2_len = t2.shape[0]
            t2_ptr = <cnp.float32_t *> cnp.PyArray_DATA(t2)
            #DM[i,j] = czhang(t1_len, t1_ptr, t2_len, t2_ptr, min_buffer, metric_type)
            track_direct_flip_dist(t1_ptr, t2_ptr,t_len,<float *>d)
            if d[0]<d[1]:
                DM[i,j]=d[0]
            else:
                DM[i,j]=d[1]
    return DM




cdef cnp.float32_t inf = np.inf

@cython.cdivision(True)
cdef inline cnp.float32_t czhang(cnp.npy_intp t1_len,
                                 cnp.float32_t *track1_ptr,
                                 cnp.npy_intp t2_len,
                                 cnp.float32_t *track2_ptr,
                                 cnp.float32_t *min_buffer,
                                 int metric_type) nogil:
    """ Note ``nogil`` - no python calls allowed in this function """
    cdef:
        cnp.float32_t *min_t2t1
        cnp.float32_t *min_t1t2
    min_t2t1 = min_buffer
    min_t1t2 = min_buffer + t2_len
    min_distances(t1_len, track1_ptr,
                  t2_len, track2_ptr,
                  min_t2t1,
                  min_t1t2)
    cdef:
        cnp.npy_intp t1_pi, t2_pi
        cnp.float32_t mean_t2t1 = 0, mean_t1t2 = 0, dist_val = 0
    for t1_pi from 0<= t1_pi < t1_len:
        mean_t1t2+=min_t1t2[t1_pi]
    mean_t1t2=mean_t1t2/t1_len
    for t2_pi from 0<= t2_pi < t2_len:
        mean_t2t1+=min_t2t1[t2_pi]
    mean_t2t1=mean_t2t1/t2_len
    if metric_type == 0:
        dist_val=(mean_t2t1+mean_t1t2)/2.0
    elif metric_type == 1:
        if mean_t2t1 < mean_t1t2:
            dist_val=mean_t2t1
        else:
            dist_val=mean_t1t2
    elif metric_type == 2:
        if mean_t2t1 > mean_t1t2:
            dist_val=mean_t2t1
        else:
            dist_val=mean_t1t2
    return dist_val

@cython.cdivision(True)
cdef inline void min_distances(cnp.npy_intp t1_len,
                               cnp.float32_t *track1_ptr,
                               cnp.npy_intp t2_len,
                               cnp.float32_t *track2_ptr,
                               cnp.float32_t *min_t2t1,
                               cnp.float32_t *min_t1t2) nogil:
    cdef:
        cnp.float32_t *t1_pt
        cnp.float32_t *t2_pt
        cnp.float32_t d0, d1, d2
        cnp.float32_t delta2
        cnp.npy_intp t1_pi, t2_pi
    for t2_pi from 0<= t2_pi < t2_len:
        min_t2t1[t2_pi] = inf
    for t1_pi from 0<= t1_pi < t1_len:
        min_t1t2[t1_pi] = inf
    # pointer to current point in track 1
    t1_pt = track1_ptr
    # calculate min squared distance between each point in the two
    # lines.  Squared distance to delay doing the sqrt until after this
    # speed-critical loop
    for t1_pi from 0<= t1_pi < t1_len:
        # pointer to current point in track 2
        t2_pt = track2_ptr
        for t2_pi from 0<= t2_pi < t2_len:
            d0 = t1_pt[0] - t2_pt[0]
            d1 = t1_pt[1] - t2_pt[1]
            d2 = t1_pt[2] - t2_pt[2]
            delta2 = d0*d0 + d1*d1 + d2*d2
            if delta2 < min_t2t1[t2_pi]:
                min_t2t1[t2_pi]=delta2
            if delta2 < min_t1t2[t1_pi]:
                min_t1t2[t1_pi]=delta2
            t2_pt += 3 # to next point in track 2
        t1_pt += 3 # to next point in track 1
    # sqrt to get Euclidean distance from squared distance
    for t1_pi from 0<= t1_pi < t1_len:
        min_t1t2[t1_pi]=sqrt(min_t1t2[t1_pi])
    for t2_pi from 0<= t2_pi < t2_len:
        min_t2t1[t2_pi]=sqrt(min_t2t1[t2_pi])



def mam_distances(xyz1,xyz2,metric='all'):
    """ Min/Max/Mean Average Minimum Distance between tracks xyz1 and xyz2

    Based on the metrics in Zhang, Correia, Laidlaw 2008
    http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=4479455
    which in turn are based on those of Corouge et al. 2004

    Parameters
    ----------
    xyz1 : array, shape (N1,3), dtype float32
    xyz2 : array, shape (N2,3), dtype float32
       arrays representing x,y,z of the N1 and N2 points of two tracks
    metrics : {'avg','min','max','all'}
       Metric to calculate.  {'avg','min','max'} return a scalar. 'all'
       returns a tuple

    Returns
    -------
    avg_mcd : float
       average_mean_closest_distance
    min_mcd : float
       minimum_mean_closest_distance
    max_mcd : float
       maximum_mean_closest_distance

    Notes
    -----
    Algorithmic description

    Let's say we have curves A and B.

    For every point in A calculate the minimum distance from every point
    in B stored in minAB

    For every point in B calculate the minimum distance from every point
    in A stored in minBA

    find average of minAB stored as avg_minAB
    find average of minBA stored as avg_minBA

    if metric is 'avg' then return (avg_minAB + avg_minBA)/2.0
    if metric is 'min' then return min(avg_minAB,avg_minBA)
    if metric is 'max' then return max(avg_minAB,avg_minBA)
    """
    cdef:
        cnp.ndarray[cnp.float32_t, ndim=2] track1
        cnp.ndarray[cnp.float32_t, ndim=2] track2
        cnp.npy_intp t1_len, t2_len
    track1 = np.ascontiguousarray(xyz1, dtype=f32_dt)
    t1_len = cnp.PyArray_DIM(track1, 0)
    track2 = np.ascontiguousarray(xyz2, dtype=f32_dt)
    t2_len = cnp.PyArray_DIM(track2, 0)
    # preallocate buffer array for track distance calculations
    cdef:
        cnp.float32_t *min_t2t1
        cnp.float32_t *min_t1t2
        cnp.ndarray [cnp.float32_t, ndim=1] distances_buffer
    distances_buffer = np.zeros((t1_len + t2_len,), dtype=np.float32)
    min_t2t1 = <cnp.float32_t *> cnp.PyArray_DATA(distances_buffer)
    min_t1t2 = min_t2t1 + t2_len
    min_distances(t1_len, <cnp.float32_t *> cnp.PyArray_DATA(track1),
                  t2_len, <cnp.float32_t *> cnp.PyArray_DATA(track2),
                  min_t2t1,
                  min_t1t2)
    cdef:
        cnp.npy_intp t1_pi, t2_pi
        cnp.float32_t mean_t2t1 = 0, mean_t1t2 = 0
    for t1_pi from 0<= t1_pi < t1_len:
        mean_t1t2+=min_t1t2[t1_pi]
    mean_t1t2=mean_t1t2/t1_len
    for t2_pi from 0<= t2_pi < t2_len:
        mean_t2t1+=min_t2t1[t2_pi]
    mean_t2t1=mean_t2t1/t2_len
    if metric=='all':
        return ((mean_t2t1+mean_t1t2)/2.0,
                np.min((mean_t2t1,mean_t1t2)),
                np.max((mean_t2t1,mean_t1t2)))
    elif metric=='avg':
        return (mean_t2t1+mean_t1t2)/2.0
    elif metric=='min':
        return np.min((mean_t2t1,mean_t1t2))
    elif metric =='max':
        return np.max((mean_t2t1,mean_t1t2))
    else :
        ValueError('Wrong argument for metric')


def minimum_closest_distance(xyz1,xyz2):
    """ Find the minimum distance between two curves xyz1, xyz2

    Parameters
    ----------
    xyz1 : array, shape (N1,3), dtype float32
    xyz2 : array, shape (N2,3), dtype float32
        arrays representing x,y,z of the N1 and N2 points  of two tracks

    Returns
    -------
    md : minimum distance

    Notes
    -----
    Algorithmic description

    Let's say we have curves A and B

    for every point in A calculate the minimum distance from every point in B stored in minAB
    for every point in B calculate the minimum distance from every point in A stored in minBA
    find min of minAB stored in min_minAB
    find min of minBA stored in min_minBA

    Then return (min_minAB + min_minBA)/2.0
    """
    cdef:
        cnp.ndarray[cnp.float32_t, ndim=2] track1
        cnp.ndarray[cnp.float32_t, ndim=2] track2
        cnp.npy_intp t1_len, t2_len
    track1 = np.ascontiguousarray(xyz1, dtype=f32_dt)
    t1_len = cnp.PyArray_DIM(track1, 0)
    track2 = np.ascontiguousarray(xyz2, dtype=f32_dt)
    t2_len = cnp.PyArray_DIM(track2, 0)
    # preallocate buffer array for track distance calculations
    cdef:
        cnp.float32_t *min_t2t1
        cnp.float32_t *min_t1t2
        cnp.ndarray [cnp.float32_t, ndim=1] distances_buffer
    distances_buffer = np.zeros((t1_len + t2_len,), dtype=np.float32)
    min_t2t1 = <cnp.float32_t *> cnp.PyArray_DATA(distances_buffer)
    min_t1t2 = min_t2t1 + t2_len
    min_distances(t1_len, <cnp.float32_t *> cnp.PyArray_DATA(track1),
                  t2_len, <cnp.float32_t *> cnp.PyArray_DATA(track2),
                  min_t2t1,
                  min_t1t2)
    cdef:
        cnp.npy_intp t1_pi, t2_pi
        double min_min_t2t1 = inf
        double min_min_t1t2 = inf
    for t1_pi in range(t1_len):
        if min_min_t1t2 > min_t1t2[t1_pi]:
            min_min_t1t2 = min_t1t2[t1_pi]
    for t2_pi in range(t2_len):
        if min_min_t2t1 > min_t2t1[t2_pi]:
            min_min_t2t1 = min_t2t1[t2_pi]
    return (min_min_t1t2+min_min_t2t1)/2.0


def lee_perpendicular_distance(start0, end0, start1, end1):
    """ Calculates perpendicular distance metric for the distance between two line segments

    Based on Lee , Han & Whang SIGMOD07.

    This function assumes that norm(end0-start0)>norm(end1-start1) i.e. that the
    first segment will be bigger than the second one.

    Parameters
    ----------
    start0 : float array(3,)
    end0 : float array(3,)
    start1 : float array(3,)
    end1 : float array(3,)

    Returns
    -------
    perpendicular_distance: float

    Notes
    -----
    l0 = np.inner(end0-start0,end0-start0)
    l1 = np.inner(end1-start1,end1-start1)

    k0=end0-start0

    u1 = np.inner(start1-start0,k0)/l0
    u2 = np.inner(end1-start0,k0)/l0

    ps = start0+u1*k0
    pe = start0+u2*k0

    lperp1 = np.sqrt(np.inner(ps-start1,ps-start1))
    lperp2 = np.sqrt(np.inner(pe-end1,pe-end1))

    if lperp1+lperp2 > 0.:
        return (lperp1**2+lperp2**2)/(lperp1+lperp2)
    else:
        return 0.

    Examples
    --------
    >>> d = lee_perpendicular_distance([0,0,0],[1,0,0],[3,4,5],[5,4,3])
    >>> print('%.6f' % d)
    5.787888
    """

    cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec1,fvec2,fvec3,fvec4

    fvec1 = as_float_3vec(start0)
    fvec2 = as_float_3vec(end0)
    fvec3 = as_float_3vec(start1)
    fvec4 = as_float_3vec(end1)

    return clee_perpendicular_distance(<float *> cnp.PyArray_DATA(fvec1),
                                       <float *> cnp.PyArray_DATA(fvec2),
                                       <float *> cnp.PyArray_DATA(fvec3),
                                       <float *> cnp.PyArray_DATA(fvec4))


cdef float clee_perpendicular_distance(float *start0, float *end0,float *start1, float *end1):
    """ This function assumes that norm(end0-start0)>norm(end1-start1)
    """

    cdef:
        float l0,l1,ltmp,u1,u2,lperp1,lperp2
        float *s_tmp
        float *e_tmp
        float k0[3]
        float ps[3]
        float pe[3]
        float ps1[3]
        float pe1[3]
        float tmp[3]

    csub_3vecs(end0,start0,k0)
    l0 = cinner_3vecs(k0,k0)

    csub_3vecs(end1,start1,tmp)
    l1 = cinner_3vecs(tmp, tmp)


    #csub_3vecs(end0,start0,k0)

    #u1 = np.inner(start1-start0,k0)/l0
    #u2 = np.inner(end1-start0,k0)/l0
    csub_3vecs(start1,start0,tmp)
    u1 = cinner_3vecs(tmp,k0)/l0

    csub_3vecs(end1,start0,tmp)
    u2 = cinner_3vecs(tmp,k0)/l0

    cmul_3vec(u1,k0,tmp)
    cadd_3vecs(start0,tmp,ps)

    cmul_3vec(u2,k0,tmp)
    cadd_3vecs(start0,tmp,pe)

    #lperp1 = np.sqrt(np.inner(ps-start1,ps-start1))
    #lperp2 = np.sqrt(np.inner(pe-end1,pe-end1))

    csub_3vecs(ps,start1,ps1)
    csub_3vecs(pe,end1,pe1)

    lperp1 = sqrt(cinner_3vecs(ps1,ps1))
    lperp2 = sqrt(cinner_3vecs(pe1,pe1))

    if lperp1+lperp2 > 0.:
        return (lperp1*lperp1+lperp2*lperp2)/(lperp1+lperp2)
    else:
        return 0.


def lee_angle_distance(start0, end0, start1, end1):
    """ Calculates angle distance metric for the distance between two line segments

    Based on Lee , Han & Whang SIGMOD07.

    This function assumes that norm(end0-start0)>norm(end1-start1) i.e. that the
    first segment will be bigger than the second one.

    Parameters
    ----------
    start0 : float array(3,)
    end0 : float array(3,)
    start1 : float array(3,)
    end1 : float array(3,)

    Returns
    -------
    angle_distance : float

    Notes
    -----
    l_0 = np.inner(end0-start0,end0-start0)
    l_1 = np.inner(end1-start1,end1-start1)

    cos_theta_squared = np.inner(end0-start0,end1-start1)**2/ (l_0*l_1)
    return np.sqrt((1-cos_theta_squared)*l_1)

    Examples
    --------
    >>> lee_angle_distance([0,0,0],[1,0,0],[3,4,5],[5,4,3])
    2.0
    """

    cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec1,fvec2,fvec3,fvec4

    fvec1 = as_float_3vec(start0)
    fvec2 = as_float_3vec(end0)
    fvec3 = as_float_3vec(start1)
    fvec4 = as_float_3vec(end1)

    return clee_angle_distance(<float *> cnp.PyArray_DATA(fvec1),
                               <float *> cnp.PyArray_DATA(fvec2),
                               <float *> cnp.PyArray_DATA(fvec3),
                               <float *> cnp.PyArray_DATA(fvec4))


cdef float clee_angle_distance(float *start0, float *end0,float *start1, float *end1):
    """ This function assumes that norm(end0-start0)>norm(end1-start1)
    """

    cdef:
        float l0,l1,ltmp,cos_theta_squared
        float *s_tmp
        float *e_tmp
        float k0[3]
        float k1[3]
        float tmp[3]

    csub_3vecs(end0,start0,k0)
    l0 = cinner_3vecs(k0,k0)
    #print l0

    csub_3vecs(end1,start1,k1)
    l1 = cinner_3vecs(k1, k1)
    #print l1

    ltmp=cinner_3vecs(k0,k1)

    cos_theta_squared = (ltmp*ltmp)/ (l0*l1)
    #print cos_theta_squared
    return sqrt((1-cos_theta_squared)*l1)


def approx_polygon_track(xyz,alpha=0.392):
    """ Fast and simple trajectory approximation algorithm by Eleftherios and Ian

    It will reduce the number of points of the track by keeping
    intact the start and endpoints of the track and trying to remove
    as many points as possible without distorting much the shape of
    the track

    Parameters
    ----------
    xyz : array(N,3)
        initial trajectory
    alpha : float
        smoothing parameter (<0.392 smoother, >0.392 rougher) if the
        trajectory was a smooth circle then with alpha =0.393
        ~=pi/8. the circle would be approximated with an decahexagon if
        alpha = 0.7853 ~=pi/4. with an octagon.

    Returns
    -------
    characteristic_points: list of M array(3,) points

    Examples
    --------

    Approximating a helix:

    >>> t=np.linspace(0,1.75*2*np.pi,100)
    >>> x = np.sin(t)
    >>> y = np.cos(t)
    >>> z = t
    >>> xyz=np.vstack((x,y,z)).T
    >>> xyza = approx_polygon_track(xyz)
    >>> len(xyza) < len(xyz)
    True

    Notes
    -----
    Assuming that a good approximation for a circle is an octagon then
    that means that the points of the octagon will have angle alpha =
    2*pi/8 = pi/4 . We calculate the angle between every two neighbour
    segments of a trajectory and if the angle is higher than pi/4 we
    choose that point as a characteristic point otherwise we move at the
    next point.
    """
    cdef :
        cnp.npy_intp mid_index
        cnp.ndarray[cnp.float32_t, ndim=2] track
        float *fvec0
        float *fvec1
        float *fvec2
        object characteristic_points
        cnp.npy_intp t_len
        double angle,tmp, denom
        float vec0[3]
        float vec1[3]

    angle=alpha
    track = np.ascontiguousarray(xyz, dtype=f32_dt)
    t_len=len(track)
    characteristic_points=[track[0]]
    mid_index = 1
    angle=0

    while mid_index < t_len-1:
        #fvec0 = as_float_3vec(track[mid_index-1])
        #<float *> cnp.PyArray_DATA(track[0])
        fvec0 = asfp(track[mid_index-1])
        fvec1 = asfp(track[mid_index])
        fvec2 = asfp(track[mid_index+1])
        #csub_3vecs(<float *> cnp.PyArray_DATA(fvec1),<float *> cnp.PyArray_DATA(fvec0),vec0)
        csub_3vecs(fvec1,fvec0,vec0)
        csub_3vecs(fvec2,fvec1,vec1)
        denom = cnorm_3vec(vec0)*cnorm_3vec(vec1)
        tmp=<double>fabs(acos(cinner_3vecs(vec0,vec1)/ denom)) if denom else 0
        if dpy_isnan(tmp) :
            angle+=0.
        else:
            angle+=tmp
        if  angle > alpha:
            characteristic_points.append(track[mid_index])
            angle=0
        mid_index+=1

    characteristic_points.append(track[-1])
    return np.array(characteristic_points)


def approximate_mdl_trajectory(xyz, alpha=1.):
    """ Implementation of Lee et al Approximate Trajectory
    Partitioning Algorithm

    This is base on the minimum description length principle

    Parameters
    ----------
    xyz : array(N,3)
        initial trajectory
    alpha : float
        smoothing parameter (>1 smoother, <1  rougher)

    Returns
    -------
    characteristic_points : list of M array(3,) points

    """
    cdef :
        int start_index,length,current_index, i
        double cost_par,cost_nopar,alphac
        object characteristic_points
        cnp.npy_intp t_len
        cnp.ndarray[cnp.float32_t, ndim=2] track
        float tmp[3]
        cnp.ndarray[cnp.float32_t, ndim=1] fvec1,fvec2,fvec3,fvec4

    track = np.ascontiguousarray(xyz, dtype=f32_dt)
    t_len=len(track)

    alphac=alpha
    characteristic_points=[xyz[0]]
    start_index = 0
    length = 2
    #print t_len

    while start_index+length < <int>t_len-1:
        current_index = start_index+length
        fvec1 = as_float_3vec(track[start_index])
        fvec2 = as_float_3vec(track[current_index])
        # L(H)
        csub_3vecs(<float *> cnp.PyArray_DATA(fvec2),
                   <float *> cnp.PyArray_DATA(fvec1), tmp)
        cost_par=dpy_log2(sqrt(cinner_3vecs(tmp,tmp)))
        cost_nopar=0
        #print start_index,current_index
        # L(D|H)
        #for i in range(start_index+1,current_index):#+1):
        for i in range(start_index,current_index+1):
            #print i
            fvec3 = as_float_3vec(track[i])
            fvec4 = as_float_3vec(track[i+1])
            cost_par += dpy_log2(clee_perpendicular_distance(<float *> cnp.PyArray_DATA(fvec3),
                                                             <float *> cnp.PyArray_DATA(fvec4),
                                                             <float *> cnp.PyArray_DATA(fvec1),
                                                             <float *> cnp.PyArray_DATA(fvec2)))
            cost_par += dpy_log2(clee_angle_distance(<float *> cnp.PyArray_DATA(fvec3),
                                                     <float *> cnp.PyArray_DATA(fvec4),
                                                     <float *> cnp.PyArray_DATA(fvec1),
                                                     <float *> cnp.PyArray_DATA(fvec2)))
            csub_3vecs(<float *> cnp.PyArray_DATA(fvec4),
                       <float *> cnp.PyArray_DATA(fvec3), tmp)
            cost_nopar += dpy_log2(cinner_3vecs(tmp,tmp))
        cost_nopar /= 2
        #print cost_par, cost_nopar, start_index,length
        if alphac*cost_par>cost_nopar:
            characteristic_points.append(track[current_index-1])
            start_index = current_index-1
            length = 2
        else:
            length+=1
    characteristic_points.append(track[-1])
    return np.array(characteristic_points)


def intersect_segment_cylinder(sa,sb,p,q,r):
    """ Intersect Segment S(t) = sa +t(sb-sa), 0 <=t<= 1 against cylinder specified by p,q and r

    See p.197 from Real Time Collision Detection by C. Ericson

    Examples
    --------
    Define cylinder using a segment defined by

    >>> p=np.array([0,0,0],dtype=np.float32)
    >>> q=np.array([1,0,0],dtype=np.float32)
    >>> r=0.5

    Define segment

    >>> sa=np.array([0.5,1 ,0],dtype=np.float32)
    >>> sb=np.array([0.5,-1,0],dtype=np.float32)

    Intersection

    >>> intersect_segment_cylinder(sa, sb, p, q, r)
    (1.0, 0.25, 0.75)
    """
    cdef:
        float *csa
        float *csb
        float *cp
        float *cq
        float cr
        float ct[2]

    csa = asfp(sa)
    csb = asfp(sb)
    cp = asfp(p)
    cq = asfp(q)
    cr=r
    ct[0]=-100
    ct[1]=-100

    tmp = cintersect_segment_cylinder(csa,csb,cp, cq, cr, ct)

    return tmp, ct[0], ct[1]

cdef float cintersect_segment_cylinder(float *sa,float *sb,float *p, float *q, float r, float *t):
    """ Intersect Segment S(t) = sa +t(sb-sa), 0 <=t<= 1 against cylinder specified by p,q and r

    Look p.197 from Real Time Collision Detection C. Ericson

    Returns
    -------
    inter : bool
        0 no intersection
        1 intersection
    """
    cdef:
        float d[3]
        float m[3]
        float n[3]
        float md,nd,dd, nn, mn, a, k, c,b, discr

        float epsilon_float=5.96e-08

    csub_3vecs(q,p,d)
    csub_3vecs(sa,p,m)
    csub_3vecs(sb,sa,n)

    md=cinner_3vecs(m,d)
    nd=cinner_3vecs(n,d)
    dd=cinner_3vecs(d,d)

    #test if segment fully outside either endcap of cylinder
    if md < 0. and md + nd < 0.:  return 0 #segment outside p side
    if md > dd and md + nd > dd:  return 0 #segment outside q side

    nn=cinner_3vecs(n,n)
    mn=cinner_3vecs(m,n)
    a=dd*nn-nd*nd
    k=cinner_3vecs(m,m) -r*r
    c=dd*k-md*md

    if fabs(a) < epsilon_float:
        #segment runs parallel to cylinder axis
        if c>0.:  return 0. # segment lies outside cylinder
        if md < 0.:
            t[0]=-mn/nn # intersect against p endcap
        elif md > dd :
            t[0]=(nd-mn)/nn # intersect against q endcap
        else:
            t[0]=0. # lies inside cylinder
        return 1

    b=dd*mn -nd*md
    discr=b*b-a*c
    if discr < 0.: return 0. # no real roots ; no intersection

    t[0]=(-b-sqrt(discr))/a
    t[1]=(-b+sqrt(discr))/a
    if t[0]<0. or t[0] > 1. : return 0. # intersection lies outside segment

    if md + t[0]* nd < 0.:
        #intersection outside cylinder on 'p' side
        if nd <= 0. : return 0. # segment pointing away from endcap
        t[0]=-md/nd
        #keep intersection if Dot(S(t)-p,S(t)-p) <= r^2
        if k+2*t[0]*(mn+t[0]*nn) <=0.:
            return 1.

    elif md+t[0]*nd > dd :
        #intersection outside cylinder on 'q' side
        if nd >= 0.: return 0. # segment pointing away from endcap
        t[0]= (dd-md)/nd
        #keep intersection if Dot(S(t)-q,S(t)-q) <= r^2
        if k+dd-2*md+t[0]*(2*(mn-nd)+t[0]*nn) <= 0.:
            return 1.
    # segment intersects cylinder between the endcaps; t is correct
    return 1.


def point_segment_sq_distance(a, b, c):
    """ Calculate the squared distance from a point c to a finite line segment ab.

    Examples
    --------
    >>> a=np.array([0,0,0], dtype=np.float32)
    >>> b=np.array([1,0,0], dtype=np.float32)
    >>> c=np.array([0,1,0], dtype=np.float32)
    >>> point_segment_sq_distance(a, b, c)
    1.0
    >>> c = np.array([0,3,0], dtype=np.float32)
    >>> point_segment_sq_distance(a,b,c)
    9.0
    >>> c = np.array([-1,1,0], dtype=np.float32)
    >>> point_segment_sq_distance(a, b, c)
    2.0
    """
    cdef:
        float *ca
        float *cb
        float *cc
        float cr
        float ct[2]

    ca = asfp(a)
    cb = asfp(b)
    cc = asfp(c)

    return cpoint_segment_sq_dist(ca, cb, cc)


@cython.cdivision(True)
cdef inline float cpoint_segment_sq_dist(float * a, float * b, float * c) nogil:
    """ Calculate the squared distance from a point c to a line segment ab.

    """
    cdef:
        float ab[3]
        float ac[3]
        float bc[3]
        float e,f

    csub_3vecs(b,a,ab)
    csub_3vecs(c,a,ac)
    csub_3vecs(c,b,bc)

    e = cinner_3vecs(ac, ab)
    #Handle cases where c projects outside ab
    if e <= 0.:
        return cinner_3vecs(ac, ac)
    f = cinner_3vecs(ab, ab)
    if e >= f :
        return cinner_3vecs(bc, bc)
    #Handle case where c projects onto ab
    return cinner_3vecs(ac, ac) - e * e / f


def track_dist_3pts(tracka,trackb):
    """ Calculate the euclidean distance between two 3pt tracks

    Both direct and flip distances are calculated but only the smallest is returned

    Parameters
    ----------
    a : array, shape (3,3)
    a three point track
    b : array, shape (3,3)
    a three point track

    Returns
    -------
    dist :float

    Examples
    --------
    >>> a = np.array([[0,0,0],[1,0,0,],[2,0,0]])
    >>> b = np.array([[3,0,0],[3.5,1,0],[4,2,0]])
    >>> c = track_dist_3pts(a, b)
    >>> print('%.6f' % c)
    2.721573
    """

    cdef cnp.ndarray[cnp.float32_t, ndim=2] a,b
    cdef float d[2]

    a=np.ascontiguousarray(tracka,dtype=f32_dt)
    b=np.ascontiguousarray(trackb,dtype=f32_dt)

    track_direct_flip_3dist(asfp(a[0]),asfp(a[1]),asfp(a[2]),
                            asfp(b[0]),asfp(b[1]),asfp(b[2]),d)

    if d[0]<d[1]:
        return d[0]
    else:
        return d[1]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void track_direct_flip_dist(float *a,float *b,long rows,float *out) nogil:
    r""" Direct and flip average distance between two tracks

    Parameters
    ----------
    a : first track
    b : second track
    rows : number of points of the track
        both tracks need to have the same number of points

    Returns
    -------
    out : direct and flipped average distance added

    Notes
    -----
    The distance calculated between two tracks::

        t_1       t_2

        0*   a    *0
          \       |
           \      |
           1*     |
            |  b  *1
            |      \
            2*      \
                c    *2

    is equal to $(a+b+c)/3$ where $a$ the euclidean distance between t_1[0] and
    t_2[0], $b$ between t_1[1] and t_2[1] and $c$ between t_1[2] and t_2[2].
    Also the same with t2 flipped (so t_1[0] compared to t_2[2] etc).

    See Also
    --------
    dipy.tracking.distances.local_skeleton_clustering
    """
    cdef:
        cnp.npy_intp i=0
        cnp.npy_intp j=0
        cnp.float32_t sub=0,subf=0, tmprow=0, tmprowf=0
        double distf=0,dist=0

    for i from 0<=i<rows:
        tmprow=0
        tmprowf=0
        for j from 0<=j<3:
            sub=a[i*3+j]-b[i*3+j]
            subf=a[i*3+j]-b[(rows-1-i)*3+j]
            tmprow+=sub*sub
            tmprowf+=subf*subf
        dist+=sqrt(tmprow)
        distf+=sqrt(tmprowf)

    out[0]=<cnp.float32_t>dist/<cnp.float32_t>rows
    out[1]=<cnp.float32_t>distf/<cnp.float32_t>rows


@cython.cdivision(True)
cdef inline void track_direct_flip_3dist(float *a1, float *b1,float  *c1,float *a2, float *b2, float *c2, float *out) nogil:
    """ Calculate the euclidean distance between two 3pt tracks
    both direct and flip are given as output

    Parameters
    ----------
    a1,b1,c1 : 3 float[3] arrays representing the first track
    a2,b2,c2 : 3 float[3] arrays representing the second track

    Returns
    -------
    out : a float[2] array having the euclidean distance and the fliped euclidean distance

    """

    cdef:
        int i
        float tmp1=0,tmp2=0,tmp3=0,tmp1f=0,tmp3f=0

    #for i in range(3):
    for i from 0<=i<3:
        tmp1=tmp1+(a1[i]-a2[i])*(a1[i]-a2[i])
        tmp2=tmp2+(b1[i]-b2[i])*(b1[i]-b2[i])
        tmp3=tmp3+(c1[i]-c2[i])*(c1[i]-c2[i])
        tmp1f=tmp1f+(a1[i]-c2[i])*(a1[i]-c2[i])
        tmp3f=tmp3f+(c1[i]-a2[i])*(c1[i]-a2[i])

    out[0]=(sqrt(tmp1)+sqrt(tmp2)+sqrt(tmp3))/3.0
    out[1]=(sqrt(tmp1f)+sqrt(tmp2)+sqrt(tmp3f))/3.0

    #out[0]=(tmp1+tmp2+tmp3)/3.0
    #out[1]=(tmp1f+tmp2+tmp3f)/3.0




ctypedef struct LSC_Cluster:
    long *indices
    float *hidden
    long N




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def local_skeleton_clustering(tracks, d_thr=10):
    r"""Efficient tractography clustering

    Every track can needs to have the same number of points.
    Use `dipy.tracking.metrics.downsample` to restrict the number of points

    Parameters
    ----------
    tracks : sequence
        of tracks as arrays, shape (N,3) .. (N,3) where N=points
    d_thr : float
        average euclidean distance threshold

    Returns
    -------
    C : dict
        Clusters.

    Examples
    --------
    >>> tracks=[np.array([[0,0,0],[1,0,0,],[2,0,0]]),
    ...         np.array([[3,0,0],[3.5,1,0],[4,2,0]]),
    ...         np.array([[3.2,0,0],[3.7,1,0],[4.4,2,0]]),
    ...         np.array([[3.4,0,0],[3.9,1,0],[4.6,2,0]]),
    ...         np.array([[0,0.2,0],[1,0.2,0],[2,0.2,0]]),
    ...         np.array([[2,0.2,0],[1,0.2,0],[0,0.2,0]]),
    ...         np.array([[0,0,0],[0,1,0],[0,2,0]])]
    >>> C = local_skeleton_clustering(tracks, d_thr=0.5)

    Notes
    -----
    The distance calculated between two tracks::

        t_1       t_2

        0*   a    *0
          \       |
           \      |
           1*     |
            |  b  *1
            |      \
            2*      \
                c    *2

    is equal to $(a+b+c)/3$ where $a$ the euclidean distance between ``t_1[0]``
    and ``t_2[0]``, $b$ between ``t_1[1]`` and ``t_2[1]`` and $c$ between
    ``t_1[2]`` and ``t_2[2]``. Also the same with t2 flipped (so ``t_1[0]``
    compared to ``t_2[2]`` etc).

    Visualization:

    It is possible to visualize the clustering C from the example
    above using the dipy.viz module::

        from dipy.viz import window, actor
        scene = window.Scene()
        for c in C:
            color=np.random.rand(3)
            for i in C[c]['indices']:
                scene.add(actor.line(tracks[i],color))
        window.show(scene)

    See Also
    --------
    dipy.tracking.metrics.downsample
    """
    cdef:
        cnp.ndarray[cnp.float32_t, ndim=2] track
        LSC_Cluster *cluster
        long lent = 0,lenC = 0, dim = 0, points=0
        long i=0, j=0, c=0, i_k=0, rows=0 ,cit=0
        float *ptr
        float *hid
        float *alld
        float d[2]
        float m_d, cd_thr
        long *flip

    points=len(tracks[0])
    dim = points*3
    rows = points
    cd_thr = d_thr

    #Allocate and copy memory for first cluster
    cluster=<LSC_Cluster *>realloc(NULL,sizeof(LSC_Cluster))
    cluster[0].indices=<long *>realloc(NULL,sizeof(long))
    cluster[0].hidden=<float *>realloc(NULL,dim*sizeof(float))
    cluster[0].indices[0]=0
    track=np.ascontiguousarray(tracks[0],dtype=f32_dt)
    ptr=<float *> cnp.PyArray_DATA(track)
    for i from 0<=i<dim:
        cluster[0].hidden[i]=ptr[i]
    cluster[0].N=1

    #holds number of clusters
    lenC = 1

    #store memmory for the hid variable
    hid=<float *>realloc(NULL,dim*sizeof(float))

    #Work with the rest of the tracks
    lent=len(tracks)
    for it in range(1,lent):
        track=np.ascontiguousarray(tracks[it],dtype=f32_dt)
        ptr=<float *> cnp.PyArray_DATA(track)
        cit=it

        with nogil:

            alld=<float *>calloc(lenC,sizeof(float))
            flip=<long *>calloc(lenC,sizeof(long))
            for k from 0<=k<lenC:
                for i from 0<=i<dim:
                    hid[i]=cluster[k].hidden[i]/<float>cluster[k].N

                #track_direct_flip_3dist(&ptr[0],&ptr[3],&ptr[6],&hid[0],&hid[3],&hid[6],d)
                #track_direct_flip_3dist(ptr,ptr+3,ptr+6,hid,hid+3,hid+6,<float *>d)
                track_direct_flip_dist(ptr, hid,rows,<float *>d)

                if d[1]<d[0]:
                    d[0]=d[1]
                    flip[k]=1
                alld[k]=d[0]

            m_d = biggest_float
            #find minimum distance and index
            for k from 0<=k<lenC:
                if alld[k] < m_d:
                    m_d=alld[k]
                    i_k=k

            if m_d < cd_thr:
                if flip[i_k]==1:#correct if flipping is needed
                    for i from 0<=i<rows:
                        for j from 0<=j<3:
                            cluster[i_k].hidden[i*3+j]+=ptr[(rows-1-i)*3+j]
                else:
                     for i from 0<=i<rows:
                        for j from 0<=j<3:
                            cluster[i_k].hidden[i*3+j]+=ptr[i*3+j]
                cluster[i_k].N+=1
                cluster[i_k].indices=<long *>realloc(cluster[i_k].indices,cluster[i_k].N*sizeof(long))
                cluster[i_k].indices[cluster[i_k].N-1]=cit

            else:#New cluster added
                lenC+=1
                cluster=<LSC_Cluster *>realloc(cluster,lenC*sizeof(LSC_Cluster))
                cluster[lenC-1].indices=<long *>realloc(NULL,sizeof(long))
                cluster[lenC-1].hidden=<float *>realloc(NULL,dim*sizeof(float))
                cluster[lenC-1].indices[0]=cit
                for i from 0<=i<dim:
                    cluster[lenC-1].hidden[i]=ptr[i]
                cluster[lenC-1].N=1

            free(alld)
            free(flip)


    #Copy results to a dictionary

    C={}
    for k in range(lenC):

        C[k]={}
        C[k]['hidden']=np.zeros(points*3,dtype=np.float32)

        for j in range(points*3):
            C[k]['hidden'][j]=cluster[k].hidden[j]
        C[k]['hidden'].shape=(points,3)

        C[k]['N']=cluster[k].N
        C[k]['indices']=np.zeros(cluster[k].N,dtype=np.int64)

        for i in range(cluster[k].N):
            C[k]['indices'][i]=cluster[k].indices[i]

        C[k]['indices']=list(C[k]['indices'])

    #Free memory
    with nogil:

        for k from 0<=k<lenC:
            free(cluster[k].indices)
            free(cluster[k].hidden)
        free(cluster)

    return C

def local_skeleton_clustering_3pts(tracks, d_thr=10):
    """ Does a first pass clustering

    Every track can only have 3 pts neither less or more.
    Use `dipy.tracking.metrics.downsample` to restrict the number of points

    Parameters
    ----------
    tracks : sequence
        of tracks as arrays, shape (N,3) .. (N,3) where N=3
    d_thr : float
        Average euclidean distance threshold

    Returns
    -------
    C : dict
       Clusters.

    Examples
    --------
    >>> tracks=[np.array([[0,0,0],[1,0,0,],[2,0,0]]),
    ...         np.array([[3,0,0],[3.5,1,0],[4,2,0]]),
    ...         np.array([[3.2,0,0],[3.7,1,0],[4.4,2,0]]),
    ...         np.array([[3.4,0,0],[3.9,1,0],[4.6,2,0]]),
    ...         np.array([[0,0.2,0],[1,0.2,0],[2,0.2,0]]),
    ...         np.array([[2,0.2,0],[1,0.2,0],[0,0.2,0]]),
    ...         np.array([[0,0,0],[0,1,0],[0,2,0]])]
    >>> C=local_skeleton_clustering_3pts(tracks,d_thr=0.5)

    Notes
    -----
    It is possible to visualize the clustering C from the example
    above using the fvtk module::

        r=fvtk.ren()
        for c in C:
            color=np.random.rand(3)
            for i in C[c]['indices']:
                fvtk.add(r,fos.line(tracks[i],color))
        fvtk.show(r)

    """
    cdef :
        cnp.ndarray[cnp.float32_t, ndim=2] track
        cnp.ndarray[cnp.float32_t, ndim=2] h
        int lent,k,it
        float d[2]
        #float d_sq=d_thr**2

    lent=len(tracks)

    #Network C
    C={0:{'indices':[0],'hidden':tracks[0].copy(),'N':1}}
    ts=np.zeros((3,3),dtype=np.float32)

    #for (it,t) in enumerate(tracks[1:]):
    for it in range(1,lent):
        track=np.ascontiguousarray(tracks[it],dtype=f32_dt)
        lenC=len(C.keys())
        #if it%1000==0:
        #    print it,lenC
        alld=np.zeros(lenC)
        flip=np.zeros(lenC)
        for k in range(lenC):
            h=np.ascontiguousarray(C[k]['hidden']/C[k]['N'],dtype=f32_dt)
            #print track
            #print h
            track_direct_flip_3dist(
                asfp(track[0]),asfp(track[1]),asfp(track[2]),
                asfp(h[0]), asfp(h[1]),asfp(h[2]),<float *>d)
            #d=np.sum(np.sqrt(np.sum((t-h)**2,axis=1)))/3.0
            #ts[0]=t[-1];ts[1]=t[1];ts[-1]=t[0]
            #ds=np.sum(np.sqrt(np.sum((ts-h)**2,axis=1)))/3.0
            #print d[0],d[1]
            if d[1]<d[0]:
                d[0]=d[1]
                flip[k]=1
            alld[k]=d[0]
        m_k=np.min(alld)
        i_k=np.argmin(alld)
        if m_k<d_thr:
            if flip[i_k]==1:
                ts[0]=track[-1];ts[1]=track[1];ts[-1]=track[0]
                C[i_k]['hidden'] = C[i_k]['hidden'] + ts
            else:
                #print(track.shape)
                #print(track.dtype)
                C[i_k]['hidden'] = C[i_k]['hidden'] + track
            C[i_k]['N'] = C[i_k]['N'] + 1
            C[i_k]['indices'].append(it)
        else:
            C[lenC]={}
            C[lenC]['hidden']=track.copy()
            C[lenC]['N']=1
            C[lenC]['indices']=[it]
    return C


cdef inline void track_direct_flip_3sq_dist(float *a1, float *b1,float  *c1,float *a2, float *b2, float *c2, float *out):
    """ Calculate the average squared euclidean distance between two 3pt tracks
    both direct and flip are given as output

    Parameters
    ----------
    a1,b1,c1 : 3 float[3] arrays
        First track.
    a2,b2,c2 : 3 float[3] arrays
        Second track.

    Returns
    -------
    out : float[2] array
        The euclidean distance and the flipped euclidean distance.
    """

    cdef:
        int i
        float tmp1=0,tmp2=0,tmp3=0,tmp1f=0,tmp3f=0
    #for i in range(3):
    for i from 0<=i<3:
        tmp1=tmp1+(a1[i]-a2[i])*(a1[i]-a2[i])
        tmp2=tmp2+(b1[i]-b2[i])*(b1[i]-b2[i])
        tmp3=tmp3+(c1[i]-c2[i])*(c1[i]-c2[i])
        tmp1f=tmp1f+(a1[i]-c2[i])*(a1[i]-c2[i])
        tmp3f=tmp3f+(c1[i]-a2[i])*(c1[i]-a2[i])

    out[0]=(tmp1+tmp2+tmp3)/3.0
    out[1]=(tmp1f+tmp2+tmp3f)/3.0


def larch_3split(tracks, indices=None, thr=10.):
    """Generate a first pass clustering using 3 points on the tracks only.

    Parameters
    ----------
    tracks : sequence
        of tracks as arrays, shape ``(N1,3) .. (Nm,3)``, where 3 points are
        (first, mid and last)
    indices : None or sequence, optional
        Sequence of integer indices of tracks
    trh : float, optional
        squared euclidean distance threshold

    Returns
    -------
    C : dict
        A tree graph containing the clusters.

    Notes
    -----
    If a 3 point track (3track) is far away from all clusters then add a new
    cluster and assign this 3track as the rep(representative) track for the new
    cluster. Otherwise the rep 3track of each cluster is the average track of
    the cluster.

    Examples
    --------
    >>> tracks=[np.array([[0,0,0],[1,0,0,],[2,0,0]],dtype=np.float32),
    ...         np.array([[3,0,0],[3.5,1,0],[4,2,0]],dtype=np.float32),
    ...         np.array([[3.2,0,0],[3.7,1,0],[4.4,2,0]],dtype=np.float32),
    ...         np.array([[3.4,0,0],[3.9,1,0],[4.6,2,0]],dtype=np.float32),
    ...         np.array([[0,0.2,0],[1,0.2,0],[2,0.2,0]],dtype=np.float32),
    ...         np.array([[2,0.2,0],[1,0.2,0],[0,0.2,0]],dtype=np.float32),
    ...         np.array([[0,0,0],[0,1,0],[0,2,0]],dtype=np.float32),
    ...         np.array([[0.2,0,0],[0.2,1,0],[0.2,2,0]],dtype=np.float32),
    ...         np.array([[-0.2,0,0],[-0.2,1,0],[-0.2,2,0]],dtype=np.float32)]
    >>> C = larch_3split(tracks, None, 0.5)

    Here is an example of how to visualize the clustering above::

        from dipy.viz import window, actor
        scene = window.Scene()
        scene.add(actor.line(tracks,window.colors.red))
        window.show(scene)
        for c in C:
            color=np.random.rand(3)
            for i in C[c]['indices']:
                scene.add(actor.line(tracks[i],color))
        window.show(scene)
        for c in C:
            scene.add(actor.line(C[c]['rep3']/C[c]['N'],
                                 window.colors.white))
        window.show(scene)
    """

    cdef:
        cnp.ndarray[cnp.float32_t, ndim=2] track
        cnp.ndarray[cnp.float32_t, ndim=2] h
        int lent,k,it
        float d[2]

    lent=len(tracks)
    if indices==None:
        C={0:{'indices':[0],'rep3':tracks[0].copy(),'N':1}}
        itrange=range(1,lent)
    else:
        C={0:{'indices':[indices[0]],'rep3':tracks[indices[0]].copy(),'N':1}}
        itrange=indices[1:]

    ts=np.zeros((3,3),dtype=np.float32)
    for it in itrange:
        track=np.ascontiguousarray(tracks[it],dtype=f32_dt)
        lenC=len(C.keys())
        alld=np.zeros(lenC)
        flip=np.zeros(lenC)

        for k in range(lenC):
            h=np.ascontiguousarray(C[k]['rep3']/C[k]['N'],dtype=f32_dt)
            track_direct_flip_3dist(asfp(track[0]),asfp(track[1]),asfp(track[2]),
                                    asfp(h[0]), asfp(h[1]), asfp(h[2]),d)
            if d[1]<d[0]:
                d[0]=d[1];flip[k]=1
            alld[k]=d[0]
        m_k=np.min(alld)
        i_k=np.argmin(alld)
        if m_k<thr:
            if flip[i_k]==1:
                ts[0]=track[-1];ts[1]=track[1];ts[-1]=track[0]
                C[i_k]['rep3']+=ts
            else:
                C[i_k]['rep3']+=track
            C[i_k]['N']+=1
            C[i_k]['indices'].append(it)

        else:
            C[lenC]={}
            C[lenC]['rep3']=track.copy()
            C[lenC]['N']=1
            C[lenC]['indices']=[it]


    return C


def larch_3merge(C,thr=10.):
    """
    Reassign tracks to existing clusters by merging clusters that their
    representative tracks are not very distant i.e. less than sqd_thr. Using
    tracks consisting of 3 points (first, mid and last). This is necessary
    after running larch_fast_split after multiple split in different levels
    (squared thresholds) as some of them have created independent clusters.

    Parameters
    ----------
    C : graph with clusters
        of indices 3tracks (tracks consisting of 3 points only)
    sqd_trh: float
        squared euclidean distance threshold

    Returns
    -------
    C : dict
       a tree graph containing the clusters
    """

    cdef cnp.ndarray[cnp.float32_t, ndim=2] h=np.zeros((3,3),dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] ch=np.zeros((3,3),dtype=np.float32)
    cdef int lenC,k,c
    cdef float d[2]

    ts=np.zeros((3,3),dtype=np.float32)

    lenC=len(C)
    C2=C.copy()

    for c in range(0,lenC-1):
        ch=np.ascontiguousarray(C[c]['rep3']/C[c]['N'],dtype=f32_dt)
        krange=range(c+1,lenC)
        klen=len(krange)
        alld=np.zeros(klen)
        flip=np.zeros(klen)
        for k in range(c+1,lenC):
            h=np.ascontiguousarray(C[k]['rep3']/C[k]['N'],dtype=f32_dt)
            track_direct_flip_3dist(
                asfp(ch[0]),asfp(ch[1]),asfp(ch[2]),
                asfp(h[0]), asfp(h[1]), asfp(h[2]),d)

            if d[1]<d[0]:
                d[0]=d[1]
                flip[k-c-1]=1
            alld[k-c-1]=d[0]

        m_k=np.min(alld)
        i_k=np.argmin(alld)
        if m_k<thr:
            if flip[i_k]==1:
                ts[0]=ch[-1];ts[1]=ch[1];ts[-1]=ch[0]
                C2[i_k+c]['rep3']+=ts
            else:
                C2[i_k+c]['rep3']+=ch
            C2[i_k+c]['N']+=C2[c]['N']
            C2[i_k+c]['indices']+=C2[c]['indices']
            del C2[c]

    return C2



def point_track_sq_distance_check(cnp.ndarray[float,ndim=2] track,
                                  cnp.ndarray[float,ndim=1] point,
                                  double sq_dist_thr):
    """ Check if square distance of track from point is smaller than threshold

    Parameters
    ----------
    track: array,float32, shape (N,3)
    point: array,float32, shape (3,)
    sq_dist_thr: double, threshold

    Returns
    -------
    bool: True, if sq_distance <= sq_dist_thr, otherwise False.

    Examples
    --------
    >>> t=np.random.rand(10,3).astype(np.float32)
    >>> p=np.array([0.5,0.5,0.5],dtype=np.float32)
    >>> point_track_sq_distance_check(t,p,2**2)
    True
    >>> t=np.array([[0,0,0],[1,1,1],[2,2,2]],dtype='f4')
    >>> p=np.array([-1,-1.,-1],dtype='f4')
    >>> point_track_sq_distance_check(t,p,.2**2)
    False
    >>> point_track_sq_distance_check(t,p,2**2)
    True
    """

    cdef:
        float *t=<float *> cnp.PyArray_DATA(track)
        float *p=<float *> cnp.PyArray_DATA(point)
        float a[3]
        float b[3]
        int tlen = len(track)
        int curr = 0
        float dist = 0
        int i
        int intersects = 0

    with nogil:
        for i from 0<=i<tlen-1:
            curr=i*3
            a[0]=t[curr]
            a[1]=t[curr+1]
            a[2]=t[curr+2]
            b[0]=t[curr+3]
            b[1]=t[curr+4]
            b[2]=t[curr+5]
            dist=cpoint_segment_sq_dist(<float *>a,<float *>b,p)
            if dist<=sq_dist_thr:
                intersects=1
                break

    if intersects==1:
        return True
    else:
        return False

def track_roi_intersection_check(cnp.ndarray[float,ndim=2] track, cnp.ndarray[float,ndim=2] roi, double sq_dist_thr):
    """ Check if a track is intersecting a region of interest

    Parameters
    ----------
    track: array,float32, shape (N,3)
    roi: array,float32, shape (M,3)
    sq_dist_thr: double, threshold, check squared euclidean distance from every roi point

    Returns
    -------
    bool: True, if sq_distance <= sq_dist_thr, otherwise False.

    Examples
    --------
    >>> roi=np.array([[0,0,0],[1,0,0],[2,0,0]],dtype='f4')
    >>> t=np.array([[0,0,0],[1,1,1],[2,2,2]],dtype='f4')
    >>> track_roi_intersection_check(t,roi,1)
    True
    >>> track_roi_intersection_check(t,np.array([[10,0,0]],dtype='f4'),1)
    False
    """

    cdef:
        float *t=<float *> cnp.PyArray_DATA(track)
        float *r=<float *> cnp.PyArray_DATA(roi)
        float a[3]
        float b[3]
        float p[3]
        int tlen = len(track)
        int rlen = len(roi)
        int curr = 0
        int currp = 0
        float dist = 0
        int i,j
        int intersects=0

    with nogil:
        for i from 0<=i<tlen-1:
            curr=i*3
            a[0]=t[curr]
            a[1]=t[curr+1]
            a[2]=t[curr+2]
            b[0]=t[curr+3]
            b[1]=t[curr+4]
            b[2]=t[curr+5]
            for j from 0<=j<rlen:
                currp=j*3
                p[0]=r[currp]
                p[1]=r[currp+1]
                p[2]=r[currp+2]
                dist=cpoint_segment_sq_dist(<float *>a,<float *>b,<float *>p)
                if dist<=sq_dist_thr:
                    intersects=1
                    break
            if intersects==1:
                break
    if intersects==1:
        return True
    else:
        return False
