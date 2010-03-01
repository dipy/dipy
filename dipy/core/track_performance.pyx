''' A type of -*- python -*- file

Performance functions for dipy


'''
# cython: profile=True

cimport cython

import numpy as np
import time
cimport numpy as cnp


cdef extern from "math.h" nogil:
    double floor(double x)
    float sqrt(float x)
    float fabs(float x)
    double log2(double x)
    float acos(float x )    
    bint isnan(double x) 
    
#cdef extern from "stdio.h":
#	void printf ( const char * format, ... )
    
cdef extern from "stdlib.h" nogil:
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *malloc(size_t size)
    void *calloc(size_t nelem, size_t elsize)
    void *realloc (void *ptr, size_t size)
    void *memcpy(void *str1, void *str2, size_t n)

#@cython.boundscheck(False)
#@cython.wraparound(False)


cdef inline cnp.ndarray[cnp.float32_t, ndim=1] as_float_3vec(object vec):
    ''' Utility function to convert object to 3D float vector '''
    return np.squeeze(np.asarray(vec, dtype=np.float32))


cdef inline float* asfp(cnp.ndarray pt):
    return <float *>pt.data


def normalized_3vec(vec):
    ''' Return normalized 3D vector

    Vector divided by Euclidean (L2) norm

    Parameters
    ----------
    vec : array-like shape (3,)

    Returns
    -------
    vec_out : array shape (3,)
    '''
    cdef cnp.ndarray[cnp.float32_t, ndim=1] vec_in = as_float_3vec(vec)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] vec_out = np.zeros((3,), np.float32)
    cnormalized_3vec(<float *>vec_in.data, <float*>vec_out.data)
    return vec_out


def norm_3vec(vec):
    ''' Euclidean (L2) norm of length 3 vector

    Parameters
    ----------
    vec : array-like shape (3,)

    Returns
    -------
    norm : float
       Euclidean norm
    '''
    cdef cnp.ndarray[cnp.float32_t, ndim=1] vec_in = as_float_3vec(vec)
    return cnorm_3vec(<float *>vec_in.data)


cdef inline float cnorm_3vec(float *vec):
    ''' Calculate Euclidean norm of input vector

    Parameters
    ----------
    vec : float *
       length 3 float vector

    Returns
    -------
    norm : float
       Euclidean norm
    '''
    cdef float v0, v1, v2
    v0 = vec[0]
    v1 = vec[1]
    v2 = vec[2]
    return sqrt(v0 * v0 + v1*v1 + v2*v2)


cdef inline void cnormalized_3vec(float *vec_in, float *vec_out):
    ''' Calculate and fill normalized 3D vector 

    Parameters
    ----------
    vec_in : float *
       Length 3 vector to normalize
    vec_out : float *
       Memory into which to write normalized length 3 vector

    Returns
    -------
    void
    '''
    cdef float norm = cnorm_3vec(vec_in)
    cdef int i
    for i in range(3):
        vec_out[i] = vec_in[i] / norm
        

def inner_3vecs(vec1, vec2):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec1 = as_float_3vec(vec1)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec2 = as_float_3vec(vec2)
    return cinner_3vecs(<float *>fvec1.data, <float*>fvec2.data)


cdef inline float cinner_3vecs(float *vec1, float *vec2):
    cdef int i
    cdef float ip = 0
    for i in range(3):
        ip += vec1[i]*vec2[i]
    return ip


def sub_3vecs(vec1, vec2):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec1 = as_float_3vec(vec1)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec2 = as_float_3vec(vec2)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] vec_out = np.zeros((3,), np.float32)    
    csub_3vecs(<float *>fvec1.data, <float*>fvec2.data, <float *>vec_out.data)
    return vec_out


cdef inline void csub_3vecs(float *vec1, float *vec2, float *vec_out):
    cdef int i
    for i in range(3):
        vec_out[i] = vec1[i]-vec2[i]


def add_3vecs(vec1, vec2):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec1 = as_float_3vec(vec1)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec2 = as_float_3vec(vec2)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] vec_out = np.zeros((3,), np.float32)    
    cadd_3vecs(<float *>fvec1.data, <float*>fvec2.data, <float *>vec_out.data)
    return vec_out


cdef inline void cadd_3vecs(float *vec1, float *vec2, float *vec_out):
    cdef int i
    for i in range(3):
        vec_out[i] = vec1[i]+vec2[i]

def mul_3vecs(vec1, vec2):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec1 = as_float_3vec(vec1)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec2 = as_float_3vec(vec2)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] vec_out = np.zeros((3,), np.float32)    
    cmul_3vecs(<float *>fvec1.data, <float*>fvec2.data, <float *>vec_out.data)
    return vec_out

cdef inline void cmul_3vecs(float *vec1, float *vec2, float *vec_out):
    cdef int i
    for i in range(3):
        vec_out[i] = vec1[i]*vec2[i]

def mul_3vec(a, vec):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec = as_float_3vec(vec)    
    cdef cnp.ndarray[cnp.float32_t, ndim=1] vec_out = np.zeros((3,), np.float32)    
    cmul_3vec(a,<float *>fvec.data, <float *>vec_out.data)
    return vec_out        

cdef inline void cmul_3vec(float a, float *vec, float *vec_out):
    cdef int i
    for i in range(3):
        vec_out[i] = a*vec[i]



# float 32 dtype for casting
cdef cnp.dtype f32_dt = np.dtype(np.float32)


def cut_plane(tracks,ref):
    ''' Extract divergence vectors and points of intersection 
    between planes normal to the reference fiber and other tracks
    
    Parameters
    ----------
    tracks: sequence
        of tracks as arrays, shape (N1,3) .. (Nm,3)
    ref: array, shape (N,3)
        reference track
        
    Returns
    -------
    hits: sequence
       list of points and rcds (radial coefficient of divergence)``
    
    Examples
    --------
    >>> refx = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]],dtype='float32')
    >>> bundlex = [np.array([[0.5,1,0],[1.5,2,0],[2.5,3,0]],dtype='float32')]
    >>> cut_plane(bundlex,refx)
        [array([[ 1.        ,  1.5       ,  0.        ,  0.70710683, 0,]], dtype=float32),
         array([[ 2.        ,  2.5       ,  0.        ,  0.70710677, 0.]], dtype=float32)]
        
        The orthogonality relationship
        np.inner(hits[p][q][0:3]-ref[p+1],ref[p+2]-ref[r][p+1])
        will hold throughout for every point q in the hits plane
        at point (p+1) on the reference track.
    '''
    cdef:
        size_t n_hits, hit_no, max_hit_len
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
        size_t N_tracks=len(tracks)
        cnp.ndarray[cnp.uint64_t, ndim=1] track_lengths
        size_t t_no, N_track
    cdef object tracks32 = []
    track_lengths = np.empty((N_tracks,), dtype=np.uint64)
    for t_no in range(N_tracks):
        track = np.ascontiguousarray(tracks[t_no], f32_dt)
        track_lengths[t_no] = track.shape[0]
        tracks32.append(track)
    # set up loop across reference fiber points
    cdef:
        size_t N_ref = ref32.shape[0]
        size_t p_no, q_no
        float *this_ref_p, *next_ref_p, *this_trk_p, *next_trk_p
        float along[3], normal[3]
        float qMp[3], rMp[3], rMq[3], pMq[3]
        float hit[3], hitMp[3], *delta
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
                            ''' # Summary of stuff in comments
                            # divergence =((r-q)-inner(r-q,normal)*normal)/|r-q|
                            div[0] = (rMq[0]-beta*normal[0]) / ld
                            div[1] = (rMq[1]-beta*normal[1]) / ld
                            div[2] = (rMq[2]-beta*normal[2]) / ld
                            # radial coefficient of divergence d.(h-p)/|h-p|
                            '''
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
                            hit_ptr = <float *>one_hit.data
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


DEF biggest_double = 1.79769e+308


def most_similar_track_zhang(tracks,metric='avg'):    
    ''' The purpose of this function is to implement a much faster version of 
    most_similar_track_zhang from dipy.core.track_metrics  as we implemented 
    from Zhang et. al 2008. 
    
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
    '''
    cdef:
        size_t i, j, lent
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
        size_t longest_track_len = 0, track_len
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
    cdef cnp.double_t *sum_track2others = <cnp.double_t *>track2others.data
    # preallocate buffer array for track distance calculations
    cdef:
        cnp.ndarray [cnp.float32_t, ndim=1] distances_buffer
        cnp.float32_t *t1_ptr, *t2_ptr, *min_buffer, distance
    distances_buffer = np.zeros((longest_track_len*2,), dtype=np.float32)
    min_buffer = <cnp.float32_t *> distances_buffer.data
    # cycle over tracks
    cdef:
        cnp.ndarray [cnp.float32_t, ndim=2] t1, t2
        size_t t1_len, t2_len
    for i from 0 <= i < lent-1:
        t1 = tracks32[i]
        t1_len = t1.shape[0]
        t1_ptr = <cnp.float32_t *>t1.data
        for j from i+1 <= j < lent:
            t2 = tracks32[j]
            t2_len = t2.shape[0]
            t2_ptr = <cnp.float32_t *>t2.data
            distance = czhang(t1_len, t1_ptr, t2_len, t2_ptr, min_buffer, metric_type)
            # get metric
            sum_track2others[i]+=distance
            sum_track2others[j]+=distance
    # find track with smallest summed metric with other tracks
    cdef double mn = sum_track2others[0]
    cdef size_t si = 0
    for i in range(lent):
        if sum_track2others[i] < mn:
            si = i
            mn = sum_track2others[i]
    # recalculate distance of this track from the others
    t1 = tracks32[si]
    t1_len = t1.shape[0]
    t1_ptr = <cnp.float32_t *>t1.data
    for j from 0 <= j < lent:
        t2 = tracks32[j]
        t2_len = t2.shape[0]
        t2_ptr = <cnp.float32_t *>t2.data
        track2others[j] = czhang(t1_len, t1_ptr, t2_len, t2_ptr, min_buffer, metric_type)
    return si, track2others


cdef cnp.float32_t inf = np.inf


cdef inline cnp.float32_t czhang(size_t t1_len,
                                 cnp.float32_t *track1_ptr,
                                 size_t t2_len,
                                 cnp.float32_t *track2_ptr,
                                 cnp.float32_t *min_buffer,
                                 int metric_type) nogil:
    ''' Note ``nogil`` - no python calls allowed in this function '''
    cdef:
        cnp.float32_t *min_t2t1, *min_t1t2
    min_t2t1 = min_buffer
    min_t1t2 = min_buffer + t2_len
    min_distances(t1_len, track1_ptr,
                  t2_len, track2_ptr,
                  min_t2t1,
                  min_t1t2)
    cdef:
        size_t t1_pi, t2_pi
        cnp.float32_t mean_t2t1 = 0, mean_t1t2 = 0, dist_val
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


cdef inline void min_distances(size_t t1_len,
                               cnp.float32_t *track1_ptr,
                               size_t t2_len,
                               cnp.float32_t *track2_ptr,
                               cnp.float32_t *min_t2t1,
                               cnp.float32_t *min_t1t2) nogil:
    cdef:
        cnp.float32_t *t1_pt, *t2_pt, d0, d1, d2
        cnp.float32_t delta2
        int t1_pi, t2_pi
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


def zhang_distances(xyz1,xyz2,metric='all'):
    ''' Distance between tracks xyz1 and xyz2 using Zhang metrics
    
    Based on the metrics in Zhang, Correia, Laidlaw 2008
    http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=4479455
    which in turn are based on those of Corouge et al. 2004
        
    This function should return the same results with zhang_distances 
    from track_metrics but hopefully faster.
 
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
    avg_mcd: float
       average_mean_closest_distance
    min_mcd: float
       minimum_mean_closest_distance
    max_mcd: float
       maximum_mean_closest_distance
                    
    Notes
    -----
    Algorithmic description
    
    Lets say we have curves A and B.
    
    For every point in A calculate the minimum distance from every point
    in B stored in minAB
    
    For every point in B calculate the minimum distance from every point
    in A stored in minBA
    
    find average of minAB stored as avg_minAB
    find average of minBA stored as avg_minBA
    
    if metric is 'avg' then return (avg_minAB + avg_minBA)/2.0
    if metric is 'min' then return min(avg_minAB,avg_minBA)
    if metric is 'max' then return max(avg_minAB,avg_minBA)
    '''
    cdef:
        cnp.ndarray[cnp.float32_t, ndim=2] track1 
        cnp.ndarray[cnp.float32_t, ndim=2] track2
        size_t t1_len, t2_len
    track1 = np.ascontiguousarray(xyz1, dtype=f32_dt)
    t1_len = track1.shape[0]
    track2 = np.ascontiguousarray(xyz2, dtype=f32_dt)
    t2_len = track2.shape[0]
    # preallocate buffer array for track distance calculations
    cdef:
        cnp.float32_t *min_t2t1, *min_t1t2
        cnp.ndarray [cnp.float32_t, ndim=1] distances_buffer
    distances_buffer = np.zeros((t1_len + t2_len,), dtype=np.float32)
    min_t2t1 = <cnp.float32_t *> distances_buffer.data
    min_t1t2 = min_t2t1 + t2_len
    min_distances(t1_len, <cnp.float32_t *>track1.data,
                  t2_len, <cnp.float32_t *>track2.data,
                  min_t2t1,
                  min_t1t2)
    cdef:
        size_t t1_pi, t2_pi
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
    ''' Find the minimum distance between two curves xyz1, xyz2
    
    Parameters:
    -----------
        xyz1 : array, shape (N1,3), dtype float32
        xyz2 : array, shape (N2,3), dtype float32
        arrays representing x,y,z of the N1 and N2 points  of two tracks
    
    Returns:
    -----------
    
    Notes:
    ---------
    Algorithmic description
    
    Lets say we have curves A and B
    
    for every point in A calculate the minimum distance from every point in B stored in minAB
    for every point in B calculate the minimum distance from every point in A stored in minBA
    find min of minAB stored in min_minAB
    find min of minBA stored in min_minBA
    
    Then return (min_minAB + min_minBA)/2.0
    '''
    cdef:
        cnp.ndarray[cnp.float32_t, ndim=2] track1 
        cnp.ndarray[cnp.float32_t, ndim=2] track2
        size_t t1_len, t2_len
    track1 = np.ascontiguousarray(xyz1, dtype=f32_dt)
    t1_len = track1.shape[0]
    track2 = np.ascontiguousarray(xyz2, dtype=f32_dt)
    t2_len = track2.shape[0]
    # preallocate buffer array for track distance calculations
    cdef:
        cnp.float32_t *min_t2t1, *min_t1t2
        cnp.ndarray [cnp.float32_t, ndim=1] distances_buffer
    distances_buffer = np.zeros((t1_len + t2_len,), dtype=np.float32)
    min_t2t1 = <cnp.float32_t *> distances_buffer.data
    min_t1t2 = min_t2t1 + t2_len
    min_distances(t1_len, <cnp.float32_t *>track1.data,
                  t2_len, <cnp.float32_t *>track2.data,
                  min_t2t1,
                  min_t1t2)
    cdef:
        size_t t1_pi, t2_pi
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
    ''' Based on Lee , Han & Whang SIGMOD07.
        Calculates perpendicular distance metric for the distance between two line segments
        
    This function assumes that norm(end0-start0)>norm(end1-start1)
    i.e. that the first segment will be bigger than the second one.
    Parameters:
    -----------
        start0: float array(3,)
        end0: float array(3,)
        start1: float array(3,)
        end1: float array(3,)
    
    Returns:
    --------
        perpendicular_distance: float

    Examples:
    ---------
    >>> import dipy.core.performance as pf
    >>> pf.lee_perpendicular_distance([0,0,0],[1,0,0],[3,4,5],[5,4,3])
    >>> 5.9380966767403436  
    
    Description:
    ------------
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
    
    '''
    
    cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec1,fvec2,fvec3,fvec4
    
    fvec1 = as_float_3vec(start0)
    fvec2 = as_float_3vec(end0)
    fvec3 = as_float_3vec(start1)
    fvec4 = as_float_3vec(end1)
    
    return clee_perpendicular_distance(<float *>fvec1.data,<float *>fvec2.data,<float *>fvec3.data,<float *>fvec4.data)
        
    
cdef float clee_perpendicular_distance(float *start0, float *end0,float *start1, float *end1):
    '''
    This function assumes that
    norm(end0-start0)>norm(end1-start1)
    '''

    cdef:
        float l0,l1,ltmp,u1,u2,lperp1,lperp2         
        float *s_tmp,*e_tmp,k0[3],ps[3],pe[3],ps1[3],pe1[3],tmp[3]
               
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
    ''' Based on Lee , Han & Whang SIGMOD07.
        Calculates angle distance metric for the distance between two line segments
            
    This function assumes that norm(end0-start0)>norm(end1-start1)
    i.e. that the first segment will be bigger than the second one.
    
    Parameters:
    -----------
        start0: float array(3,)
        end0: float array(3,)
        start1: float array(3,)
        end1: float array(3,)
    
    Returns:
    --------
        angle_distance: float

    Examples:
    --------
    >>> import dipy.core.track_metrics as tm 
    >>> tm.lee_angle_distance([0,0,0],[1,0,0],[3,4,5],[5,4,3])
    >>> 2.0 
    
    Descritpion:
    ------------
    
    l_0 = np.inner(end0-start0,end0-start0)
    l_1 = np.inner(end1-start1,end1-start1)
    
    cos_theta_squared = np.inner(end0-start0,end1-start1)**2/ (l_0*l_1)
    return np.sqrt((1-cos_theta_squared)*l_1)

    '''

    cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec1,fvec2,fvec3,fvec4
    
    fvec1 = as_float_3vec(start0)
    fvec2 = as_float_3vec(end0)
    fvec3 = as_float_3vec(start1)
    fvec4 = as_float_3vec(end1)
    
    return clee_angle_distance(<float *>fvec1.data,<float *>fvec2.data,<float *>fvec3.data,<float *>fvec4.data)

cdef float clee_angle_distance(float *start0, float *end0,float *start1, float *end1):
    '''
    This function assumes that
    norm(end0-start0)>norm(end1-start1)
    '''

    cdef:
        float l0,l1,ltmp,cos_theta_squared         
        float *s_tmp,*e_tmp,k0[3],k1[3],tmp[3]
               
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

def approximate_ei_trajectory(xyz,alpha=0.392):
    ''' Fast and simple Approximate Trajectory
        Algorithm by Eleftherios and Ian
    
    Parameters:
    ------------------
    xyz: array(N,3) 
        initial trajectory
    alpha: float
        smoothing parameter (<0.392 smoother, <0.392  rougher)
    
    Returns:
    ------------
    characteristic_points: list of M array(3,) points
    
    Examples:
    -------------
    >>> #approximating a helix
    >>> t=np.linspace(0,1.75*2*np.pi,100)
    >>> x = np.sin(t)
    >>> y = np.cos(t)
    >>> z = t        
    >>> xyz=np.vstack((x,y,z)).T     
    >>> xyza = pf.approximate_ei_trajectory(xyz)
    >>> len(xyz)
    >>> len(xyza)
    
    Description :
    -----------------
    Assuming that a good approximation for a circle is an octagon then that means that the points of the octagon will have 
    angle alpha = 2*pi/8 = pi/4 . We calculate the angle between every two neighbour segments of a trajectory and if the angle
    is higher than pi/4 we choose that point as a characteristic point otherwise      
    '''
    cdef :
        int mid_index
        cnp.ndarray[cnp.float32_t, ndim=2] track 
        float *fvec0,*fvec1,*fvec2
        object characteristic_points
        size_t t_len
        double angle,tmp
        float vec0[3],vec1[3]
    
    angle=alpha
    
    track = np.ascontiguousarray(xyz, dtype=f32_dt)
    t_len=len(track)
    
    characteristic_points=[track[0]]
    mid_index = 1
    angle=0
    
    while mid_index < t_len-1:
        
        #fvec0 = as_float_3vec(track[mid_index-1])
        #<float *>track[0].data
        
        fvec0 = asfp(track[mid_index-1])
        fvec1 = asfp(track[mid_index])
        fvec2 = asfp(track[mid_index+1])
        
        #csub_3vecs(<float *>fvec1.data,<float *>fvec0.data,vec0)
        csub_3vecs(fvec1,fvec0,vec0)
        csub_3vecs(fvec2,fvec1,vec1)
          
        
        tmp=<double>fabs(acos(cinner_3vecs(vec0,vec1)/(cnorm_3vec(vec0)*cnorm_3vec(vec1))))        
        
        if isnan(tmp) :
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
    ''' Implementation of Lee et al Approximate Trajectory
        Partitioning Algorithm
    
    Parameters:
    ------------------
    xyz: array(N,3) 
        initial trajectory
    alpha: float
        smoothing parameter (>1 smoother, <1  rougher)
    
    Returns:
    ------------
    characteristic_points: list of M array(3,) points
        
    '''
    cdef :
        int start_index,length,current_index, i
        double cost_par,cost_nopar,alphac
        object characteristic_points
        size_t t_len
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
        csub_3vecs(<float *>fvec2.data,<float *>fvec1.data,tmp)
        cost_par=log2(sqrt(cinner_3vecs(tmp,tmp)))
        cost_nopar=0
        #print start_index,current_index
        
        # L(D|H)
        #for i in range(start_index+1,current_index):#+1):
        for i in range(start_index,current_index+1):
            
            #print i
            fvec3 = as_float_3vec(track[i])
            fvec4 = as_float_3vec(track[i+1])
                
            cost_par += log2(clee_perpendicular_distance(<float *>fvec3.data,<float *>fvec4.data,<float *>fvec1.data,<float *>fvec2.data))        
            cost_par += log2(clee_angle_distance(<float *>fvec3.data,<float *>fvec4.data,<float *>fvec1.data,<float *>fvec2.data))

            csub_3vecs(<float *>fvec4.data,<float *>fvec3.data,tmp)
            cost_nopar += log2(cinner_3vecs(tmp,tmp))
            
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
    '''
    Intersect Segment S(t) = sa +t(sb-sa), 0 <=t<= 1 against cylinder specified by p,q and r    
    
    Look p.197 from Real Time Collision Detection C. Ericson
    
    Example:
    ------------
    >>> # Define cylinder using a segment defined by 
    >>> p=np.array([0,0,0],dtype=float32)
    >>> q=np.array([1,0,0],dtype=float32)
    >>> r=0.5
    >>> # Define segment
    >>> sa=np.array([0.5,1 ,0],dtype=float32)
    >>> sb=np.array([0.5,-1,0],dtype=float32)
    >>> from dipy.core import performance as pf
    >>> 
    '''
    cdef:
        float *csa,*csb,*cp,*cq
        float cr
        float ct[2]
        
                
    csa = asfp(sa)
    csb = asfp(sb)
    cp = asfp(p)
    cq = asfp(q)
    cr=r
    ct[0]=-100
    ct[1]=-100
    
    tmp= cintersect_segment_cylinder(csa,csb,cp, cq, cr, ct)

    return tmp,ct[0],ct[1]


    
cdef float cintersect_segment_cylinder(float *sa,float *sb,float *p, float *q, float r, float *t):
    ''' Intersect Segment S(t) = sa +t(sb-sa), 0 <=t<= 1 against cylinder specified by p,q and r    
    
    Look p.197 from Real Time Collision Detection C. Ericson
    
    0 no intersection
    1 intersection   
            
    '''
    cdef:
        float d[3],m[3],n[3]
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
    

def point_segment_sq_distance(a,b,c):
    ''' Calculate the squared distance from a point c to a finite line segment ab.
 
    Examples:
    -------------
    >>> from dipy.core import performance as pf
    >>> a=np.array([0,0,0],dtype=float32)
    >>> b=np.array([1,0,0],dtype=float32)
    >>> c=np.array([0,1,0],dtype=float32)    
    >>> pf.point_segment_sq_distance(a,b,c)
    >>> 1.0
    >>> c=np.array([0,3,0],dtype=float32)
    >>> pf.point_segment_sq_distance(a,b,c)
    >>> 9.0 
    >>> c=np.array([-1,1,0],dtype=float32)
    >>> pf.point_segment_sq_distance(a,b,c)
    >>> 2.0
    

    '''
    cdef:
        float *ca,*cb,*cc
        float cr
        float ct[2]
        
                
    ca = asfp(a)
    cb = asfp(b)
    cc = asfp(c)
    
    return cpoint_segment_sq_dist(ca, cb, cc)
    
cdef inline float cpoint_segment_sq_dist(float * a, float * b, float * c):
    ''' Calculate the squared distance from a point c to a line segment ab.
    
    '''
    cdef:
        float ab[3],ac[3],bc[3]
        float e,f

    csub_3vecs(b,a,ab)
    csub_3vecs(c,a,ac)
    csub_3vecs(c,b,bc)
    
    e = cinner_3vecs(ac, ab)
    #Handle cases where c projects outside ab
    if e <= 0.:  return cinner_3vecs(ac, ac)
    f = cinner_3vecs(ab, ab)
    if e >= f : return cinner_3vecs(bc, bc)
    #Handle case where c projects onto ab
    return cinner_3vecs(ac, ac) - e * e / f

    
def local_skeleton_3pts(tracks):
    ''' Calculate a very fast connectivity profile using only three equidistant points along the track
    

    
    '''
    cdef:
    
        int i,j,lent
        float *i_pts0,*i_pts1,*i_pts2, *j_pts0,*j_pts1,*j_pts2
        cnp.ndarray[cnp.float32_t, ndim=2] T
        
       
    lent = len(tracks)
    
    T=np.concatenate(tracks)
            
    for i in range(lent):

        if i %10000 ==0 :
            print i
            
        i_pts0=asfp(3*T[i])
        i_pts1=asfp(3*T[i]+1)
        i_pts2=asfp(3*T[i]+2)
                
        for j in range(250000-i):
                                    
            j_pts0=asfp(3*T[j])
            j_pts1=asfp(3*T[j]+1)
            j_pts2=asfp(3*T[j]+2)
        
            
            
            
    return lent


def track_dist_3pts(tracka,trackb):

    ''' Calculate the euclidean distance between two 3pt tracks
    both direct and flip distances are calculated but only the smallest is returned

    Parameters:
    -----------
    a: array, shape (3,3)
    a three point track
    b: array, shape (3,3)
    a three point track

    Returns:
    --------
    dist:float

    Example:
    -------
    >>> import numpy as np
    >>> a=np.array([[0,0,0],[1,0,0,],[2,0,0]])            
    >>> b=np.array([[3,0,0],[3.5,1,0],[4,2,0]])
    >>> track_dist_3pts(a,b)
    

    '''
    
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

    
    
        
    

cdef inline void track_direct_flip_3dist(float *a1, float *b1,float  *c1,float *a2, float *b2, float *c2, float *out):
    ''' Calculate the euclidean distance between two 3pt tracks
    both direct and flip are given as output
    
    
    Parameters:
    ----------------
    a1,b1,c1: 3 float[3] arrays representing the first track
    a2,b2,c2: 3 float[3] arrays representing the second track
    
    Returns:
    -----------
    out: a float[2] array having the euclidean distance and the fliped euclidean distance
    
    
    '''
    
    cdef:
        int i
        float tmp1=0,tmp2=0,tmp3=0,tmp1f=0,tmp3f=0
        
    
    for i in range(3):
        tmp1=tmp1+(a1[i]-a2[i])*(a1[i]-a2[i])
        tmp2=tmp2+(b1[i]-b2[i])*(b1[i]-b2[i])
        tmp3=tmp3+(c1[i]-c2[i])*(c1[i]-c2[i])
        tmp1f=tmp1f+(a1[i]-c2[i])*(a1[i]-c2[i])
        tmp3f=tmp3f+(c1[i]-a2[i])*(c1[i]-a2[i])
                
    out[0]=(sqrt(tmp1)+sqrt(tmp2)+sqrt(tmp3))/3.0
    out[1]=(sqrt(tmp1f)+sqrt(tmp2)+sqrt(tmp3f))/3.0

    #out[0]=(tmp1+tmp2+tmp3)/3.0
    #out[1]=(tmp1f+tmp2+tmp3f)/3.0





 
    

def local_skeleton_clustering(tracks, d_thr=10):
    ''' For historical purposes as it was used for the HBM2010 abstract
    "Fast Dimensionality Reduction for Brain Tractography Clustering" by E.Garyfallidis et.al
    we keep this function that does a first pass clustering.

    Parameters:
    -----------
    tracks: sequence
        of tracks as arrays, shape (N1,3) .. (Nm,3)

    d_thr: float, average euclidean distance threshold


    Returns:
    --------
    C: dict
    
    
    Example:
    -----------
    >>> from dipy.viz import fos
        
    >>> tracks=[np.array([[0,0,0],[1,0,0,],[2,0,0]]),            
            np.array([[3,0,0],[3.5,1,0],[4,2,0]]),
            np.array([[3.2,0,0],[3.7,1,0],[4.4,2,0]]),
            np.array([[3.4,0,0],[3.9,1,0],[4.6,2,0]]),
            np.array([[0,0.2,0],[1,0.2,0],[2,0.2,0]]),
            np.array([[2,0.2,0],[1,0.2,0],[0,0.2,0]]),
            np.array([[0,0,0],[0,1,0],[0,2,0]])]
                                    
    >>> C=local_skeleton_clustering(tracks,d_thr=0.5)    
    
    >>> r=fos.ren()

    >>> for c in C:
        color=np.random.rand(3)
        for i in C[c]['indices']:
            fos.add(r,fos.line(tracks[i],color))

    >>> fos.show(r)

    '''
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
        
        if it%1000==0:
            print it,lenC
        
        alld=np.zeros(lenC)
        flip=np.zeros(lenC)
        

        for k in range(lenC):
        
            h=np.ascontiguousarray(C[k]['hidden']/C[k]['N'],dtype=f32_dt)
            
            #print track
            #print h
            track_direct_flip_3dist(
                asfp(track[0]),asfp(track[1]),asfp(track[2]), 
                asfp(h[0]), asfp(h[1]),asfp(h[2]),d)
                
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
                C[i_k]['hidden']+=ts
            else:                
                C[i_k]['hidden']+=track
                
            C[i_k]['N']+=1
            C[i_k]['indices'].append(it)
            
        else:
            C[lenC]={}
            C[lenC]['hidden']=track.copy()
            C[lenC]['N']=1
            C[lenC]['indices']=[it]
    
    '''   
    fos.clear(r)

    color=[fos.red,fos.green,fos.blue,fos.yellow]
    for c in C:
        for i in C[c]['indices']:
            fos.add(r,fos.line(tracks[i],color[c]))
                
    fos.show(r)
    '''
    
    return C


cdef inline void track_direct_flip_3sq_dist(float *a1, float *b1,float  *c1,float *a2, float *b2, float *c2, float *out):
    ''' Calculate the average squared euclidean distance between two 3pt tracks
    both direct and flip are given as output
    
    
    Parameters:
    -----------
    a1,b1,c1: 3 float[3] arrays representing the first track
    a2,b2,c2: 3 float[3] arrays representing the second track
    
    Returns:
    --------
    out: a float[2] array having the euclidean distance and the fliped euclidean distance
        
    '''
    
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


def larch_fast_split(tracks,indices=None,sqd_thr=50**2):

    ''' Generate a first pass clustering using 3 points (first, mid and last) on the tracks only.


    Parameters:
    -----------

    tracks: sequence
        of tracks as arrays, shape (N1,3) .. (Nm,3)

    indices: sequence 
        of integer indices of tracks  
    
    sqd_trh: float
        squared euclidean distance threshold
    
    Returns:
    --------

    C: dict, a tree graph containing the clusters

    Examples:
    ---------

    >>> from dipy.viz import fos        
    >>> tracks=[np.array([[0,0,0],[1,0,0,],[2,0,0]]),            
            np.array([[3,0,0],[3.5,1,0],[4,2,0]]),
            np.array([[3.2,0,0],[3.7,1,0],[4.4,2,0]]),
            np.array([[3.4,0,0],[3.9,1,0],[4.6,2,0]]),
            np.array([[0,0.2,0],[1,0.2,0],[2,0.2,0]]),
            np.array([[2,0.2,0],[1,0.2,0],[0,0.2,0]]),
            np.array([[0,0,0],[0,1,0],[0,2,0]])]
                                    
    >>> C=larch_fast_split(tracks,None,0.5**2)        
    >>> r=fos.ren()
    >>> for c in C:
        color=np.random.rand(3)
        for i in C[c]['indices']:
            fos.add(r,fos.line(tracks[i],color))
    >>> fos.show(r)
    
    


    Notes:
    ------
    
    If a 3 point track (3track) is far away from all clusters then add a new cluster and assign
    this 3track as the rep(resentative) track for the new cluster. Otherwise the rep
    3track of each cluster is the average track of the cluster

    '''

    cdef :
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
            
            track_direct_flip_3sq_dist(
                asfp(track[0]),asfp(track[1]),asfp(track[2]), 
                asfp(h[0]), asfp(h[1]), asfp(h[2]),d)
                
            if d[1]<d[0]:                
                d[0]=d[1];flip[k]=1
                
            alld[k]=d[0]

        m_k=np.min(alld)
        i_k=np.argmin(alld)
        
        if m_k<sqd_thr:            
            
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


def larch_fast_reassign(C,sqd_thr=100):

    ''' Reassign tracks to existing clusters by merging clusters that their representative tracks are not very distant i.e. less than sqd_thr. Using tracks consisting of 3 points (first, mid and last). This is necessary after running larch_fast_split after multiple split in different levels (squared thresholds) as some of them have created independent clusters.

    Parameters:
    -----------      

    C: graph with clusters
        of indices 3tracks (tracks consisting of 3 points only)

    sqd_trh: float
        squared euclidean distance threshold
    
    Returns:
    --------

    C: dict, a tree graph containing the clusters


    '''

    cdef cnp.ndarray[cnp.float32_t, ndim=2] h=np.zeros((3,3),dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] ch=np.zeros((3,3),dtype=np.float32)
    cdef int lenC,k,c
    cdef float d[2] 

    ts=np.zeros((3,3),dtype=np.float32)

    lenC=len(C)
    
    for c in range(0,lenC-1):


        ch=np.ascontiguousarray(C[c]['rep3']/C[c]['N'],dtype=f32_dt)        

        krange=range(c+1,lenC)
        klen=len(krange)

        alld=np.zeros(klen)
        flip=np.zeros(klen)


        for k in range(c+1,lenC):

            h=np.ascontiguousarray(C[k]['rep3']/C[k]['N'],dtype=f32_dt)

            track_direct_flip_3sq_dist(
                asfp(ch[0]),asfp(ch[1]),asfp(ch[2]), 
                asfp(h[0]), asfp(h[1]), asfp(h[2]),d)
                
            if d[1]<d[0]:                
                
                d[0]=d[1]
                flip[k-c-1]=1
                
            alld[k-c-1]=d[0]

        m_k=np.min(alld)
        i_k=np.argmin(alld)


        if m_k<sqd_thr:     

            if flip[i_k]==1:                
                ts[0]=ch[-1];ts[1]=ch[1];ts[-1]=ch[0]
                C[i_k+c]['rep3']+=ts
            else:
                C[i_k+c]['rep3']+=ch
                
            C[i_k+c]['N']+=C[c]['N']

            C[i_k+c]['indices']+=C[c]['indices']

            del C[c]


    return C


def larch_preproc(tracks,split_thrs=[50**2,20**2,10.**2],info=False):
    ''' Preprocessing stage

    Parameters:
    -----------
    tracks: sequence
        of tracks as arrays, shape (N1,3) .. (Nm,3)

    split_thrs: sequence
        of floats with thresholds.

    Returns:
    --------
    C: dict, a tree graph containing the clusters

    '''
    t1=time.clock()

    C=larch_fast_split(tracks,None,20.**2)

    print 'Splitting done in ',time.clock()-t1, 'secs', 'len', len(C)

    t2=time.clock()
    C=larch_fast_reassign(C,20.**2)

    print 'Reassignment done in ',time.clock()-t2, 'secs', 'len', len(C)


    return C

    '''

    if info: print 'Spliting in 3 levels with thresholds',split_thrs
    
    #1st level spliting
    C=larch_fast_split(tracks,None,split_thrs[0])

    

    C_leafs={}

    cdef int c_id=0 

    #2nd level spliting
    for k in C:

        C[k]['sub']=larch_fast_split(tracks,C[k]['indices'],split_thrs[1])

        #3rd level spliting
        for l in C[k]['sub']:
            
            C[k]['sub'][l]['sub']=larch_fast_split(tracks,C[k]['sub'][l]['indices'],split_thrs[2])

            #copying the leafs in a new graph
            for m in C[k]['sub'][l]['sub']:

                C_leafs[c_id]=C[k]['sub'][l]['sub'][m]
                c_id+=1


    if info: 
        print 'Number of clusters after spliting ...', len(C_leafs)
        print 'Starting larch_fast_reassignment  ...'
    
    t1=time.clock()
    C_leafs=larch_fast_reassign(C_leafs,split_thrs[2])

    print 'Reassignment done in ',time.clock()-t1, 'secs'


    if info: print 'Number of clusters after reassignment', len(C_leafs)

    return C_leafs

    '''


