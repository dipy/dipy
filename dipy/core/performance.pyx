''' A type of -*- python -*- file

Performance functions for dipy

'''
cimport cython

import numpy as np
cimport numpy as cnp


cdef extern from "math.h" nogil:
    double floor(double x)
    float sqrt(float x)
    float fabs(float x)
    double log2(double x)
    
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


cdef inline float* as_float_ptr(cnp.ndarray pt):
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
    next_ref_p = as_float_ptr(ref32[0])
    for p_no in range(N_ref-1):
        # extract point to point vector into `along`
        this_ref_p = next_ref_p
        next_ref_p = as_float_ptr(ref32[p_no+1])
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
            next_trk_p = as_float_ptr(track[0])
            for q_no in range(N_track-1):
                # p = ref32[p_no]
                # q = track[q_no]
                # r = track[q_no+1]
                # float* versions of above: p == this_ref_p
                this_trk_p = next_trk_p # q
                next_trk_p = as_float_ptr(track[q_no+1]) # r
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
    
    is 'avg' then return (min_minAB + min_minBA)/2.0
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


def approximate_mdl_trajectory(xyz):
    
    pass

def lee_perpendicular_distance(start0, end0, start1, end1):
    ''' Based on Lee , Han & Whang SIGMOD07.
        Calculates perpendicular distance metric for the distance between two line segments
    
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

    if l1 > l0:
        
        s_tmp = start0
        e_tmp = end0        
        start0 = start1
        end0 = end1
        start1 = s_tmp
        end1 = e_tmp
        
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

    cdef:
        float l0,l1,ltmp,u1,u2,lperp1,lperp2         
        float *s_tmp,*e_tmp,k0[3],ps[3],pe[3],ps1[3],pe1[3],tmp[3]
               
    csub_3vecs(end0,start0,tmp)    
    l0 = cinner_3vecs(tmp,tmp)
    
    csub_3vecs(end1,start1,tmp)    
    l1 = cinner_3vecs(tmp, tmp)
    
    if l1 > l0:
        
        s_tmp = start0
        e_tmp = end0        
        start0 = start1
        end0 = end1
        start1 = s_tmp
        end1 = e_tmp
        
        ltmp=l0
        l0=l1
        l1=ltmp
                
    csub_3vecs(end0,start0,k0)
    
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

    if l_1 > l_0:
        s_tmp = start0
        e_tmp = end0
        start0 = start1
        end0 = end1
        start1 = s_tmp
        end1 = e_tmp
    
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

    cdef:
        float l0,l1,ltmp,cos_theta_squared         
        float *s_tmp,*e_tmp,k0[3],k1[3],tmp[3]
               
    csub_3vecs(end0,start0,tmp)    
    l0 = cinner_3vecs(tmp,tmp)
    
    csub_3vecs(end1,start1,tmp)    
    l1 = cinner_3vecs(tmp, tmp)
    
    if l1 > l0:
        
        s_tmp = start0
        e_tmp = end0        
        start0 = start1
        end0 = end1
        start1 = s_tmp
        end1 = e_tmp
        
        ltmp=l0
        l0=l1
        l1=ltmp
                
    csub_3vecs(end0,start0,k0)
    csub_3vecs(end1,start1,k1)
    ltmp=cinner_3vecs(k0,k1)
    
    cos_theta_squared = (ltmp*ltmp)/ (l0*l1)
    
    return sqrt((1-cos_theta_squared)*l1)

def approximate_trajectory_partitioning(xyz, alpha=1.):
    ''' Implementation of Lee et al Approximate Trajectory
        Partitioning Algorithm
    
    Parameters:
    ------------------
    xyz: array(N,3) 
        initial trajectory
    alpha: float
        smoothing parameter (>1 => smoother, <1 => rougher)
    
    Returns:
    ------------
    characteristic_points: list of M array(3,) points
        which can be turned into an array with np.asarray() 
    '''
    
    characteristic_points=[xyz[0]]
    start_index = 0
    length = 2
    while start_index+length < len(xyz):
        current_index = start_index+length
        cost_par = minimum_description_length_partitoned(xyz[start_index:current_index+1])
        cost_nopar = minimum_description_length_unpartitoned(xyz[start_index:current_index+1])
        if alpha*cost_par>cost_nopar:
 
            characteristic_points.append(xyz[current_index-1])
            start_index = current_index-1
            length = 2
        else:
            length+=1
 
    characteristic_points.append(xyz[-1])
    return np.array(characteristic_points)
                
#@cython.boundscheck(False)
def minimum_description_length_partitoned(xyz):
    
    # L(H)
    cdef double val=np.log2(np.sqrt(np.inner(xyz[-1]-xyz[0],xyz[-1]-xyz[0])))
    cdef:
        cnp.ndarray[cnp.float32_t, ndim=2] track 
        cdef cnp.ndarray[cnp.float32_t, ndim=1] fvec1,fvec2,fvec3,fvec4
        size_t t_len
    track = np.ascontiguousarray(xyz, dtype=f32_dt)
    t_len=len(track)
    
    # L(D|H) 
    cdef int i    
    
    for i in range(1, t_len-1):
 
        #val += log2(lee_perpendicular_distance(track[i],track[i+1],track[0],track[t_len-1]))
        
        fvec1 = as_float_3vec(track[i])
        fvec2 = as_float_3vec(track[i+1])
        fvec3 = as_float_3vec(track[0])
        fvec4 = as_float_3vec(track[t_len-1])
                
        val += log2(clee_perpendicular_distance(<float *>fvec1.data,<float *>fvec2.data,<float *>fvec3.data,<float *>fvec4.data))        
        val += log2(clee_angle_distance(<float *>fvec1.data,<float *>fvec2.data,<float *>fvec3.data,<float *>fvec4.data))
        
        
    #val+=np.sum(np.log2([lee_perpendicular_distance(xyz[j],xyz[j+1],xyz[0],xyz[-1]) for j in range(1,len(xyz)-1)]))
    #val+=np.sum(np.log2([lee_angle_distance(xyz[j],xyz[j+1],xyz[0],xyz[-1]) for j in range(1,len(xyz)-1)]))
    
    return val
    
def minimum_description_length_unpartitoned(xyz):
    '''
    Example:
    --------
    >>> xyz = np.array([[0,0,0],[2,2,0],[3,1,0],[4,2,0],[5,0,0]])
    >>> tm.minimum_description_length_unpartitoned(xyz) == np.sum(np.log2([8,2,2,5]))/2
    '''
    return np.sum(np.log2((np.diff(xyz, axis=0)**2).sum(axis=1)))/2
