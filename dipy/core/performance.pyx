''' A type of -*- python -*- file

Performance functions for dipy

'''
cimport cython

import numpy as np
cimport numpy as cnp

from pyalloc cimport pyalloc_v


cdef extern from "math.h":
    double floor(double x)
    float sqrt(float x)
    float fabs(float x)
    
#cdef extern from "stdio.h":
#	void printf ( const char * format, ... )
    
cdef extern from "stdlib.h":
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *malloc(size_t size)
    void *calloc(size_t nelem, size_t elsize)
    void *realloc (void *ptr, size_t size)

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
        [array([[ 1.        ,  1.5       ,  0.        ,  0.70710683]], dtype=float32),
         array([[ 2.        ,  2.5       ,  0.        ,  0.70710677]], dtype=float32)]
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
        size_t lent=len(tracks)
        size_t i,j,k
        int si,m,n,lti,ltj,met
        double sumi, sumj, tmp, delta
        cnp.ndarray[cnp.float32_t, ndim=2] A
        cnp.ndarray[cnp.float32_t, ndim=2] B
        #lentp=lent*(lent-1)/2 # number of combinations
        double *mini, *minj
        object mini_str, minj_str
        cnp.ndarray[cnp.double_t, ndim=1] s
    if metric=='avg':
        met=0
    elif metric == 'min':
        met=1
    elif metric == 'max':
        met=2
    else:
        raise ValueError('Metric should be one of avg, min, max')
    s = np.zeros((lent,), dtype=np.double)
    for i from 0 <= i < lent-1:
        for j from i+1 <= j < lent:        
            lti=tracks[i].shape[0]
            ltj=tracks[j].shape[0]
            A=tracks[i]
            B=tracks[j]
            mini_str = pyalloc_v(ltj*sizeof(double), <void **>&mini)
            minj_str = pyalloc_v(lti*sizeof(double), <void **>&minj)
            for n from 0<= n < ltj:
                mini[n]=biggest_double
            for m from 0<= m < lti:
                minj[m]=biggest_double
            for m from 0<= m < lti:                
                for n from 0<= n < ltj:
                    delta=sqrt((A[m,0]-B[n,0])*(A[m,0]-B[n,0])+(A[m,1]-B[n,1])*(A[m,1]-B[n,1])+(A[m,2]-B[n,2])*(A[m,2]-B[n,2]))
                    if delta < mini[n]:
                        mini[n]=delta
                    if delta < minj[m]:
                        minj[m]=delta
            sumi=0
            sumj=0
            for m from 0<= m < lti:
                sumj+=minj[m]
            sumj=sumj/lti
            for n from 0<= n < ltj:
                sumi+=mini[n]
            sumi=sumi/ltj
            if met ==0:                
                tmp=(sumi+sumj)/2.0
            elif met ==1:        
                if sumi < sumj:
                    tmp=sumi
                else:
                    tmp=sumj
            elif met ==2:                
                if sumi > sumj:
                    tmp=sumi
                else:
                    tmp=sumj                    
            s[i]+=tmp
            s[j]+=tmp
    si = np.argmin(s)
    for j from 0 <= j < lent:
        s[j]=zhang_distances(tracks[si],tracks[j],metric)
    return si,s


def zhang_distances(xyz1,xyz2,metric='all'):
    
    ''' Calculating the distance between tracks xyz1 and xyz2 
        Based on the metrics in Zhang,  Correia,   Laidlaw 2008 
        http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=4479455
        which in turn are based on those of Corouge et al. 2004
        
        This function should return the same results with zhang_distances 
        from track_metrics but hopefully faster.
 
    Parameters:
    -----------
        xyz1 : array, shape (N1,3), dtype float32
        xyz2 : array, shape (N2,3), dtype float32
        arrays representing x,y,z of the N1 and N2 points  of two tracks
    
    Returns:
    --------
        avg_mcd: float
                    average_mean_closest_distance
        min_mcd: float
                    minimum_mean_closest_distance
        max_mcd: float
                    maximum_mean_closest_distance
                    
    Notes:
    --------
    
    Algorithmic description
    
    Lets say we have curves A and B
    
    for every point in A calculate the minimum distance from every point in B stored in minAB
    for every point in B calculate the minimum distance from every point in A stored in minBA
    find average of minAB stored as avg_minAB
    find average of minBA stored as avg_minBA
    
    if metic is 'avg' then return (avg_minAB + avg_minBA)/2.0
    if metic is 'min' then return min(avg_minAB,avg_minBA)
    if metic is 'max' then return max(avg_minAB,avg_minBA)
    
    
    '''
    
    DEF biggest_double = 1.79769e+308

    cdef int m,n,lti,ltj
    cdef double sumi, sumj,delta
    
    cdef cnp.ndarray[cnp.float32_t, ndim=2] A
    cdef cnp.ndarray[cnp.float32_t, ndim=2] B

    cdef double *mini
    cdef double *minj
    
    lti=xyz1.shape[0]
    ltj=xyz2.shape[0]
    
    A=xyz1
    B=xyz2    

    mini = <double *>malloc(ltj*sizeof(double))
    minj = <double *>malloc(lti*sizeof(double))
    
    for n from 0<= n < ltj:
        mini[n]=biggest_double
        
    for m from 0<= m < lti:
        minj[m]=biggest_double
        
    for m from 0<= m < lti:                
        for n from 0<= n < ltj:

            delta=sqrt((A[m,0]-B[n,0])*(A[m,0]-B[n,0])+(A[m,1]-B[n,1])*(A[m,1]-B[n,1])+(A[m,2]-B[n,2])*(A[m,2]-B[n,2]))
            
            if delta < mini[n]:
                mini[n]=delta
                
            if delta < minj[m]:
                minj[m]=delta
    
    sumi=0
    sumj=0
    
    for m from 0<= m < lti:
        sumj+=minj[m]
    sumj=sumj/lti
               
    for n from 0<= n < ltj:
        sumi+=mini[n]
    sumi=sumi/ltj

    free(mini)
    free(minj)
        
    if metric=='all':
        return (sumi+sumj)/2.0, np.min((sumi,sumj)), np.max((sumi,sumj))
    elif metric=='avg':
        return (sumi+sumj)/2.0
    elif metric=='min':            
        return np.min((sumi,sumj))
    elif metric =='max':
        return np.max((sumi,sumj))
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
    
    DEF biggest_double = 1.79769e+308

    cdef int m,n,lti,ltj
    cdef double min_i, min_j,delta
    
    cdef cnp.ndarray[cnp.float32_t, ndim=2] A
    cdef cnp.ndarray[cnp.float32_t, ndim=2] B

    cdef double *mini
    cdef double *minj
    
    lti=xyz1.shape[0]
    ltj=xyz2.shape[0]
    
    A=xyz1
    B=xyz2    

    mini = <double *>malloc(ltj*sizeof(double))
    minj = <double *>malloc(lti*sizeof(double))
    
    for n from 0<= n < ltj:
        mini[n]=biggest_double
        
    for m from 0<= m < lti:
        minj[m]=biggest_double
        
    for m from 0<= m < lti:                
        for n from 0<= n < ltj:

            delta=sqrt((A[m,0]-B[n,0])*(A[m,0]-B[n,0])+(A[m,1]-B[n,1])*(A[m,1]-B[n,1])+(A[m,2]-B[n,2])*(A[m,2]-B[n,2]))
            
            if delta < mini[n]:
                mini[n]=delta
                
            if delta < minj[m]:
                minj[m]=delta
    
    min_i=biggest_double
    min_j=biggest_double
    
    for m from 0<= m < lti:
        if min_j > minj[m]:
            min_j=minj[m]
    
               
    for n from 0<= n < ltj:
        if min_i > mini[n]:
            min_i =mini[n]

    free(mini)
    free(minj)
        
    return (min_i+min_j)/2.0

    
