''' A type of -*- python -*- file

Counting incidence of tracks in voxels of volume

'''
cimport cython
import numpy as np
cimport numpy as cnp
from dipy.core import track_metrics as tm
#from stdlib cimport *




cdef extern from "math.h":
    double floor(double x)
    float sqrt(float x)
    
cdef extern from "stdlib.h":
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *malloc(size_t size)
    void *calloc(size_t nelem, size_t elsize)

@cython.boundscheck(False)
@cython.wraparound(False)

def most_similar_track_zhang(tracks,metric='avg'):    
    ''' The purpose of this function is to implement a much faster version of 
    most_similar_track_zhang from dipy.core.track_metrics  as we implemented 
    from Zhang et. al 2008. 
    
    Parameters:
    ---------------
    tracks: sequence 
            of tracks as arrays, shape (N1,3) .. (Nm,3) , dtype float32 (only float32)
    metric: string
            'avg', 'min', 'max'
            
    Returns:
    ----------
    si : int
        index of the most similar track in tracks
    s : array, shape (len(tracks),)
        similarities between si and the rest of the tracks in the bundle
    
    Notes :
    ---------
    
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
    DEF biggest_double = 1.79769e+308

    cdef long lent=len(tracks)
    cdef long i,j,k
    cdef int si,m,n,lti,ltj,met
    cdef double sumi, sumj, tmp,delta
    
    cdef cnp.ndarray[cnp.float32_t, ndim=2] A
    cdef cnp.ndarray[cnp.float32_t, ndim=2] B
       
    #lentp=lent*(lent-1)/2 # number of combinations
    cdef double *mini
    cdef double *minj
   
    cdef cnp.ndarray[cnp.double_t, ndim=1] s
    
    if metric=='avg':
        met=0
    if metric == 'min':
        met=1
    if metric == 'max':
        met=2
    
    s = np.zeros((lent,), dtype=np.double)
    
    for i from 0 <= i < lent-1:
        for j from i+1 <= j < lent:        

            lti=tracks[i].shape[0]
            ltj=tracks[j].shape[0]
            
            A=tracks[i]
            B=tracks[j]
            
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
            
            if met ==0:                
                tmp=(sumi+sumj)/2.0
                
            if met ==1:        
                if sumi < sumj:
                    tmp=sumi
                else:
                    tmp=sumj
                    
            if met ==2:                
                if sumi > sumj:
                    tmp=sumi
                else:
                    tmp=sumj                    
            
            s[i]+=tmp
            s[j]+=tmp
            
    si = np.argmin(s)
    print(si,tracks[0].dtype)

    for j from 0 <= j < lent:
        s[j]=tm.zhang_distances(tracks[si],tracks[j],metric)

    return si,s

