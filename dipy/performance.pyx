''' A type of -*- python -*- file

Performance functions for dipy

'''
cimport cython
import numpy as np
cimport numpy as cnp
from dipy.core import track_metrics as tm


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
        return (sumi+sumj)/2.0, np.min(sumi,sumj), np.max(sumi,sumj)
    elif metric=='avg':
        return (sumi+sumj)/2.0
    elif metric=='min':            
        return np.min(sumi,sumj)
    elif metric =='max':
        return np.max(sumi,sumj)
    else :
        ValueError('Wrong argument for metric')
        

