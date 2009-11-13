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
    float abs( float num )
    double abs( double num )

    
cdef extern from "stdlib.h":
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *malloc(size_t size)
    void *calloc(size_t nelem, size_t elsize)
    void *realloc (void *ptr, size_t size)

#@cython.boundscheck(False)
#@cython.wraparound(False)

def cut_plane(tracks,ref,thr=20.0):
    
    ''' Extract divergence vectors and points of intersection 
    between planes normal to the reference fiber and other tracks
    
    Parameters:
    ----------------
    tracks: sequence 
        of tracks as arrays, shape (N1,3) .. (Nm,3) , dtype float32 (only float32)
    
    ref: array, shape (N,3)
        reference track
    
    thr: float
        distance threshold
        
    Returns:
    -----------
    
    hits: sequence
            list of points where the 
    
    divs : sequence  
        
    '''
    
    cdef long lent=len(tracks)
    cdef long i,j,k,
    cdef double alpha,beta,lrq,rcd,lhp
    cdef int cnthits=0
    
    cdef cnp.ndarray[cnp.float32_t, ndim=2] P
    cdef cnp.ndarray[cnp.float32_t, ndim=2] Q 
    cdef cnp.ndarray[cnp.float32_t, ndim=1] hit
    cdef cnp.ndarray[cnp.float32_t, ndim=1] divergence
        
    Hit=[]
    Div=[]
    
    hit = np.zeros((3,), dtype=np.float32)
    #hit = np.zeros((3,), dtype=np.float32)
    divergence = np.zeros((3,), dtype=np.float32)
    
    P=ref
    
    cdef int Plen=P.shape[0]
    cdef int Qlen
    
    #for every point along the reference
    for p from 0 <= p < Plen-1:
        
        along = P[p+1]-P[p]        
        normal=along/sqrt(along[0]*along[0]+along[1]*along[1]+along[2]*along[2])
        
        #hits=np.array([0,0,0],dtype='float32')
        #divs=np.array([0,0,0],dtype='float32')
        hits=np.array([0,0,0,0],dtype='float32')
        
        #for every track
        for t from 0 <= t < lent:        
            
            Q=tracks[t]
            Qlen=Q.shape[0]          
            
            #for every point on the track
            for q from 0<= q < Qlen-1:
                
                #only for close enough points please
                if sqrt((Q[q][0]-P[p][0])*(Q[q][0]-P[p][0])+(Q[q][1]-P[p][1])*(Q[q][1]-P[p][1])+(Q[q][2]-P[p][2])*(Q[q][2]-P[p][2])) < thr : 
                
                
                    #if np.inner(normal,q-p)*np.inner(normal,r-p) <= 0:
                    if (normal[0]*(Q[q][0]-P[p][0])+normal[1]*(Q[q][1]-P[p][1]) \
                        +normal[2]*(Q[q][2]-P[p][2])) * (normal[0]*(Q[q+1][0]-P[p][0])+normal[1]*(Q[q+1][1]-P[p][1]) \
                        +normal[2]*(Q[q+1][2]-P[p][2])) <=0 :
                        #if np.inner((r-q),normal) != 0:
                    
                        beta=(normal[0]*(Q[q+1][0]-Q[q][0])+normal[1]*(Q[q+1][1]-Q[q][1]) \
                            +normal[2]*(Q[q+1][2]-Q[q][2]))                                        
                            
                        if beta !=0 :
                        
                                #alpha = np.inner((p-q),normal)/np.inner((r-q),normal)
                                alpha = (normal[0]*(P[p][0]-Q[q][0])+normal[1]*(P[p][1]-Q[q][1]) \
                                        +normal[2]*(P[p][2]-Q[q][2]))/ \
                                        (normal[0]*(Q[q+1][0]-Q[q][0])+normal[1]*(Q[q+1][1]-Q[q][1]) \
                                        +normal[2]*(Q[q+1][2]-Q[q][2]))
                                        
                                #hit = q+alpha*(r-q)
                                hit[0] = Q[q][0]+alpha*(Q[q+1][0]-Q[q][0])
                                hit[1] = Q[q][1]+alpha*(Q[q+1][1]-Q[q][1])
                                hit[2] = Q[q][2]+alpha*(Q[q+1][2]-Q[q][2])
                               
                                #divergence =( (r-q)-np.inner(r-q,normal)*normal)/|r-q|
                                lrq = sqrt((Q[q][0]-Q[q+1][0])*(Q[q][0]-Q[q+1][0])+(Q[q][1]-Q[q+1][1])*(Q[q][1]-Q[q+1][1])+(Q[q][2]-Q[q+1][2])*(Q[q][2]-Q[q+1][2]))
                                divergence[0] = (Q[q+1][0]-Q[q][0] - beta*normal[0])/lrq
                                divergence[1] = (Q[q+1][1]-Q[q][1] - beta*normal[1])/lrq
                                divergence[2] = (Q[q+1][2]-Q[q][2] - beta*normal[2])/lrq
                                
                                #radial coefficient of divergence d.(h-p)/|h-p|
                                lhp = sqrt((hit[0]-P[p][0])*(hit[0]-P[p][0])+(hit[1]-P[p][1])*(hit[1]-P[p][1])+(hit[2]-P[p][2])*(hit[2]-P[p][2]))
                                
                                if lhp>0.0 :
                                    rcd=abs(divergence[0]*(hit[0]-P[p][0])+divergence[1]*(hit[1]-P[p][1])+divergence[2]*(hit[2]-P[p][2]))/lhp
                                else:
                                    rcd=0.0
                                        
                                #add points
                                #divs=np.vstack( (divs, np.array([divergence[0], divergence[1], divergence[2] ])) )                                
                                #hits=np.vstack( (hits, np.array([hit[0], hit[1], hit[2] ])) )
                                                                
                                hits=np.vstack( (hits, np.array([hit[0], hit[1], hit[2],rcd ])) )
                                
        
        Hit.append(hits[1:])
        #Div.append(divs[1:])
        
    return Hit
            
def cut_planeV2(tracks,ref):
    
    ''' Extract divergence vectors and points of intersection 
    between planes normal to the reference fiber and other tracks
    
    Version 2
    Parameters:
    ----------------
    tracks: sequence 
        of tracks as arrays, shape (N1,3) .. (Nm,3) , dtype float32 (only float32)
    
    ref: array, shape (N,3)
        reference track
        
    Returns:
    -----------
    
    hits: sequence
            list of points where each plane intersects a track
    
    divs : sequence  
            projection divergence of tracks on each plane 
    '''
    
    cdef long lent=len(tracks)
    cdef long i,j,k
    cdef double alpha,beta
    cdef int cnthits=0
    
    cdef double *hits_divs
    
    cdef cnp.ndarray[cnp.float32_t, ndim=2] P
    cdef cnp.ndarray[cnp.float32_t, ndim=2] Q 
    cdef cnp.ndarray[cnp.float32_t, ndim=1] hit
    cdef cnp.ndarray[cnp.float32_t, ndim=1] divergence
    
    #cdef cnp.ndarray[cnp.double_t, ndim=1]  tmp
    #tmp=np.ndarray(shape=(10000000),dtype=np.double)
    
        
    Hit=[]
    Div=[]
    Hit_Div=[]
    
    hit = np.zeros((3,), dtype=np.float32)
    divergence = np.zeros((3,), dtype=np.float32)
    
    P=ref
    
    cdef int Plen=P.shape[0]
    cdef int Qlen
    
    #for every point along the reference
    for p from 0 <= p < Plen-1:
        
        along = P[p+1]-P[p]        
        normal=along/sqrt(along[0]*along[0]+along[1]*along[1]+along[2]*along[2])
        
        #hits=np.array([0,0,0],dtype='float32')
        #divs=np.array([0,0,0],dtype='float32')
        
        hits_divs = <double *>malloc(0)
        cnthits=1
        #for every track
        for t from 0 <= t < lent:        
            
            Q=tracks[t]
            Qlen=Q.shape[0]          
            
            #for every point on the track
            for q from 0<= q < Qlen-1:
                
                #if np.inner(normal,q-p)*np.inner(normal,r-p) <= 0:
                if (normal[0]*(Q[q][0]-P[p][0])+normal[1]*(Q[q][1]-P[p][1]) \
                    +normal[2]*(Q[q][2]-P[p][2])) * (normal[0]*(Q[q+1][0]-P[p][0])+normal[1]*(Q[q+1][1]-P[p][1]) \
                    +normal[2]*(Q[q+1][2]-P[p][2])) <=0 :
                
                    #if np.inner((r-q),normal) != 0:
                    beta=(normal[0]*(Q[q+1][0]-Q[q][0])+normal[1]*(Q[q+1][1]-Q[q][1]) \
                        +normal[2]*(Q[q+1][2]-Q[q][2]))                                        
                        
                    if beta !=0 :
                    
                            #alpha = np.inner((p-q),normal)/np.inner((r-q),normal)
                            alpha = (normal[0]*(P[p][0]-Q[q][0])+normal[1]*(P[p][1]-Q[q][1]) \
                                    +normal[2]*(P[p][2]-Q[q][2]))/ \
                                    (normal[0]*(Q[q+1][0]-Q[q][0])+normal[1]*(Q[q+1][1]-Q[q][1]) \
                                    +normal[2]*(Q[q+1][2]-Q[q][2]))
                                    
                            #hit = q+alpha*(r-q)
                            hit[0] = Q[q][0]+alpha*(Q[q+1][0]-Q[q][0])
                            hit[1] = Q[q][1]+alpha*(Q[q+1][1]-Q[q][1])
                            hit[2] = Q[q][2]+alpha*(Q[q+1][2]-Q[q][2])
                           
                            #divergence = (r-q)-np.inner(r-q,normal)*normal
                            divergence[0] = Q[q+1][0]-Q[q][0] - beta*normal[0]
                            divergence[1] = Q[q+1][1]-Q[q][1] - beta*normal[1]
                            divergence[2] = Q[q+1][2]-Q[q][2] - beta*normal[2]
                            
                            #add points
                            
                            #divs=np.vstack( (divs, np.array([divergence[0], divergence[1], divergence[2] ])) )
                            #hits=np.vstack( (hits, np.array([hit[0], hit[1], hit[2] ])) )
                            
                            hits_divs=<double *>realloc(hits_divs,cnthits*6*sizeof(double))
                            
                            hits_divs[cnthits*6-1] =divergence[2]
                            hits_divs[cnthits*6-2] =divergence[1]
                            hits_divs[cnthits*6-3] =divergence[0]                            
                            hits_divs[cnthits*6-4] =hit[2]
                            hits_divs[cnthits*6-5] =hit[1]
                            hits_divs[cnthits*6-6] =hit[0]
                                           
                            cnthits+=1
                                            
                                            
        #tmp.data=hits_divs
        
        #print tmp.shape
        
        #hits=tmp[].copy()
                                            
        #free(hits_divs)
        
        #Hit.append(hits[1:])
        #Div.append(divs[1:])
        
        #Hit_Div.append(tmp)
        
    return Hit_Div


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
        index of the most similar track in tracks. This can be used as a reference track for a bundle.
    s : array, shape (len(tracks),)
        similarities between tracks[si] and the rest of the tracks in the bundle
    
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
    
    #print(si,tracks[0].dtype)

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
        return (sumi+sumj)/2.0, np.min(sumi,sumj), np.max(sumi,sumj)
    elif metric=='avg':
        return (sumi+sumj)/2.0
    elif metric=='min':            
        return np.min(sumi,sumj)
    elif metric =='max':
        return np.max(sumi,sumj)
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

    
