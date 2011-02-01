# A type of -*- python -*- file
""" Track propagation performance functions
"""

# cython: profile=True
# cython: embedsignature=True

cimport cython

import numpy as np
cimport numpy as cnp

cdef extern from "math.h" nogil:
    double floor(double x)
    float sqrt(float x)
    float fabs(float x)
    double log2(double x)
    double cos(double x)
    double sin(double x)
    float acos(float x )   
    bint isnan(double x)
    double sqrt(double x)
    
    
DEF PI=3.1415926535897931
DEF PEAK_NO=5

# initialize numpy runtime
cnp.import_array()

#numpy pointers
cdef inline float* asfp(cnp.ndarray pt):
    return <float *>pt.data

cdef inline double* asdp(cnp.ndarray pt):
    return <double *>pt.data

@cython.cdivision(True)
cdef  long offset(long *indices,long *strides,int lenind, int typesize) nogil:

    ''' Very general way to access any element of any ndimensional numpy array
    using cython.
    
    Parameters
    ------------
    indices : long * (int64 *), indices of the array which we want to
    find the offset
    strides : long * strides
    lenind : int, len(indices)
    typesize : int, number of bytes for data type e.g. if double is 8 if
    int32 is 4

    Returns
    ----------
    offset : integer, offset from 0 pointer in memory normalized by dtype
    '''
 
    cdef int i
    cdef long summ=0
    for i from 0<=i<lenind:
        #print('st',strides[i],indices[i])
        summ+=strides[i]*indices[i]        
    summ/=<long>typesize
    return summ

def ndarray_offset(cnp.ndarray[long, ndim=1] indices, \
                 cnp.ndarray[long, ndim=1] strides,int lenind, int typesize):
    ''' find offset in an ndarray using strides

    Parameters
    ----------
    indices : array, shape(N,), indices of the array which we want to
    find the offset
    strides : array, shape(N,), strides
    lenind : int, len(indices)
    typesize : int, number of bytes for data type e.g. if double is 8 if
    int32 is 4
    
    Returns
    -------
    offset : integer, offset from 0 pointer in memory normalized by dtype
    
    Examples
    --------
    >>> import numpy as np
    >>> from dipy.tracking.propspeed import ndarray_offset
    >>> I=np.array([1,1])
    >>> A=np.array([[1,0,0],[0,2,0],[0,0,3]])
    >>> S=np.array(A.strides)
    >>> ndarray_offset(I,S,2,8)
    4
    >>> A.ravel()[4]==A[1,1]
    True
    '''
    return offset(<long*>indices.data,<long*>strides.data,lenind, typesize)

cdef  void _trilinear_interpolation(double *X, double *W, long *IN) nogil:

    ''' interpolate in 3d volumes given point X
    Returns
    -------
    W : weights
    IN : indices of the volume
    '''
    cdef double Xf[3],d[3],nd[3]
    cdef long i
    #define the rectangular box where every corner is a neighboring voxel (assuming center)
    #!!! this needs to change for the affine case
    for i from 0<=i<3:        
        Xf[i]=floor(X[i])
        d[i]=X[i]-Xf[i]
        nd[i]=1-d[i]
    #weights
    #the weights are actualy the volumes of the 8 smaller boxes that define the initial rectangular box
    #for more on trilinear have a look here
    #http://en.wikipedia.org/wiki/Trilinear_interpolation
    #http://local.wasp.uwa.edu.au/~pbourke/miscellaneous/interpolation/index.html
    W[0]=nd[0] * nd[1] * nd[2]
    W[1]= d[0] * nd[1] * nd[2]
    W[2]=nd[0] *  d[1] * nd[2]
    W[3]=nd[0] * nd[1] *  d[2]
    W[4]= d[0] *  d[1] * nd[2]
    W[5]=nd[0] *  d[1] *  d[2]
    W[6]= d[0] * nd[1] *  d[2]
    W[7]= d[0] *  d[1] *  d[2]
    #indices
    #the indices give you the indices of the neighboring voxels (the corners of the box) e.g. the qa coordinates
    IN[0] =<long>Xf[0];   IN[1] =<long>Xf[1];    IN[2] =<long>Xf[2]     
    IN[3] =<long>Xf[0]+1; IN[4] =<long>Xf[1];    IN[5] =<long>Xf[2]
    IN[6] =<long>Xf[0];   IN[7] =<long>Xf[1]+1;  IN[8] =<long>Xf[2]
    IN[9] =<long>Xf[0];   IN[10]=<long>Xf[1];    IN[11]=<long>Xf[2]+1    
    IN[12]=<long>Xf[0]+1; IN[13]=<long>Xf[1]+1;  IN[14]=<long>Xf[2]
    IN[15]=<long>Xf[0];   IN[16]=<long>Xf[1]+1;  IN[17]=<long>Xf[2]+1
    IN[18]=<long>Xf[0]+1; IN[19]=<long>Xf[1];    IN[20]=<long>Xf[2]+1
    IN[21]=<long>Xf[0]+1; IN[22]=<long>Xf[1]+1;  IN[23]=<long>Xf[2]+1

    return    
    
cdef  long _nearest_direction(double* dx,double* qa,\
                                        double *ind,long peaks,double *odf_vertices,\
                                        double qa_thr, double ang_thr,\
                                        double *direction) nogil:

    ''' Give the nearest direction to a point and also check for the threshold and the angle

        Parameters
        ------------        
        dx : array, shape(3,), as float, moving direction of the current
        tracking

        qa : array, shape(Np,), float, quantitative anisotropy matrix,
        where Np the number of peaks, found using self.Np

        ind : array, shape(Np,), float, index of the track orientation

        odf_vertices : array, shape(N,3), float, sampling directions on the sphere

        qa_thr : float, threshold for QA, we want everything higher than
        this threshold 

        ang_thr : float, theshold, we only select fiber orientation with
        this range 

        Returns
        --------
        delta : bool, delta funtion, if 1 we give it weighting if it is 0
        we don't give any weighting

        direction : array, shape(3,), the fiber orientation to be
        consider in the interpolation
    '''
    cdef:
        double max_dot=0
        double angl,curr_dot
        double odfv[3]
        long i,j,max_doti=0

    #calculate the cos with radians 
    angl=cos((PI*ang_thr)/180.)    
    #if the maximum peak is lower than the threshold then there is no point continuing tracking
    if qa[0] <= qa_thr:
        return 0
    #for all peaks find the minimum angle between odf_vertices and dx
    for i from 0<=i<peaks:
        #if the current peak is smaller than the threshold then jump out
        if qa[i]<=qa_thr:
            break
        #copy odf_vertices
        for j from 0<=j<3:
            odfv[j]=odf_vertices[3*<long>ind[i]+j]
        #calculate the absolute dot product between dx and odf_vertices
        curr_dot = dx[0]*odfv[0]+dx[1]*odfv[1]+dx[2]*odfv[2]         
        if curr_dot < 0: #abs check
            curr_dot = -curr_dot
        #maximum dot means minimum angle
        #store tha maximum dot and the corresponding index from the neighboring voxel in maxdoti
        if curr_dot > max_dot:
            max_dot=curr_dot
            max_doti = i
    #if maxdot smaller than our angular *dot* threshold stop tracking
    if max_dot < angl:        
        return 0       
    #copy the odf_vertices for the voxel qa indices which have the smaller angle
    for j from 0<=j<3:
        odfv[j]=odf_vertices[3*<long>ind[max_doti]+j]        
    #if the dot product is negative then return the opposite direction otherwise return the same direction
    if dx[0]*odfv[0]+dx[1]*odfv[1]+dx[2]*odfv[2] < 0:
        for j from 0<=j<3:
            direction[j]=-odf_vertices[3*<long>ind[max_doti]+j]
        return 1    
    else:
        for j from 0<=j<3:
            direction[j]= odf_vertices[3*<long>ind[max_doti]+j]
        return 1
           

@cython.cdivision(True)
cdef long _propagation_direction(double *point,double* dx,double* qa,\
                                double *ind, double *odf_vertices,\
                                double qa_thr, double ang_thr,\
                                long *qa_shape,long* strides,\
                                double *direction,double total_weight) nogil:
    cdef:
        double total_w=0 #total weighting useful for interpolation  
        double delta=0 #store delta function (stopping function) result
        double new_direction[3] #new propagation direction
        double w[8],qa_tmp[PEAK_NO],ind_tmp[PEAK_NO]
        long index[24],i,j,m,xyz[4]
        double normd
        long peaks=qa_shape[3]#number of allowed peaks e.g. for fa is 1 for gqi.qa is 5
        
    #calculate qa & ind of each of the 8 neighboring voxels
    #to do that we use trilinear interpolation and return the weights 
    #and the indices for the weights i.e. xyz in qa[x,y,z]
    _trilinear_interpolation(point,<double *>w,<long *>index)
    #check if you are outside of the volume
    for i from 0<=i<3:
        new_direction[i]=0
        if index[7*3+i] >= qa_shape[i] or index[i] < 0:
            return 0
    #for every weight sum the total weighting
    for m from 0<=m<8:
        for i from 0<=i<3:
            xyz[i]=index[m*3+i]
        #fill qa_tmp and ind_tmp 
        for j from 0<=j<peaks:
            xyz[3]=j
            off=offset(<long*>xyz,strides,4,8)
            qa_tmp[j]=qa[off]
            ind_tmp[j]=ind[off]            
        #return the nearest direction by searching in all peaks
        delta=_nearest_direction(dx,qa_tmp,ind_tmp,peaks,odf_vertices,\
                                         qa_thr, ang_thr,direction)
        #if delta is 0 then that means that there was no good direction (obeying the thresholds) 
        #from that neighboring voxel, so this voxel is not adding to the total weight
        if delta==0:
            continue
        #add in total
        total_w+=w[m]
        for i from 0<=i<3:
            new_direction[i]+=w[m]*direction[i]
    #if less than half the volume is time to stop propagating
    if total_w < total_weight: #termination
        return 0
    #all good return normalized weighted next direction
    normd=new_direction[0]**2+new_direction[1]**2+new_direction[2]**2
    normd=1/sqrt(normd)    
    for i from 0<=i<3:
        direction[i]=new_direction[i]*normd    
    return 1


cdef  long _initial_direction(double* seed,double *qa,\
                                        double* ind, double* odf_vertices,\
                                        double qa_thr, long* strides, long ref,\
                                        double* direction) nogil:
    ''' First direction that we get from a seeding point
    '''
    cdef:
        long point[4],off
        long i
        double qa_tmp,ind_tmp
    #very tricky/cool addition/flooring that helps create a valid
    #neighborhood (grid) for the trilinear interpolation to run smoothly
    #find the index for qa
    for i from 0<=i<3:
        point[i]=<long>floor(seed[i]+.5)
    point[3]=ref
    #find the offcet in memory to access the qa value
    off=offset(<long*>point,strides,4,8)    
    qa_tmp=qa[off] 
    #check for scalar threshold
    if qa_tmp < qa_thr:
        return 0
    else:
        #find the correct direction from the indices
        ind_tmp=ind[off] #similar to ind[point] in numpy syntax
        #return initial direction through odf_vertices by ind
        for i from 0<=i<3:
            direction[i]=odf_vertices[3*<long>ind_tmp+i]
        return 1
        

def eudx_both_directions(cnp.ndarray[double,ndim=1] seed,\
                    long ref,\
                    cnp.ndarray[double,ndim=4] qa,\
                    cnp.ndarray[double,ndim=4] ind,\
                    cnp.ndarray[double,ndim=2] odf_vertices,\
                    double qa_thr,double ang_thr,double step_sz,double total_weight):
    '''
    Parameters
    ------------
    seed : array, shape(3,), point where the tracking starts     
    ref : long int, which peak to follow first
    qa : array, shape(Np,), float, quantitative anisotropy matrix,
    where Np the number of peaks, found using self.Np
    ind : array, shape(Np,), float, index of the track orientation
    total_weight : double 
                
    Returns
    -------
    track : array, shape(N,3)

    '''
    cdef:
        double *ps=<double *>seed.data
        double *pqa=<double*>qa.data
        double *pin=<double*>ind.data
        double *pverts=<double*>odf_vertices.data
        long *pstr=<long *>qa.strides
        long *qa_shape=<long *>qa.shape
        long *pvstr=<long *>odf_vertices.strides
        long d,i,j
        double direction[3],dx[3],idirection[3],ps2[3],tmp,ftmp
    
    
    """
    #don't track seeds on the boundaries    
    for i from 0<=i<3:
        if seed[i] ==qa_shape[i]-1 or seed[i] == 0:
            return None
    """
    
    d=_initial_direction(ps,pqa,pin,pverts,qa_thr,pstr,ref,idirection)    
    if d==0:
        return None
    
    for i from 0<=i<3:
        #store the initial direction
        dx[i]=idirection[i]
        #ps2 is for downwards and ps for upwards propagation
        ps2[i]=ps[i]
    
    point=seed.copy()
    track = []
    track.append(point.copy())   

    #track towards one direction
    while d:
        d= _propagation_direction(ps,dx,pqa,pin,pverts,qa_thr,\
                                   ang_thr,qa_shape,pstr,direction,total_weight)
        if d==0:
            break
       
        #update the track
        for i from 0<=i<3:
            dx[i]=direction[i]
            
            #check for boundaries
            tmp=ps[i]+step_sz*dx[i]
            #ftmp=floor(tmp+.5)
            
            if ftmp > qa_shape[i]-1 or tmp < 0.:
                 d=0
                 break
            
            #propagate
            ps[i]=tmp           
            point[i]=ps[i]
        
        #print('point up',point)
        if d==1:
            track.append(point.copy())
        
       
    d=1
        
    for i from 0<=i<3:
        dx[i]=-idirection[i]

    #track towards the opposite direction 
    while d:
        d= _propagation_direction(ps2,dx,pqa,pin,pverts,qa_thr,\
                                   ang_thr,qa_shape,pstr,direction,total_weight)
        if d==0:
            break
        #update the track
        for i from 0<=i<3:
            dx[i]=direction[i]
            
            #check for boundaries
            tmp=ps2[i]+step_sz*dx[i]            
            #ftmp=floor(tmp+.5)            
            if tmp > qa_shape[i]-1 or tmp < 0.:
                 d=0
                 break

            #propagate
            ps2[i]=tmp        
            point[i]=ps2[i] #to be changed

        #add track point
        if d==1:               
            track.insert(0,point.copy())
       

    #prepare to return final track for the current seed
    tmp_track=np.array(track,dtype=np.float32)
    #some times one of the ends takes small negative values
    #needs to be investigated further

    """

    try:
        if tmp_track[0,0]<0 or tmp_track[0,1] or tmp_track[0,2]:
            tmp_track=np.delete(tmp_track,0,0)
    except:
        pass
    
    try:   
        if tmp_track[-1,0]<0 or tmp_track[-1,1] or tmp_track[-1,2]:
            tmp_track=np.delete(tmp_track,len(tmp_track)-1,0)
    except:
        pass

    """
    #return track for the current seed point and ref
    return tmp_track




