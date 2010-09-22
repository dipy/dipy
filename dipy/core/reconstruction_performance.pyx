''' A type of -*- python -*- file

Performance functions for dipy


'''
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


#@cython.boundscheck(False)
@cython.wraparound(False)
def pf_bago(odf, edges_on_sphere):

    cdef:
        cnp.ndarray[cnp.uint16_t, ndim=2] cedges = np.ascontiguousarray(edges_on_sphere)
        cnp.ndarray[cnp.float64_t, ndim=1] codf = np.ascontiguousarray(odf)
        cnp.ndarray[cnp.uint8_t, ndim=1] cpeak = np.ones(odf.shape, np.uint8)
        int i=0
        int lenedges = len(cedges)
        int find0,find1
        double odf0,odf1
    
    for i from 0 <= i < lenedges:

        find0 = cedges[i,0]
        find1 = cedges[i,1]

        odf0 = codf[find0]
        odf1 = codf[find1]

        if odf0 > odf1:
            cpeak[find1] = 0
        elif odf0 < odf1:
            cpeak[find1] = 0

    cpeak = np.array(cpeak)

    #find local maxima and give fiber orientation (inds) and magnitude
    #peaks in a descending order

    inds = cpeak.nonzero()[0]
    pinds = odf[inds].argsort()
    inds = inds[pinds][::-1]
    peaks = odf[inds]

    return peaks, inds

@cython.boundscheck(False)
@cython.wraparound(False)
def peak_finding(odf, odf_faces):
    ''' Hemisphere local maxima from sphere values and faces

    Return local maximum values and indices. Local maxima (peaks) are
    given in descending order.

    The sphere mesh, as defined by the vertex coordinates ``vertices``
    and the face indices ``odf_faces``, has to conform to the check in
    ``dipy.core.meshes.peak_finding_compatible``.  If it does not, then
    the results from peak finding routine will be unpredictable.

    Parameters
    ----------
    odf : (N,) array
       function values on the sphere, where N is the number of vertices
       on the sphere
    odf_faces : (M,3) array
       faces of the triangulation on the sphere, where M is the number
       of faces on the sphere

    Returns
    -------
    peaks : (L,) array, dtype np.float64
       peak values, shape (L,) where L can vary and is the number of
       local moximae (peaks).  Values are sorted, largest first
    inds : (L,) array, dtype np.uint16
       indices of the peak values on the `odf` array corresponding to
       the maxima in `peaks`
    
    Notes
    -----
    In summary this function does the following:

    Where the smallest odf values in the vertices of a face put
    zeros on them. By doing that for the vertices of all faces at the
    end you only have the peak points with nonzero values.

    For precalculated odf_faces look under
    dipy/core/matrices/evenly*.npz to use them try numpy.load()['faces']
    
    Examples
    --------
    Coming soon ..

    See also
    --------
    dipy.core.meshes
    '''
    cdef:
        cnp.ndarray[cnp.uint16_t, ndim=2] cfaces = np.ascontiguousarray(odf_faces)
        cnp.ndarray[cnp.float64_t, ndim=1] codf = np.ascontiguousarray(odf)
        cnp.ndarray[cnp.float64_t, ndim=1] cpeak = odf.copy()
        int i=0
        int test=0
        int lenfaces = len(cfaces)
        double odf0,odf1,odf2
        int find0,find1,find2
    
    for i in range(lenfaces):

        find0 = cfaces[i,0]
        find1 = cfaces[i,1]
        find2 = cfaces[i,2]        
        
        odf0=codf[find0]
        odf1=codf[find1]
        odf2=codf[find2]       

        if odf0 >= odf1 and odf0 >= odf2:
            cpeak[find1] = 0
            cpeak[find2] = 0
            continue

        if odf1 >= odf0 and odf1 >= odf2:
            cpeak[find0] = 0
            cpeak[find2] = 0
            continue

        if odf2 >= odf0 and odf2 >= odf1:
            cpeak[find0] = 0
            cpeak[find1] = 0
            continue

    peak=np.array(cpeak)
    peak=peak[0:len(peak)/2]

    #find local maxima and give fiber orientation (inds) and magnitude
    #peaks in a descending order

    inds=np.where(peak>0)[0]
    pinds=np.argsort(peak[inds])
    peaks=peak[inds[pinds]][::-1]

    return peaks, inds[pinds][::-1]


def argmax_from_adj(vals, vertex_inds, adj_inds):
    """ Indices of local maxima from `vals` given adjacent points

    Parameters
    ----------
    vals : (N,) array, dtype np.float64
       values at all vertices referred to in either of `vertex_inds` or
       `adj_inds`'
    vertex_inds : (V,) array
       indices into `vals` giving vertices that may be local maxima.
    adj_inds : sequence
       For every vertex in ``vertex_inds``, the indices (into `vals`) of
       the neighboring points

    Returns
    -------
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
    ----------
    adj_indices : sequence
       length V sequence of sequences, where sequence ``i`` contains the
       neighbors of a particular vertex.

    Returns
    -------
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
    ----------
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
    -------
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

cdef inline long offset(long *indices,long *strides,int lenind, int typesize) nogil:

    '''
    Parameters
    ----------
    indices: long * (int64 *), indices of the array which we want to
    find the offset
    strides: long * strides
    lenind: int, len(indices)
    typesize: int, number of bytes for data type e.g. if double is 8 if
    int32 is 4

    Returns:
    --------
    offset: integer, offset from 0 pointer in memory normalized by dtype
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
    indices: array, shape(N,), indices of the array which we want to
    find the offset
    strides: array, shape(N,), strides
    lenind: int, len(indices)
    typesize: int, number of bytes for data type e.g. if double is 8 if
    int32 is 4
    
    Returns:
    --------
    offset: integer, offset from 0 pointer in memory normalized by dtype
    
    Example
    -------
    >>> import numpy as np
    >>> from dipy.core.reconstruction_performance import ndarray_offset
    >>> I=np.array([1,1])
    >>> A=np.array([[1,0,0],[0,2,0],[0,0,3]])
    >>> S=np.array(A.strides)
    >>> ndarray_offset(I,S,2,8)
    4
    >>> A.ravel()[4]==A[1,1]
    True

    '''

    return offset(<long*>indices.data,<long*>strides.data,lenind, typesize)


def trilinear_interpolation(X):

    Xf=np.floor(X)        
    #d holds the distance from the (floor) corner of the voxel
    d=X-Xf
    #nd holds the distance from the opposite corner
    nd = 1-d
    #filling the weights
    W=np.array([[ nd[0] * nd[1] * nd[2] ],
                    [  d[0] * nd[1] * nd[2] ],
                    [ nd[0] *  d[1] * nd[2] ],
                    [ nd[0] * nd[1] *  d[2] ],
                    [  d[0] *  d[1] * nd[2] ],
                    [ nd[0] *  d[1] *  d[2] ],
                    [  d[0] * nd[1] *  d[2] ],
                    [  d[0] *  d[1] *  d[2] ]])

    IN=np.array([[ Xf[0]  , Xf[1]  , Xf[2] ],
                    [ Xf[0]+1 , Xf[1]  , Xf[2] ],
                    [ Xf[0]   , Xf[1]+1, Xf[2] ],
                    [ Xf[0]   , Xf[1]  , Xf[2]+1 ],
                    [ Xf[0]+1 , Xf[1]+1, Xf[2] ],
                    [ Xf[0]   , Xf[1]+1, Xf[2]+1 ],
                    [ Xf[0]+1 , Xf[1]  , Xf[2]+1 ],
                    [ Xf[0]+1 , Xf[1]+1, Xf[2]+1 ]])

    return W,IN.astype(np.int)

cdef inline void _trilinear_interpolation(double *X, double *W, long *IN) nogil:

    cdef double Xf[3],d[3],nd[3]
    cdef long i

    for i from 0<=i<3:
        
        Xf[i]=floor(X[i])
        d[i]=X[i]-Xf[i]
        nd[i]=1-d[i]

    #weights

    W[0]=nd[0] * nd[1] * nd[2]
    W[1]= d[0] * nd[1] * nd[2]
    W[2]=nd[0] *  d[1] * nd[2]
    W[3]=nd[0] * nd[1] *  d[2]
    W[4]= d[0] *  d[1] * nd[2]
    W[5]=nd[0] *  d[1] *  d[2]
    W[6]= d[0] * nd[1] *  d[2]
    W[7]= d[0] *  d[1] *  d[2]

    #indices

    IN[0] =<long>Xf[0];   IN[1] =<long>Xf[1];    IN[2] =<long>Xf[2]     
    IN[3] =<long>Xf[0]+1; IN[4] =<long>Xf[1];    IN[5] =<long>Xf[2]
    IN[6] =<long>Xf[0];   IN[7] =<long>Xf[1]+1;  IN[8] =<long>Xf[2]
    IN[9] =<long>Xf[0];   IN[10]=<long>Xf[1];    IN[11]=<long>Xf[2]+1    
    IN[12]=<long>Xf[0]+1; IN[13]=<long>Xf[1]+1;  IN[14]=<long>Xf[2]
    IN[15]=<long>Xf[0];   IN[16]=<long>Xf[1]+1;  IN[17]=<long>Xf[2]+1
    IN[18]=<long>Xf[0]+1; IN[19]=<long>Xf[1];    IN[20]=<long>Xf[2]+1
    IN[21]=<long>Xf[0]+1; IN[22]=<long>Xf[1]+1;  IN[23]=<long>Xf[2]+1
    

    return 

def nearest_direction(dx,qa,ind,odf_vertices,qa_thr=0.0245,ang_thr=60.):
    ''' Give the nearest direction to a point

        Parameters
        ----------        
        dx: array, shape(3,), as float, moving direction of the current
        tracking

        qa: array, shape(Np,), float, quantitative anisotropy matrix,
        where Np the number of peaks, found using self.Np

        ind: array, shape(Np,), float, index of the track orientation

        odf_vertices: array, shape(N,3), float, odf sampling directions

        qa_thr: float, threshold for QA, we want everything higher than
        this threshold 

        ang_thr: float, theshold, we only select fiber orientation with
        this range 

        Returns
        --------
        delta: bool, delta funtion, if 1 we give it weighting if it is 0
        we don't give any weighting

        direction: array, shape(3,), the fiber orientation to be
        consider in the interpolation

    '''

    max_dot=0
    max_doti=0
    angl = np.cos((np.pi*ang_thr)/180.) 
    if qa[0] <= qa_thr:
        return False, np.array([0,0,0])
        
    for i in range(len(qa)):
        if qa[i]<= qa_thr:
            break
        curr_dot = np.abs(np.dot(dx, odf_vertices[ind[i]]))
        if curr_dot > max_dot:
            max_dot = curr_dot
            max_doti = i
                
    if max_dot < angl :
        return False, np.array([0,0,0])

    if np.dot(dx,odf_vertices[ind[max_doti]]) < 0:
        return True, - odf_vertices[ind[max_doti]]
    else:
        return True,   odf_vertices[ind[max_doti]]
    
    
cdef inline long _nearest_direction(double* dx,double* qa,\
                                        double *ind, double *odf_vertices,\
                                        double qa_thr, double ang_thr,\
                                        double *direction) nogil:
    cdef:
        double max_dot=0
        double angl,curr_dot
        double odfv[3]
        long i,j,max_doti=0

    angl=cos((PI*ang_thr)/180.)
    if qa[0] <= qa_thr:
        return 0

    for i from 0<=i<5:#hardcoded 5? needs to change
        if qa[i]<=qa_thr:
            break
        for j from 0<=j<3:
            odfv[j]=odf_vertices[3*<long>ind[i]+j]
        curr_dot = dx[0]*odfv[0]+dx[1]*odfv[1]+dx[2]*odfv[2] 
        if curr_dot < 0: #abs
            curr_dot = -curr_dot
        if curr_dot > max_dot:
            max_dot=curr_dot
            max_doti = i

    if max_dot < angl:        
        return 0
    
    for j from 0<=j<3:
        odfv[j]=odf_vertices[3*<long>ind[max_doti]+j]
        
    if dx[0]*odfv[0]+dx[1]*odfv[1]+dx[2]*odfv[2] < 0:
        for j from 0<=j<3:
            direction[j]=-odf_vertices[3*<long>ind[max_doti]+j]
        return 1    
    else:
        for j from 0<=j<3:
            direction[j]= odf_vertices[3*<long>ind[max_doti]+j]
        return 1
    



        
def propagation_direction(point,dx,qa,ind,odf_vertices,qa_thr,ang_thr):
    ''' Find where you are moving next
    '''
    total_w = 0 # total weighting
    new_direction = np.array([0,0,0])
    w,index=trilinear_interpolation(point)

    #check if you are outside of the volume
    for i in range(3):
        if index[7][i] >= qa.shape[i] or index[0][i] < 0:
            return False, np.array([0,0,0])

    #calculate qa & ind of each of the 8 corners
    for m in range(8):
        x,y,z = index[m]
        qa_tmp = qa[x,y,z]
        ind_tmp = ind[x,y,z]
        delta,direction = nearest_direction(dx,qa_tmp,ind_tmp,odf_vertices,qa_thr,ang_thr)
        #print delta, direction
        if not delta:
            continue
        total_w += w[m]
        new_direction = new_direction +  w[m][0]*direction

    if total_w < .5: # termination criteria
        return False, np.array([0,0,0])

    return True, new_direction/np.sqrt(np.sum(new_direction**2))

@cython.cdivision(True)
cdef inline long _propagation_direction(double *point,double* dx,double* qa,\
                                double *ind, double *odf_vertices,\
                                double qa_thr, double ang_thr,\
                                long *qa_shape,long* strides,\
                                double *direction) nogil:
    cdef:
        double total_w=0,delta=0
        double new_direction[3]
        double w[8],qa_tmp[5],ind_tmp[5]
        long index[24],i,j,m,xyz[4]
        double normd
        
    #calculate qa & ind of each of the 8 neighboring voxels
    #to do that we use trilinear interpolation
    _trilinear_interpolation(point,<double *>w,<long *>index)
    
    #check if you are outside of the volume
    for i from 0<=i<3:
        new_direction[i]=0
        if index[7*3+i] >= qa_shape[i] or index[i] < 0:
            return 0

    for m from 0<=m<8:
        for i from 0<=i<3:
            xyz[i]=index[m*3+i]
        
        for j from 0<=j<5:#hardcoded needs to change
            xyz[3]=j
            off=offset(<long*>xyz,strides,4,8)
            qa_tmp[j]=qa[off]
            ind_tmp[j]=ind[off]
        delta=_nearest_direction(dx,qa_tmp,ind_tmp,odf_vertices,\
                                         qa_thr, ang_thr,direction)
        if delta==0:
            continue
        total_w+=w[m]
        for i from 0<=i<3:
            new_direction[i]+=w[m]*direction[i]

    if total_w < .5: #termination
        return 0

    normd=new_direction[0]**2+new_direction[1]**2+new_direction[2]**2
    normd=1/sqrt(normd)
    
    for i from 0<=i<3:
        direction[i]=new_direction[i]*normd
    
    return 1

        
 


    
def initial_direction(cnp.ndarray[double,ndim=1] seed,\
                          cnp.ndarray[double,ndim=4] qa,\
                          cnp.ndarray[double,ndim=4] ind,\
                          cnp.ndarray[double,ndim=2] odf_vertices,\
                          double qa_thr):
    ''' First direction that we get from a seeding point

    '''
    #very tricky/cool addition/flooring that helps create a valid
    #neighborhood (grid) for the trilinear interpolation to run smoothly
    seed+=0.5
    point=np.floor(seed)
    x,y,z = point
    qa_tmp=qa[x,y,z,0]#maximum qa
    ind_tmp=ind[x,y,z,0]#corresponing orientation indices for max qa
    print('qa_tmp_initial',qa_tmp)
    print('qa_thr_initial',qa_thr)
    if qa_tmp < qa_thr:
        return False, np.array([0,0,0])
    else:
        return True, odf_vertices[ind_tmp]

cdef inline long _initial_direction(double* seed,double *qa,\
                                        double* ind, double* odf_vertices,\
                                        double qa_thr, long* strides, int ref,\
                                        double* direction) nogil:
    cdef:
        long point[4],off
        long i
        double qa_tmp,ind_tmp

    #find the index for qa
    for i from 0<=i<3:
        point[i]=<long>floor(seed[i]+.5)
    point[3]=ref
    #find the offcet in memory to access the qa value
    off=offset(<long*>point,strides,4,8)    
    qa_tmp=qa[off]
    #print('qa_tmp  _initial',qa_tmp)
    #check for threshold
    if qa_tmp < qa_thr:
        return 0
    else:
        #find the correct direction from the indices
        ind_tmp=ind[off]
        #return initial direction through odf_vertices by ind
        for i from 0<=i<3:
            direction[i]=odf_vertices[3*<long>ind_tmp+i]
        return 1
        

def propagation(cnp.ndarray[double,ndim=1] seed,\
                    cnp.ndarray[double,ndim=4] qa,\
                    cnp.ndarray[double,ndim=4] ind,\
                    cnp.ndarray[double,ndim=2] odf_vertices,\
                    double qa_thr,double ang_thr,double step_sz):
    '''
    Parameters
    ----------
    seed: array, shape(3,), point where the tracking starts        
    qa: array, shape(Np,), float, quantitative anisotropy matrix,
    where Np the number of peaks, found using self.Np
    ind: array, shape(Np,), float, index of the track orientation        
                
    Returns
    -------
    d: bool, delta function result        
    idirection: array, shape(3,), index of the direction of the propagation

    '''
    cdef:
        double *ps=<double *>seed.data
        double *pqa=<double*>qa.data
        double *pin=<double*>ind.data
        double *pverts=<double*>odf_vertices.data
        long *pstr=<long *>qa.strides
        long *qa_shape=<long *>qa.shape
        long *pvstr=<long *>odf_vertices.strides
        long ref,d,i,j
        double direction[3],dx[3],idirection[3],ps2[3]
        double trajectory[30000]
        

    ref=0
    d=_initial_direction(ps,pqa,pin,pverts,qa_thr,pstr,ref,idirection)
    #print 'res',res, direction[0],direction[1],direction[2]

    #d2,idirection2=initial_direction(seed.copy(),qa,ind,odf_vertices,qa_thr)


    if d==0:
        return None

    #print('idirection',idirection[0],idirection[1],idirection[2])
    #print('idirection2',idirection2)
    
    for i from 0<=i<3:
        #store the initial direction
        dx[i]=idirection[i]
        #ps2 is for downwards and ps for upwards propagation
        ps2[i]=ps[i]
    
    point=seed.copy()
    track = []
    #print('point first',point)
    track.append(point.copy())
    #return np.array(track)

    while d:
       d= _propagation_direction(ps,dx,pqa,pin,pverts,qa_thr,\
                                   ang_thr,qa_shape,pstr,direction)

       #d2,direction2 = propagation_direction(point,idirection2,qa,ind,\
       #                                           odf_vertices,qa_thr,ang_thr)

       #print('while direction ',direction[0] ,direction[1] ,direction[2])
       #print('while direction2',direction2[0],direction2[1],direction2[2])
        
       
       if d==0:
           break
       for i from 0<=i<3:
           dx[i]=direction[i]
           ps[i]+=step_sz*dx[i]
           point[i]=ps[i]#to be changed
       #print('point up',point)        
       track.append(point.copy())
       
    d=1
    for i from 0<=i<3:
        dx[i]=-idirection[i]

    #track towards the opposite direction 
    while d:
        d= _propagation_direction(ps2,dx,pqa,pin,pverts,qa_thr,\
                                   ang_thr,qa_shape,pstr,direction)

                
        if d==0:
            break
        for i from 0<=i<3:
            dx[i]=direction[i]
            ps2[i]+=step_sz*dx[i]
            point[i]=ps2[i]#to be changed           
        #print('point down',point)               
        track.insert(0,point.copy())
    
    #print(np.array(track))
    return np.array(track)


    '''
    d,idirection=initial_direction(seed,qa,ind,odf_vertices,qa_thr)
    print d, idirection
    if not d:
        return None
        
    dx = idirection
    point = seed.copy()
    track = []
    track.append(point)

    #return np.array(track)    

    #track towards one direction 
    while d:
        d,dx = propagation_direction(point,dx,qa,ind,\
                                                  odf_vertices,qa_thr,ang_thr)
        if not d:
            break
        point = point + step_sz*dx
        track.append(point)

    d = True
    dx = - idirection
    point = seed
    #track towards the opposite direction
    while d:
        d,dx = propagation_direction(point,dx,qa,ind,\
                                         odf_vertices,qa_thr,ang_thr)
        if not d:
            break
        point = point + step_sz*dx
        track.insert(0,point)

    return np.array(track)

    '''



