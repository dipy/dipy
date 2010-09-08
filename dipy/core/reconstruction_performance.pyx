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

    IN=np.array([[ Xf[0]   , Xf[1]  , Xf[2] ],
                 [ Xf[0]+1 , Xf[1]  , Xf[2] ],
                 [ Xf[0]   , Xf[1]+1, Xf[2] ],
                 [ Xf[0]   , Xf[1]  , Xf[2]+1 ],
                 [ Xf[0]+1 , Xf[1]+1, Xf[2] ],
                 [ Xf[0]   , Xf[1]+1, Xf[2]+1 ],
                 [ Xf[0]+1 , Xf[1]  , Xf[2]+1 ],
                 [ Xf[0]+1 , Xf[1]+1, Xf[2]+1 ]])

    return W,IN.astype(np.int)


cdef inline void ctrilinear_interpolation(double * X, double W[8], double IN[8][3]):

    cdef int i,j
    cdef double Xf[3],d[3],nd[3]
    cdef double f0,f1,f2
    #cdef double W[8],IN[8][3]


    for i in range(3):
        Xf[i]=floor(X[i])
        d[i]=X[i]-Xf[i]#d holds the distance from the (floor) corner of the voxel
        nd[i]=1-d[i]#nd holds the distance from the opposite corner

    f0=Xf[0]
    f1=Xf[1]
    f2=Xf[2]
    
    #fill the weights
    W[0]=nd[0] * nd[1] * nd[2]
    W[1]= d[0] * nd[1] * nd[2]
    W[2]=nd[0] *  d[1] * nd[2]
    W[3]=nd[0] * nd[1] *  d[2]
    W[4]= d[0] *  d[1] * nd[2]
    W[5]=nd[0] *  d[1] *  d[2]
    W[6]= d[0] * nd[1] *  d[2]
    W[7]= d[0] *  d[1] *  d[2]                  

    #fill the indices

    IN[0][0]=f0  ; IN[0][1]=f1  ;  IN[0][2]=f2                            
    IN[1][0]=f0+1; IN[1][1]=f1  ;  IN[1][2]=f2
    IN[2][0]=f0  ; IN[2][1]=f1+1;  IN[2][2]=f2                           
    IN[3][0]=f0  ; IN[3][1]=f1  ;  IN[3][2]=f2+1
    IN[4][0]=f0+1; IN[4][1]=f1+1;  IN[4][2]=f2                              
    IN[5][0]=f0  ; IN[5][1]=f1+1;  IN[5][2]=f2+1
    IN[6][0]=f0+1; IN[6][1]=f1  ;  IN[6][2]=f2+1
    IN[7][0]=f0+1; IN[7][1]=f1+1;  IN[7][2]=f2+1



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

    Np=qa.shape[-1]
    #Np=65
    

    max_dot=0
    max_doti=0
    angl = np.cos((np.pi*ang_thr)/180.) 
    if qa[0] <= qa_thr:
        return False, np.array([0,0,0])
        
    for i in range(Np):
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

    
#cdef cnearest_direction(<double *>dx,<double *>qa, <double *>ind, odf_vertices,qa_thr,ang_thr):
    


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
        new_direction = new_direction + w[m][0]*direction

    if total_w < .5: # termination criteria
        return False, np.array([0,0,0])

    return True, new_direction/np.sqrt(np.sum(new_direction**2))


    

def initial_direction(seed,qa,ref,ind,odf_vertices,qa_thr):
    ''' First direction that we get from a seeding point

    '''
    #very tricky/cool addition/flooring that helps create a valid
    #neighborhood (grid) for the trilinear interpolation to run smoothly
    seed+=0.5
    point=np.floor(seed)
    x,y,z = point

    #check if you are outside of the volume
    for i in range(3):
        if point[i] >= qa.shape[i] or point[0] < 0:
            return False, np.array([0,0,0])
    
    qa_tmp=qa[x,y,z,ref]#maximum qa
    ind_tmp=ind[x,y,z,ref]#corresponing orientation indices for max qa

    if qa_tmp < qa_thr:
        return False, np.array([0,0,0])
    else:
        return True, odf_vertices[ind_tmp]


#@cython.boundscheck(False)
#@cython.wraparound(False)
def propagation(seed,qa,ref,ind,odf_vertices,qa_thr,ang_thr,step_sz):
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

    cdef :
        cnp.ndarray[cnp.float64_t, ndim=4] cqa = np.ascontiguousarray(qa)
        cnp.ndarray[cnp.float64_t, ndim=4] cind = np.ascontiguousarray(ind)
        cnp.ndarray[cnp.float64_t, ndim=2] codf_vertices = np.ascontiguousarray(odf_vertices)
        cnp.ndarray[cnp.float64_t, ndim=2] track = np.zeros((500,3))
        double *cseed=asdp(seed)        
        double cqshape[4]
        double cpoint[3]
        double dx[3]
        double qa_tmp
        int ind_tmp
        double qa_t[10] # carefull up to 10 peaks allowed
        int ind_t[10] #
        double idirection[3],new_direction[3],direction[3]
        int i,j,m
        int n_points=0
        int up_range=0
        int down_range=0
        int delta=0
        double W[8]
        double IN[8][3]
        double total_w=0
        
        int x,y,z
        int Np
        int middle=0#250
        double maxdot, maxdoti
        
        cdef double Xf[3],d[3],nd[3]
        cdef double f0,f1,f2

    #number of allowed peaks    
    Np=qa.shape[-1]
    #print('Np',Np)

    #check if you are outside of the volume
    for i in range(3):
        cqshape[i]=<double>qa.shape[i]
        cpoint[i]=floor(cseed[i]+0.5)
        #print('cpoint_i',cpoint[i])
        #print('cqshape',cqshape[i])
        if cpoint[i] >= cqshape[i] or cpoint[i] < 0:
            return None
        
    #get peak and index of interest according to ref
    #ref=0 is max peak, ref=1 is second maximum etc.
    qa_tmp=cqa[<int>cpoint[0],<int>cpoint[1],<int>cpoint[2],ref]#maximum qa
    ind_tmp=cind[<int>cpoint[0],<int>cpoint[1],<int>cpoint[2],ref]#corresponing orientation indices for max qa

    #check qa threshold
    if qa_tmp < qa_thr:
        return None
    else:
        #if higher than threshold then get primary direction
        for i in range(3):
            idirection[i]=codf_vertices[ind_tmp,i]

    #copy and store first point
    for i in range(3):
        cpoint[i]=cseed[i] 
        track[middle,i]=cseed[i]
        #copy first_direction
        dx[i]=idirection[i]

    #increase counter for number of points in track
    n_points+=1
        
    #return track#[:n_points]            
    #return np.random.rand(10,3)

    #delta function is now 1
    delta=1
    
    #tracking towards one direction
    while delta==1:

        #here is propagation direction implemented
        total_w = 0 # total weighting        
        for i in range(3):
            new_direction[i]=0
            
        #ctrilinear_interpolation(cpoint,W,IN)
        #'''
        for i in range(3):
            Xf[i]=floor(cpoint[i])
            d[i]=cpoint[i]-Xf[i]#d holds the distance from the (floor) corner of the voxel
            nd[i]=1-d[i]#nd holds the distance from the opposite corner

        f0=Xf[0]
        f1=Xf[1]
        f2=Xf[2]
    
        #fill the weights
        W[0]=nd[0] * nd[1] * nd[2]
        W[1]= d[0] * nd[1] * nd[2]
        W[2]=nd[0] *  d[1] * nd[2]
        W[3]=nd[0] * nd[1] *  d[2]
        W[4]= d[0] *  d[1] * nd[2]
        W[5]=nd[0] *  d[1] *  d[2]
        W[6]= d[0] * nd[1] *  d[2]
        W[7]= d[0] *  d[1] *  d[2]                  

        #fill the indices

        IN[0][0]=f0  ; IN[0][1]=f1  ;  IN[0][2]=f2                            
        IN[1][0]=f0+1; IN[1][1]=f1  ;  IN[1][2]=f2
        IN[2][0]=f0  ; IN[2][1]=f1+1;  IN[2][2]=f2                           
        IN[3][0]=f0  ; IN[3][1]=f1  ;  IN[3][2]=f2+1
        IN[4][0]=f0+1; IN[4][1]=f1+1;  IN[4][2]=f2                              
        IN[5][0]=f0  ; IN[5][1]=f1+1;  IN[5][2]=f2+1
        IN[6][0]=f0+1; IN[6][1]=f1  ;  IN[6][2]=f2+1
        IN[7][0]=f0+1; IN[7][1]=f1+1;  IN[7][2]=f2+1
        #'''

        #check if you are outside of the volume
        for i in range(3):
            if IN[7][i] >= cqshape[i] or IN[0][i] < 0:
                #return None
                delta = 0
                break

        #print('1.')
        
        #make sure delta is true
        if delta==0:
            break
        
        #print('2.')
        #calculate qa & ind of each of the 8 corners
        for m in range(8):
            
            x = <int>IN[m][0]
            y = <int>IN[m][1]
            z = <int>IN[m][2]

            #print(x,y,z)
            
            for i in range(Np):                
                qa_t[i] = cqa[x,y,z,i]
                ind_t[i] = <int>cind[x,y,z,i]

            #here is nearest_direction implemented
            #delta,direction = nearest_direction(dx,qa_tmp,ind_tmp,odf_vertices,qa_thr,ang_thr)
            max_dot=0
            max_doti=0
            #angl = np.cos((np.pi*ang_thr)/180.)
            angl = cos((3.1415926535897931*ang_thr)/180.)

            #check threshold for maximum peak
            if qa_t[0] <= qa_thr:
                #return None
                delta=0
                #break
                continue
            
            #print('3.')
            
            for i in range(Np):
                if qa_t[i]<= qa_thr:
                    break
                #curr_dot = np.abs(np.dot(dx, odf_vertices[ind[i]]))
                curr_dot = dx[0]*codf_vertices[ind_t[i],0]+dx[1]*codf_vertices[ind_t[i],1]+dx[2]*codf_vertices[ind_t[i],2]
                if curr_dot < 0:
                    curr_dot=-curr_dot                      

                if curr_dot > max_dot:
                    max_dot = curr_dot
                    max_doti = i
                
            if max_dot < angl :
                delta=0
                #return None
                #break
                continue

            if dx[0]*codf_vertices[ind_t[max_doti],0]+dx[1]*codf_vertices[ind_t[max_doti],1]+dx[2]*codf_vertices[ind_t[max_doti],2] < 0:
                delta=1
                for i in range(3):
                    direction[i]=-codf_vertices[ind_t[max_doti],i]
            else:
                delta=1
                for i in range(3):
                    direction[i]= codf_vertices[ind_t[max_doti],i]
                    
            '''
            if np.dot(dx,odf_vertices[ind[max_doti]]) < 0:
                return True, - odf_vertices[ind[max_doti]]
            else:
                return True,   odf_vertices[ind[max_doti]]
            '''
            
            #print delta, direction
            #if delta == 0:
            #    continue

            #sum oll the weights
            total_w += W[m]
            #update new direction
            for i in range(3):                
                new_direction[i] = new_direction[i] + W[m]*direction[i]

        if total_w < .5: # termination criteria
            delta=0
            break
            #return None

        #return True, new_direction/np.sqrt(np.sum(new_direction**2))
        delta=1
        new_dirnorm=sqrt(new_direction[0]**2+new_direction[1]**2+new_direction[2]**2)
        #normalize
        for i in range(3):
            dx[i]=new_direction[i]/new_dirnorm

        #check delta function
        #if delta==0:
        #    break
        
        #update the track points
        for i in range(3):
            cpoint[i]=cpoint[i]+step_sz*dx[i]
            track[middle+n_points,i]=cpoint[i]
            
        n_points+=1

        #print(n_points)
        


    #delta function is now 1
    delta=1
    up_range=n_points    

    return track[middle:middle+up_range]
  




    '''
    
    #d is the delta function 
    d,idirection=initial_direction(seed,qa,ref,ind,odf_vertices,qa_thr)
    #print d
    if not d:
        return None
        
    dx = idirection
    point = seed
    track = []
    track.append(point)
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

    




