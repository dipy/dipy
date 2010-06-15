''' A type of -*- python -*- file

Performance functions for dipy


'''
# cython: profile=True
# cython: embedsignature=True

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

# float 32 dtype for casting
cdef cnp.dtype f32_dt = np.dtype(np.float32)


#cdef inline int

@cython.boundscheck(False)
@cython.wraparound(False)
def peak_finding(odf,odf_faces):
    ''' Given a function on a sphere return the peaks' values and
    indices. Peaks are given in a descending order.    

    Parameters
    -----------

    odf: array, shape(N,) , function values on the sphere, where N is the number
    vertices on the sphere

    odf_faces: array, uint16, shape (M,3), faces of the triangulation on
    the sphere, where M is the number of faces on the sphere

    Returns
    --------
    peaks: array, peak values, shape (L,) where L can vary and is
    
    inds : array, indices of the peak values on the odf array, shape (L,)
    
    Notes:
    ------

    In a summary this function does the following

    Where the smallest odf values in the vertices of a face  put
    zeros on them. By doing that for the vertices  of all faces at the
    end you only have the peak points with nonzero values.

    For precalculated odf_faces look under
    dipy/core/matrices/evenly*.npz to use them try numpy.load()['faces']
    
    Examples:
    ---------

    Coming soon ..

    See Also:
    ---------
    ...   

    '''

    
    cdef:
        cnp.ndarray[cnp.uint16_t, ndim=2] cfaces = np.ascontiguousarray(odf_faces)
        cnp.ndarray[cnp.float64_t, ndim=1] codf = np.ascontiguousarray(odf)
        cnp.ndarray[cnp.float64_t, ndim=1] cpeak = np.ascontiguousarray(odf.copy())
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




        
            


            

        

        

    

    

    

        
