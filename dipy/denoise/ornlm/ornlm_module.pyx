#distutils: language = c++
#distutils: sources = ornlm.cpp upfirdn.cpp
from cython.view cimport memoryview
from cython.view cimport array as cvarray
import numpy as np
cdef extern from "ornlm.h":
    void ornlm(double *ima, int ndim, int *dims, int v, int f, double h, double *fima)
cdef extern from "upfirdn.h":
    void firdn_matrix(double *F, int n, int m, double *h, int len, double *out)
    void upfir_matrix(double *F, int n, int m, double *h, int len, double *out)

cpdef ornlmpy(double[:, :, :] image, volumeLen, patchLen, h):
    cdef double[:,:,:] I=image.copy_fortran()
    cdef double[:,:,:] filtered=I.copy_fortran()
    cdef int[:] dims=cvarray((3,), itemsize=sizeof(int), format="i")
    dims[0]=I.shape[0]
    dims[1]=I.shape[1]
    dims[2]=I.shape[2]
    ornlm(&I[0,0,0], 3, &dims[0], volumeLen, patchLen, h, &filtered[0,0,0])
    return filtered

cpdef firdnpy(double[:,:] image, double[:] h):
    cdef double[:,:] I=image.copy()
    cdef double[:] kernel=h.copy()
    nrows=I.shape[0]
    ncols=I.shape[1]
    len=h.shape[0]
    cdef double[:,:] filtered=np.zeros(shape=((nrows+len)//2, ncols), dtype=np.double)
    
    firdn_matrix(&I[0,0], nrows, ncols, &kernel[0], len, &filtered[0,0])
    return filtered

cpdef upfirpy(double[:,:] image, double[:] h):
    cdef double[:,:] I=image.copy()
    cdef double[:] kernel=h.copy()
    nrows=I.shape[0]
    ncols=I.shape[1]
    len=h.shape[0]
    cdef double[:,:] filtered=np.zeros(shape=(2*nrows+len-2, ncols), dtype=np.double)
    
    upfir_matrix(&I[0,0], nrows, ncols, &kernel[0], len, &filtered[0,0])
    return filtered

