import numpy as np
cimport cython
from FusedTypes cimport floating, integral, number
cdef extern from "math.h":
    double floor (double x) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int ifloor(double x) nogil:
    return int(floor(x))


def quantize_positive_image(floating[:,:]v, int num_levels):
    cdef int nrows=v.shape[0]
    cdef int ncols=v.shape[1]
    cdef int npix = nrows*ncols
    cdef int i,j,l
    cdef floating epsilon, delta
    cdef int[:] hist=np.zeros(shape=(num_levels,), dtype=np.int32)
    cdef int[:,:] out=np.zeros(shape=(nrows, ncols,), dtype=np.int32)
    cdef floating[:] levels=np.zeros(shape=(num_levels,), dtype=cython.typeof(v[0,0]))
    num_levels-=1#zero is one of the levels
    if(num_levels<1):
        return None, None, None
    cdef double min_val = -1
    cdef double max_val = -1
    for i in range(nrows):
        for j in range(ncols):
            if(v[i,j]>0):
                if((min_val<0) or (v[i,j]<min_val)):
                    min_val=v[i,j]
                if(v[i,j]>max_val):
                    max_val=v[i,j]
    epsilon=1e-8
    delta=(max_val-min_val+epsilon)/num_levels
    if((num_levels<2) or (delta<epsilon)):#notice that we decreased num_levels, so levels[0..num_levels] are well defined
        for i in range(nrows):
            for j in range(ncols):
                if(v[i,j]>0):
                    out[i,j]=1
                else:
                    out[i,j]=0
                    hist[0]+=1
        levels[0]=0
        levels[1]=0.5*(min_val+max_val)
        hist[1]=npix-hist[0]
        return out, levels, hist
    levels[0]=0
    levels[1]=delta*0.5
    for i in range(2, 1+num_levels):
        levels[i]=levels[i-1]+delta
    for i in range(nrows):
        for j in range(ncols):
            if(v[i,j]>0):
                l=ifloor((v[i,j]-min_val)/delta)
                out[i,j]=l+1
                hist[l+1]+=1
            else:
                out[i,j]=0
                hist[0]+=1
    return out, levels, hist

def quantize_positive_volume(floating[:,:,:]v, int num_levels):
    cdef int nslices = v.shape[0]
    cdef int nrows=v.shape[1]
    cdef int ncols=v.shape[2]
    cdef int nvox = nrows*ncols*nslices
    cdef int i,j,k,l
    cdef floating epsilon, delta
    cdef int[:] hist=np.zeros(shape=(num_levels,), dtype=np.int32)
    cdef int[:,:,:] out=np.zeros(shape=(nslices, nrows, ncols), dtype=np.int32)
    cdef floating[:] levels=np.zeros(shape=(num_levels,), dtype=cython.typeof(v[0,0,0]))
    num_levels-=1#zero is one of the levels
    if(num_levels<1):
        return None, None, None
    cdef double min_val = -1
    cdef double max_val = -1
    for k in range(nslices):
        for i in range(nrows):
            for j in range(ncols):
                if(v[k,i,j]>0):
                    if((min_val<0) or (v[k,i,j]<min_val)):
                        min_val=v[k,i,j]
                    if(v[k,i,j]>max_val):
                        max_val=v[k,i,j]
    epsilon=1e-8
    delta=(max_val-min_val+epsilon)/num_levels
    if((num_levels<2) or (delta<epsilon)):#notice that we decreased num_levels, so levels[0..num_levels] are well defined
        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if(v[k,i,j]>0):
                        out[k,i,j]=1
                    else:
                        out[k,i,j]=0
                        hist[0]+=1
        levels[0]=0
        levels[1]=0.5*(min_val+max_val)
        hist[1]=nvox-hist[0]
        return out, levels, hist
    levels[0]=0
    levels[1]=delta*0.5
    for i in range(2, 1+num_levels):
        levels[i]=levels[i-1]+delta
    for k in range(nslices):
        for i in range(nrows):
            for j in range(ncols):
                if(v[k,i,j]>0):
                    l=ifloor((v[k,i,j]-min_val)/delta)
                    out[k,i,j]=l+1
                    hist[l+1]+=1
                else:
                    out[k,i,j]=0
                    hist[0]+=1
    return out, levels, hist

def compute_masked_image_class_stats(int[:,:] mask, floating[:,:] v, int numLabels, int[:,:] labels):
    cdef int nrows=v.shape[0]
    cdef int ncols=v.shape[1]
    cdef int i,j
    cdef floating INF64 = np.inf
    cdef int[:] counts=np.zeros(shape=(numLabels,), dtype=np.int32)
    cdef floating[:] means=np.zeros(shape=(numLabels,), dtype=cython.typeof(v[0,0]))
    cdef floating[:] variances=np.zeros(shape=(numLabels, ), dtype=cython.typeof(v[0,0]))
    for i in range(nrows):
        for j in range(ncols):
            if(mask[i,j]!=0):
                means[labels[i,j]]+=v[i,j]
                variances[labels[i,j]]+=v[i,j]**2
                counts[labels[i,j]]+=1
    for i in range(numLabels):
        if(counts[i]>0):
            means[i]/=counts[i]
        if(counts[i]>1):
            variances[i]=variances[i]/counts[i]-means[i]**2
        else:
            variances[i]=INF64
    return means, variances

def compute_masked_volume_class_stats(int[:,:,:] mask, floating[:,:,:] v, int numLabels, int[:,:,:] labels):
    cdef int nslices=v.shape[0]
    cdef int nrows=v.shape[1]
    cdef int ncols=v.shape[2]
    cdef int i,j,k
    cdef floating INF64 = np.inf
    cdef int[:] counts=np.zeros(shape=(numLabels,), dtype=np.int32)
    cdef floating[:] means=np.zeros(shape=(numLabels,), dtype=cython.typeof(v[0,0,0]))
    cdef floating[:] variances=np.zeros(shape=(numLabels, ), dtype=cython.typeof(v[0,0,0]))
    for k in range(nslices):
        for i in range(nrows):
            for j in range(ncols):
                if(mask[k,i,j]!=0):
                    means[labels[k,i,j]]+=v[k,i,j]
                    variances[labels[k,i,j]]+=v[k,i,j]**2
                    counts[labels[k,i,j]]+=1
    for i in range(numLabels):
        if(counts[i]>0):
            means[i]/=counts[i]
        if(counts[i]>1):
            variances[i]=variances[i]/counts[i]-means[i]**2
        else:
            variances[i]=INF64
    return means, variances
