# distutils: language = c++

import numpy as np
import cython
cimport numpy as np
from cython.parallel import prange

from libc.math cimport sqrt, pow, abs


ctypedef float[:] float_array_1d_t
ctypedef double[:] double_array_1d_t

ctypedef float[:, :] float_array_2d_t
ctypedef double[:, :] double_array_2d_t

ctypedef fused floating:
    float
    double

ctypedef fused floating_array_1d_t:
    float_array_1d_t
    double_array_1d_t

ctypedef fused floating_array_2d_t:
    float_array_2d_t
    double_array_2d_t


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _arclengths(floating_array_2d_t points, floating_array_1d_t out) nogil:
    cdef int i = 0
    out[0] = 0.0
    for i in range(1, points.shape[0]):
        out[i] = out[i-1] + sqrt(pow(points[i, 0] - points[i - 1, 0], 2.0) +
                                 pow(points[i, 1] - points[i - 1, 1], 2.0) +
                                 pow(points[i, 2] - points[i - 1, 2], 2.0))

cdef void _arclengths_float(float[:,:] points, float[:] out) nogil:
    _arclengths(points, out)

cdef void _arclengths_double(double[:,:] points, double[:] out) nogil:
    _arclengths(points, out)



#@cython.wraparound(False)
#@cython.cdivision(True)
#@cython.boundscheck(False)
#cdef void _extrap(double[:, :] xyz, double[:] cumlen, double distance, double[:] xyz2) nogil:

#    cdef int ind = 0

#    for ind in range(cumlen.shape[0]):
#        if cumlen[ind] > distance:
#            break

#    #cdef int ind = np.where((cumlen - distance) > 0)[0][0]
#    cdef double len0 = cumlen[ind - 1]
#    cdef double len1 = cumlen[ind]
#    cdef double Ds = distance - len0
#    cdef double Lambda = Ds / (len1 - len0)

#    cdef double x,y,z

#    xyz2[0] = Lambda * xyz[ind, 0] + (1 - Lambda) * xyz[ind - 1, 0]
#    xyz2[1] = Lambda * xyz[ind, 1] + (1 - Lambda) * xyz[ind - 1, 1]
#    xyz2[2] = Lambda * xyz[ind, 2] + (1 - Lambda) * xyz[ind - 1, 2]

#    return

#@cython.wraparound(False)
#@cython.cdivision(True)
#@cython.boundscheck(False)
#cdef void _resample(double[:,:] xyz, double[:] cumlen, double[:] distances, double[:,:] xyz2) nogil:
#    for i in range(distances.shape[0]):
#        _extrap(xyz, cumlen, distances[i], xyz2[i])

#def resample(p_xyz, n_pols=3):
#    cdef double[:,:] xyz = np.asarray(p_xyz)
#    cdef int n_pts = xyz.shape[0]
#    if n_pts == 0:
#        raise ValueError('xyz array cannot be empty')
#    if n_pts == 1:
#        return xyz.copy().squeeze()

#    cumlen = np.zeros(n_pts)
#    #get_cum_sum(xyz, cumlen[1:])
#    cdef double step = cumlen[-1] / (n_pols - 1)
#    if cumlen[-1] < step:
#        raise ValueError('Given number of points n_pols is incorrect. ')
#    if n_pols <= 2:
#        raise ValueError('Given number of points n_pols needs to be'
#                         ' higher than 2. ')

#    ar = np.arange(0, cumlen[-1], step)
#    if np.abs(ar[-1] - cumlen[-1]) < np.finfo('f4').eps:
#        ar = ar[:-1]

#    cdef double[:,:] xyz2 = np.zeros((len(ar), 3))
#    _resample(xyz, cumlen, ar, xyz2)

#    return np.vstack((np.array(xyz2), xyz[-1]))

@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cdef void _resample(floating[:,:] points, floating[:] arclengths, floating[:,:] out) nogil:

    _arclengths(points, arclengths)

    cdef double ratio
    cdef int N = points.shape[0]
    cdef int newN = out.shape[0]
    cdef double step = arclengths[N-1] / (newN-1)

    cdef double nextPoint = 0.0
    cdef unsigned int i = 0
    cdef unsigned int j = 0
    cdef unsigned int k = 0

    while nextPoint < arclengths[N-1]:
        if nextPoint == arclengths[k]:
            out[i,0] = points[j,0];
            out[i,1] = points[j,1];
            out[i,2] = points[j,2];

            nextPoint += step
            i += 1
            j += 1
            k += 1
        elif nextPoint < arclengths[k]:
            ratio = 1 - ((arclengths[k]-nextPoint) / (arclengths[k]-arclengths[k-1]))

            out[i,0]   = points[j-1,0] + ratio * (points[j,0] - points[j-1,0])
            out[i,1] = points[j-1,1] + ratio * (points[j,1] - points[j-1,1])
            out[i,2] = points[j-1,2] + ratio * (points[j,2] - points[j-1,2])

            nextPoint += step
            i += 1
        else:
            j += 1
            k += 1

    out[newN-1,0] = points[N-1,0]
    out[newN-1,1] = points[N-1,1]
    out[newN-1,2] = points[N-1,2]



#cdef void _downsample_float(float[:,:] points, float[:] arclengths, float[:,:] out) nogil:
#    _downsample(points, arclengths, out)

#cdef void _downsample_double(double[:,:] points, double[:] arclengths, double[:,:] out) nogil:
#    _downsample(points, arclengths, out)

def resample(points, nb_points=3):
    arclengths = np.empty(points.shape[0], dtype=points.dtype)
    resampled_points = np.empty((nb_points, points.shape[1]), dtype=points.dtype)

    if points.dtype == np.float32:
        _resample[float](points, arclengths, resampled_points)
    else:
        _resample[double](points, arclengths, resampled_points)

    return resampled_points
