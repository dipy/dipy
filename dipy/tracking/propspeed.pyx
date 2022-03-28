# A type of -*- python -*- file
""" Track propagation performance functions
"""

# cython: profile=True
# cython: embedsignature=True

cimport cython

import numpy as np
cimport numpy as cnp

from dipy.core.interpolation cimport _trilinear_interpolation_iso, offset

cdef extern from "dpy_math.h" nogil:
    double floor(double x)
    float fabs(float x)
    double cos(double x)
    double sin(double x)
    float acos(float x )
    double sqrt(double x)
    double DPY_PI


DEF PEAK_NO=5

# initialize numpy runtime
cnp.import_array()


def ndarray_offset(cnp.ndarray[cnp.npy_intp, ndim=1] indices,
                   cnp.ndarray[cnp.npy_intp, ndim=1] strides,
                   int lenind,
                   int typesize):
    """ Find offset in an N-dimensional ndarray using strides

    Parameters
    ----------
    indices : array, npy_intp shape (N,)
        Indices of the array which we want to find the offset.
    strides : array, shape (N,)
        Strides of array.
    lenind : int
        len of the `indices` array.
    typesize : int
        Number of bytes for data type e.g. if 8 for double, 4 for int32

    Returns
    -------
    offset : integer
        Index position in flattened array

    Examples
    --------
    >>> import numpy as np
    >>> from dipy.tracking.propspeed import ndarray_offset
    >>> I=np.array([1,1])
    >>> A=np.array([[1,0,0],[0,2,0],[0,0,3]])
    >>> S=np.array(A.strides)
    >>> ndarray_offset(I,S,2,A.dtype.itemsize)
    4
    >>> A.ravel()[4]==A[1,1]
    True
    """
    if not cnp.PyArray_CHKFLAGS(indices, cnp.NPY_ARRAY_C_CONTIGUOUS):
        raise ValueError(u"indices is not C contiguous")
    if not cnp.PyArray_CHKFLAGS(strides, cnp.NPY_ARRAY_C_CONTIGUOUS):
        raise ValueError(u"strides is not C contiguous")
    return offset(<cnp.npy_intp*> cnp.PyArray_DATA(indices),
                  <cnp.npy_intp*> cnp.PyArray_DATA(strides),
                  lenind,
                  typesize)


cdef cnp.npy_intp _nearest_direction(double* dx,
                                     double* qa,
                                     double *ind,
                                     cnp.npy_intp peaks,
                                     double *odf_vertices,
                                     double qa_thr, double ang_thr,
                                     double *direction) nogil:
    """ Give the nearest direction to a point, checking threshold and angle

    Parameters
    ----------
    dx : double array shape (3,)
        Moving direction of the current tracking.
    qa : double array shape (Np,)
        Quantitative anisotropy matrix, where ``Np`` is the number of peaks.
    ind : array, float64 shape(x, y, z, Np)
        Index of the track orientation.
    peaks : npy_intp
    odf_vertices : double array shape (N, 3)
        Sampling directions on the sphere.
    qa_thr : float
        Threshold for QA, we want everything higher than this threshold.
    ang_thr : float
        Angle threshold, we only select fiber orientation within this range.
    direction : double array shape (3,)
        The fiber orientation to be considered in the interpolation.  The array
        gets modified in-place.

    Returns
    -------
    delta : bool
        Delta funtion: if 1 we give it weighting, if it is 0 we don't give any
        weighting.
    """
    cdef:
        double max_dot = 0
        double angl,curr_dot
        double odfv[3]
        cnp.npy_intp i, j, max_doti = 0

    # calculate the cos with radians
    angl = cos((DPY_PI * ang_thr) / 180.)
    # if the maximum peak is lower than the threshold then there is no point
    # continuing tracking
    if qa[0] <= qa_thr:
        return 0
    # for all peaks find the minimum angle between odf_vertices and dx
    for i from 0 <= i < peaks:
        # if the current peak is smaller than the threshold then jump out
        if qa[i] <= qa_thr:
            break
        # copy odf_vertices
        for j from 0 <= j < 3:
            odfv[j]=odf_vertices[3 * <cnp.npy_intp>ind[i] + j]
        # calculate the absolute dot product between dx and odf_vertices
        curr_dot = dx[0] * odfv[0] + dx[1] * odfv[1] + dx[2] * odfv[2]
        if curr_dot < 0: #abs check
            curr_dot = -curr_dot
        # maximum dot means minimum angle
        # store tha maximum dot and the corresponding index from the
        # neighboring voxel in maxdoti
        if curr_dot > max_dot:
            max_dot=curr_dot
            max_doti = i
    # if maxdot smaller than our angular *dot* threshold stop tracking
    if max_dot < angl:
        return 0
    # copy the odf_vertices for the voxel qa indices which have the smaller
    # angle
    for j from 0 <= j < 3:
        odfv[j] = odf_vertices[3 * <cnp.npy_intp>ind[max_doti] + j]
    # if the dot product is negative then return the opposite direction
    # otherwise return the same direction
    if dx[0] * odfv[0] + dx[1] * odfv[1] + dx[2] * odfv[2] < 0:
        for j from 0 <= j < 3:
            direction[j] = -odf_vertices[3 * <cnp.npy_intp>ind[max_doti] + j]
        return 1
    for j from 0 <= j < 3:
        direction[j]= odf_vertices[3 * <cnp.npy_intp>ind[max_doti] + j]
    return 1


@cython.cdivision(True)
cdef cnp.npy_intp _propagation_direction(double *point,
                                         double* dx,
                                         double* qa,
                                         double *ind,
                                         double *odf_vertices,
                                         double qa_thr,
                                         double ang_thr,
                                         cnp.npy_intp *qa_shape,
                                         cnp.npy_intp* strides,
                                         double *direction,
                                         double total_weight) nogil:
    cdef:
        double total_w = 0 # total weighting useful for interpolation
        double delta = 0 # store delta function (stopping function) result
        double new_direction[3] # new propagation direction
        double w[8]
        double qa_tmp[PEAK_NO]
        double ind_tmp[PEAK_NO]
        cnp.npy_intp index[24]
        cnp.npy_intp xyz[4]
        cnp.npy_intp i, j, m
        double normd
        # number of allowed peaks e.g. for fa is 1 for gqi.qa is 5
        cnp.npy_intp peaks = qa_shape[3]

    # Calculate qa & ind of each of the 8 neighboring voxels.
    # To do that we use trilinear interpolation and return the weights and the
    # indices for the weights i.e. xyz in qa[x,y,z]
    _trilinear_interpolation_iso(point, <double *> w, <cnp.npy_intp *> index)
    # check if you are outside of the volume
    for i from 0 <= i < 3:
        new_direction[i] = 0
        if index[7 * 3 + i] >= qa_shape[i] or index[i] < 0:
            return 0
    # for every weight sum the total weighting
    for m from 0 <= m < 8:
        for i from 0 <= i < 3:
            xyz[i]=index[m * 3 + i]
        # fill qa_tmp and ind_tmp
        for j from 0 <= j < peaks:
            xyz[3] = j
            off = offset(<cnp.npy_intp*> xyz, strides, 4, 8)
            qa_tmp[j] = qa[off]
            ind_tmp[j] = ind[off]
        # return the nearest direction by searching in all peaks
        delta=_nearest_direction(dx,
                                 qa_tmp,
                                 ind_tmp,
                                 peaks,
                                 odf_vertices,
                                 qa_thr,
                                 ang_thr,
                                 direction)
        # if delta is 0 then that means that there was no good direction
        # (obeying the thresholds) from that neighboring voxel, so this voxel
        # is not adding to the total weight
        if delta == 0:
            continue
        # add in total
        total_w += w[m]
        for i from 0 <= i < 3:
            new_direction[i] += w[m] * direction[i]
    # if less than half the volume is time to stop propagating
    if total_w < total_weight: # termination
        return 0
    # all good return normalized weighted next direction
    normd = new_direction[0]**2 + new_direction[1]**2 + new_direction[2]**2
    normd = 1 / sqrt(normd)
    for i from 0 <= i < 3:
        direction[i] = new_direction[i] * normd
    return 1


cdef cnp.npy_intp _initial_direction(double* seed,double *qa,
                                     double* ind, double* odf_vertices,
                                     double qa_thr,
                                     cnp.npy_intp* strides,
                                     cnp.npy_intp ref,
                                     double* direction) nogil:
    """ First direction that we get from a seeding point
    """
    cdef:
        cnp.npy_intp point[4]
        cnp.npy_intp off
        cnp.npy_intp i
        double qa_tmp,ind_tmp
    # Very tricky/cool addition/flooring that helps create a valid neighborhood
    # (grid) for the trilinear interpolation to run smoothly.
    # Find the index for qa
    for i from 0 <= i < 3:
        point[i] = <cnp.npy_intp>floor(seed[i] + .5)
    point[3] = ref
    # Find the offset in memory to access the qa value
    off = offset(<cnp.npy_intp*>point,strides, 4, 8)
    qa_tmp = qa[off]
    # Check for scalar threshold
    if qa_tmp < qa_thr:
        return 0
    # Find the correct direction from the indices
    ind_tmp = ind[off] # similar to ind[point] in numpy syntax
    # Return initial direction through odf_vertices by ind
    for i from 0 <= i < 3:
        direction[i] = odf_vertices[3 * <cnp.npy_intp>ind_tmp + i]
    return 1


def eudx_both_directions(cnp.ndarray[double, ndim=1] seed,
                         cnp.npy_intp ref,
                         cnp.ndarray[double, ndim=4] qa,
                         cnp.ndarray[double, ndim=4] ind,
                         cnp.ndarray[double, ndim=2] odf_vertices,
                         double qa_thr,
                         double ang_thr,
                         double step_sz,
                         double total_weight,
                         cnp.npy_intp max_points):
    """
    Parameters
    ----------
    seed : array, float64 shape (3,)
        Point where the tracking starts.
    ref : cnp.npy_intp int
        Index of peak to follow first.
    qa : array, float64 shape (X, Y, Z, Np)
        Anisotropy matrix, where ``Np`` is the number of maximum allowed peaks.
    ind : array, float64 shape(x, y, z, Np)
        Index of the track orientation.
    odf_vertices : double array shape (N, 3)
        Sampling directions on the sphere.
    qa_thr : float
        Threshold for QA, we want everything higher than this threshold.
    ang_thr : float
        Angle threshold, we only select fiber orientation within this range.
    step_sz : double
    total_weight : double
    max_points : cnp.npy_intp

    Returns
    -------
    track : array, shape (N,3)
    """
    cdef:
        double *ps = <double *> cnp.PyArray_DATA(seed)
        double *pqa = <double*> cnp.PyArray_DATA(qa)
        double *pin = <double*> cnp.PyArray_DATA(ind)
        double *pverts = <double*> cnp.PyArray_DATA(odf_vertices)
        cnp.npy_intp *pstr = <cnp.npy_intp *> cnp.PyArray_STRIDES(qa)
        cnp.npy_intp *qa_shape = <cnp.npy_intp *> cnp.PyArray_DIMS(qa)
        cnp.npy_intp *pvstr = <cnp.npy_intp *> cnp.PyArray_STRIDES(odf_vertices)
        cnp.npy_intp d, i, j, cnt
        double direction[3]
        double dx[3]
        double idirection[3]
        double ps2[3]
        double tmp, ftmp
    if not cnp.PyArray_CHKFLAGS(seed, cnp.NPY_ARRAY_C_CONTIGUOUS):
        raise ValueError(u"seed is not C contiguous")
    if not cnp.PyArray_CHKFLAGS(qa, cnp.NPY_ARRAY_C_CONTIGUOUS):
        raise ValueError(u"qa is not C contiguous")
    if not cnp.PyArray_CHKFLAGS(ind, cnp.NPY_ARRAY_C_CONTIGUOUS):
        raise ValueError(u"ind is not C contiguous")
    if not cnp.PyArray_CHKFLAGS(odf_vertices, cnp.NPY_ARRAY_C_CONTIGUOUS):
        raise ValueError(u"odf_vertices is not C contiguous")

    cnt = 0
    d = _initial_direction(ps, pqa, pin, pverts, qa_thr, pstr, ref, idirection)
    if d == 0:
        return None
    for i from 0 <= i < 3:
        # store the initial direction
        dx[i] = idirection[i]
        # ps2 is for downwards and ps for upwards propagation
        ps2[i] = ps[i]
    point = seed.copy()
    track = []
    track.append(point.copy())
    # track towards one direction
    while d:
        d = _propagation_direction(ps, dx, pqa, pin, pverts, qa_thr, ang_thr,
                                   qa_shape, pstr, direction, total_weight)
        if d == 0:
            break
        if cnt > max_points:
            break
        # update the track
        for i from 0 <= i < 3:
            dx[i] = direction[i]
            # check for boundaries
            tmp = ps[i] + step_sz * dx[i]
            if tmp > qa_shape[i] - 1 or tmp < 0.:
                 d = 0
                 break
            # propagate
            ps[i] = tmp
            point[i] = ps[i]

        if d == 1:
            track.append(point.copy())
            cnt += 1
    d = 1
    for i from 0 <= i < 3:
        dx[i] = -idirection[i]

    cnt = 0
    # track towards the opposite direction
    while d:
        d = _propagation_direction(ps2, dx, pqa, pin, pverts, qa_thr, ang_thr,
                                   qa_shape, pstr, direction, total_weight)
        if d == 0:
            break
        if cnt > max_points:
            break
        # update the track
        for i from 0 <= i < 3:
            dx[i] = direction[i]
            # check for boundaries
            tmp=ps2[i] + step_sz*dx[i]
            if tmp > qa_shape[i] - 1 or tmp < 0.:
                 d = 0
                 break
            # propagate
            ps2[i] = tmp
            point[i] = ps2[i] # to be changed
        # add track point
        if d == 1:
            track.insert(0, point.copy())
            cnt += 1
    # prepare to return final track for the current seed
    tmp_track = np.array(track, dtype=np.float32)

    # Sometimes one of the ends takes small negative values; needs to be
    # investigated further

    # Return track for the current seed point and ref
    return tmp_track
