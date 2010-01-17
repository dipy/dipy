''' A type of -*- python -*- file

Counting incidence of tracks in voxels of volume

'''
import numpy as np
cimport numpy as cnp

cdef extern from "math.h":
    double floor(double x)


def track_counts(tracks, vol_dims, vox_sizes, return_elements=True):
    ''' Counts of points in `tracks` that pass through voxels in volume

    We find whether a point passed through a track by rounding the mm
    point values to voxels.  For a track that passes through a voxel more
    than once, we only record counts and elements for the first point in
    the line that enters the voxel. 

    Parameters
    ----------
    tracks : sequence
       sequence of tracks.  Tracks are ndarrays of shape (N, 3), where N
       is the number of points in that track, and ``tracks[t][n]`` is
       the n-th point in the t-th track.  Points are of form x, y, z in
       *mm* coordinates.
    vol_dim : sequence length 3
       volume dimensions in voxels, x, y, z.
    vox_sizes : sequence length 3
       voxel sizes in mm
    return_elements : {True, False}, optional
       If True, also return object array with one list per voxel giving
       track indices and point indices passing through the voxel (see
       below)

    Returns
    -------
    tcs : ndarray shape `vol_dim`
       An array where entry ``tcs[x, y, z]`` is the number of tracks
       that passed through voxel at voxel coordinate x, y, z
    tes : ndarray dtype np.object, shape `vol_dim`
       If `return_elements` is True, we also return an object array with
       one object per voxel. The objects at each voxel are a list of
       integers, where the integers are the indices of the track that
       passed through the voxel.
    '''
    vol_dims = np.asarray(vol_dims).astype(np.int)
    vox_sizes = np.asarray(vox_sizes).astype(np.double)
    n_voxels = np.prod(vol_dims)
    # output track counts array, flattened
    cdef cnp.ndarray[cnp.int_t, ndim=1] tcs = \
        np.zeros((n_voxels,), dtype=np.int)
    # pointer to output track indices
    cdef cnp.npy_intp i
    if return_elements:
        el_inds = {}
    # cython numpy pointer to individual track array
    cdef cnp.ndarray[cnp.float_t, ndim=2] t
    # cython numpy pointer to point in track array
    cdef cnp.ndarray[cnp.float_t, ndim=1] in_pt
    # processed point
    cdef int out_pt[3]
    # various temporary loop and working variables
    cdef int tno, pno, cno
    cdef cnp.npy_intp el_no, v
    # fill native C arrays from inputs
    cdef int vd[3]
    cdef double vxs[3]
    for cno in range(3):
        vd[cno] = vol_dims[cno]
        vxs[cno] = vox_sizes[cno]
    # return_elements to C native
    cdef int ret_elf = <int>return_elements
    # x slice size (C array ordering)
    cdef cnp.npy_intp yz = vd[1] * vd[2]
    for tno in range(len(tracks)):
        t = tracks[tno].astype(np.float)
        # set to find unique voxel points in track
        in_inds = set()
        # the loop below is time-critical
        for pno in range(t.shape[0]):
            in_pt = t[pno]
            # Round to voxel coordinates, and set coordinates outside
            # volume to volume edges
            for cno in range(3):
                v = <int>floor(in_pt[cno] / vxs[cno] + 0.5)
                if v < 0:
                    v = 0
                elif v >= vd[cno]:
                    v = vd[cno]-1 # last index for this dimension
                out_pt[cno] = v
            # calculate element number in flattened tcs array
            el_no = out_pt[0] * yz + out_pt[1] * vd[2] + out_pt[2]
            # discard duplicates
            if el_no in in_inds:
                continue
            in_inds.add(el_no)
            # set elements into object array
            if ret_elf:
                key = (out_pt[0], out_pt[1], out_pt[2])
                val = tno
                if tcs[el_no]:
                    el_inds[key].append(val)
                else:
                    el_inds[key] = [val]
            # set value into counts
            tcs[el_no] += 1
    if ret_elf:
        return tcs.reshape(vol_dims), el_inds
    return tcs.reshape(vol_dims)


