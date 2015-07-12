# A type of -*- python -*- file
"""This module contains the parts of dipy.tracking.utils that need to be
implemented in cython.
"""
import cython

import numpy as np
cimport numpy as cnp
from ._utils import _mapping_to_voxel, _to_voxel_coordinates

from ..utils.six.moves import xrange

cdef extern from "dpy_math.h":
    double floor(double x)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(False)
def _voxel2streamline(sl,
                      cnp.ndarray[cnp.npy_intp, ndim=2] unique_idx):
    """
    Maps voxels to streamlines and streamlines to voxels, for setting up
    the LiFE equations matrix

    Parameters
    ----------
    sl : list
        A collection of streamlines, each n by 3, with n being the number of
        nodes in the fiber.

    unique_idx : array.
       The unique indices in the streamlines

    Returns
    -------
    v2f, v2fn : tuple of dicts

    The first dict in the tuple answers the question: Given a voxel (from
    the unique indices in this model), which fibers pass through it?

    The second answers the question: Given a streamline, for each voxel that
    this streamline passes through, which nodes of that streamline are in that
    voxel?
    """
    # Define local counters:
    cdef int s_idx, node_idx, voxel_id, ii
    cdef dict vox_dict = {}
    for ii in range(len(unique_idx)):
        vox = unique_idx[ii]
        vox_dict[vox[0], vox[1], vox[2]] = ii
    # Outputs are these dicts:
    cdef dict v2f = {}
    cdef dict v2fn = {}
    # In each fiber:
    for s_idx in range(len(sl)):
        sl_as_idx = np.array(sl[s_idx]).astype(int)
        v2fn[s_idx] = {}
        # In each voxel present in there:
        for node_idx in range(len(sl_as_idx)):
            node = sl_as_idx[node_idx]
            # What serial number is this voxel in the unique voxel indices:
            voxel_id = vox_dict[node[0], node[1], node[2]]
            # Add that combination to the dict:
            if voxel_id in v2f:
                if s_idx not in v2f[voxel_id]:
                    v2f[voxel_id].append(s_idx)
            else:
                v2f[voxel_id] = [s_idx]
            # All the nodes going through this voxel get its number:
            if voxel_id in v2fn[s_idx]:
                v2fn[s_idx][voxel_id].append(node_idx)
            else:
                v2fn[s_idx][voxel_id] = [node_idx]
    return v2f ,v2fn


def streamline_mapping(streamlines, voxel_size=None, affine=None,
                       mapping_as_streamlines=False):
    """Creates a mapping from voxel indices to streamlines.

    Returns a dictionary where each key is a 3d voxel index and the associated
    value is a list of the streamlines that pass through that voxel.

    Parameters
    ----------
    streamlines : sequence
        A sequence of streamlines.
    voxel_size : array_like (3,), optional
        The size of the voxels in the image volume. This is ignored if affine
        is set.
    affine : array_like (4, 4), optional
        The mapping from voxel coordinates to streamline coordinates. If
        neither `affine` or `voxel_size` is set, the streamline values are
        assumed to be in voxel coordinates. IE ``[0, 0, 0]`` is the center of
        the first voxel and the voxel size is ``[1, 1, 1]``.
    mapping_as_streamlines : bool, optional, False by default
        If True voxel indices map to lists of streamline objects. Otherwise
        voxel indices map to lists of integers.

    Returns
    -------
    mapping : defaultdict(list)
        A mapping from voxel indices to the streamlines that pass through that
        voxel.

    Examples
    --------
    >>> streamlines = [np.array([[0., 0., 0.],
    ...                          [1., 1., 1.],
    ...                          [2., 3., 4.]]),
    ...                np.array([[0., 0., 0.],
    ...                          [1., 2., 3.]])]
    >>> mapping = streamline_mapping(streamlines, (1, 1, 1))
    >>> mapping[0, 0, 0]
    [0, 1]
    >>> mapping[1, 1, 1]
    [0]
    >>> mapping[1, 2, 3]
    [1]
    >>> mapping.get((3, 2, 1), 'no streamlines')
    'no streamlines'
    >>> mapping = streamline_mapping(streamlines, (1, 1, 1),
    ...                              mapping_as_streamlines=True)
    >>> mapping[1, 2, 3][0] is streamlines[1]
    True

    """
    cdef:
        cnp.ndarray[cnp.int_t, ndim=2, mode='strided'] voxel_indices

    lin, offset = _mapping_to_voxel(affine, voxel_size)
    if mapping_as_streamlines:
        streamlines = list(streamlines)
    mapping = {}

    for i, sl in enumerate(streamlines):
        voxel_indices = _to_voxel_coordinates(sl, lin, offset)

        # Get the unique voxels every streamline passes though
        uniq_points = set()
        for j in range(voxel_indices.shape[0]):
            point = (voxel_indices[j, 0],
                     voxel_indices[j, 1],
                     voxel_indices[j, 2])
            uniq_points.add(point)

        # Add the index of this streamline for each uniq voxel
        for point in uniq_points:
            if point in mapping:
                mapping[point].append(i)
            else:
                mapping[point] = [i]

    # If mapping_as_streamlines replace ids with streamlines
    if mapping_as_streamlines:
        for key in mapping:
            mapping[key] = [streamlines[i] for i in mapping[key]]

    return mapping


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(False)
def track_counts(tracks, vol_dims, vox_sizes=(1,1,1), return_elements=True):
    ''' Counts of points in `tracks` that pass through voxels in volume

    We find whether a point passed through a track by rounding the mm
    point values to voxels.  For a track that passes through a voxel more
    than once, we only record counts and elements for the first point in
    the line that enters the voxel.

    Parameters
    ----------
    tracks : sequence
       sequence of T tracks.  One track is an ndarray of shape (N, 3), where N
       is the number of points in that track, and ``tracks[t][n]`` is the n-th
       point in the t-th track.  Points are of form x, y, z in *voxel mm*
       coordinates. That is, if ``i, j, k`` is the possibly non-integer voxel
       coordinate of the track point, and `vox_sizes` are 3 floats giving voxel
       sizes of dimensions 0, 1, 2 respectively, then the voxel mm coordinates
       ``x, y, z`` are simply ``i * vox_sizes[0], j * vox_sizes[1], k *
       vox_sizes[2]``.  This convention derives from trackviz.  To pass in
       tracks as voxel coordinates, just pass ``vox_sizes=(1, 1, 1)`` (see
       below).
    vol_dims : sequence length 3
       volume dimensions in voxels, x, y, z.
    vox_sizes : optional, sequence length 3
       voxel sizes in mm.  Default is (1,1,1)
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

    Examples
    --------
    Imagine you have a volume (voxel) space of dimension ``(10,20,30)``.
    Imagine you had voxel coordinate tracks in ``vs``.  To just fill an array
    with the counts of how many tracks pass through each voxel:

    >>> vox_track0 = np.array([[0, 0, 0], [1.1, 2.2, 3.3], [2.2, 4.4, 6.6]])
    >>> vox_track1 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])
    >>> vs = (vox_track0, vox_track1)
    >>> vox_dim = (10, 20, 30) # original voxel array size
    >>> tcs = track_counts(vs, vox_dim, (1, 1, 1), False)
    >>> tcs.shape
    (10, 20, 30)
    >>> tcs[0, 0, 0:4]
    array([2, 1, 1, 0])
    >>> tcs[1, 2, 3], tcs[2, 4, 7]
    (1, 1)

    You can also use the routine to count into larger-than-voxel boxes.  To do
    this, increase the voxel size and decrease the ``vox_dim`` accordingly:

    >>> tcs=track_counts(vs, (10/2., 20/2., 30/2.), (2,2,2), False)
    >>> tcs.shape
    (5, 10, 15)
    >>> tcs[1,1,2], tcs[1,2,3]
    (1, 1)
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
