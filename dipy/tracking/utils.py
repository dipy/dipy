from numpy import asarray, array, atleast_3d, ceil, concatenate, empty, \
        floor, mgrid, sqrt, zeros
from collections import defaultdict

def streamline_counts(streamlines, vol_dims, voxel_size):
    """Counts the number of unique streamlines that pass though each voxel

    Counts the number of points in each streamline that lie inside each voxel.

    Parameters:
    -----------
    streamlines : iterable
        A sequence of arrays, each streamline should a list of points in
        3-space, where (0,0,0) is one corner of the first voxel in image
        volume, voxel_size the diagonal corner of the same voxel and
        voxel_size*vol_dims is the diagonal corner of the image.
    vol_dims : 3 ints
        The shape of the volume to be returned containing the streamlines
        counts
    voxel_size : 3 floats
        The size of the voxels in the image volume

    Returns:
    --------
    image_volume : ndarray, shape=vol_dims
        The number of streamline points in each voxel of volume

    Raises:
    -------
    IndexError
        When the points of the streamlines lie outside of the return volume

    Note:
    -----
    A streamline can pass though a voxel even if one of the points of the
    streamline does not lie in the voxel. For example a step from [0,0,0] to
    [0,0,2] passes though [0,0,1]. Consider subsegmenting the streamlines when
    the edges of the voxels are smaller than the steps of the streamlines.

    """
    counts = zeros(vol_dims, 'int')
    for sl in streamlines:
        inds = floor(sl/voxel_size).astype('int')
        if inds.min() < 0:
            raise IndexError('streamline has negative values, these values ' +
                             'are outside the image volume')
        i, j, k = inds.T
        #this takes advantage of the fact that numpy's += operator only acts
        #once even if there are repeats in inds
        counts[i, j, k] += 1
    return counts

def streamline_mapping(streamlines, voxel_size):
    holder = defaultdict(list)
    for ii in xrange(len(streamlines)):
        sl = floor(streamlines[ii]/voxel_size).astype('int')
        for point in sl:
            point = tuple(point)
            inst = holder[point]
            if len(inst) < 1 or ii != inst[-1]:
                inst.append(ii)
    holder = dict(holder)
    return holder

def subsegment(streamlines, max_segment_length):
    for sl in streamlines:
        diff = (sl[1:] - sl[:-1])
        length = sqrt((diff*diff).sum(-1))
        num_segments = ceil(length/max_segment_length)
        if (num_segments == 1).all():
            yield sl
            continue

        new_sl = empty((num_segments.sum()+1, 3))
        new_sl[0] = sl[0]

        count = 1
        for ii in xrange(len(num_segments)):
            ns = num_segments[ii]
            if ns == 1:
                new_sl[count] = sl[ii+1]
                count += 1
            elif ns > 1:
                small_d = diff[ii]/ns
                point = sl[ii]
                for jj in xrange(ns):
                    point = point + small_d
                    new_sl[count] = point
                    count += 1
            elif ns == 0:
                pass
                #repeated point
            else:
                #this should never happen because ns should be a posative int
                assert(ns >= 0)
        yield new_sl

def seeds_from_mask(mask, density, voxel_size=(1,1,1)):
    """Takes a binary mask and returns seeds in voxels != 0

    places evanly spaced points in nonzero voxels of mask, spaces the points
    based on density. For example if density is [1, 2, 3], there will be 6
    points in each voxel, at x=.5, y=[.25, .75] and z=[.166, .5, .833].
    density=a is the same as density = [a, a, a]

    Examples:
    ---------
    >>> mask = zeros((3,3,3), 'bool')
    >>> mask[0,0,0] = 1
    >>> seeds_from_mask(mask, [1,1,1], [1,1,1])
    array([[ 0.5,  0.5,  0.5]])
    >>> seeds_from_mask(mask, [1,2,3], [1,1,1])
    array([[ 0.5       ,  0.25      ,  0.16666667],
           [ 0.5       ,  0.25      ,  0.5       ],
           [ 0.5       ,  0.25      ,  0.83333333],
           [ 0.5       ,  0.75      ,  0.16666667],
           [ 0.5       ,  0.75      ,  0.5       ],
           [ 0.5       ,  0.75      ,  0.83333333]])
    mask[0,1,2] = 1
    >>> seeds_from_mask(mask, [1,1,2], [1.1,1.1,2.5])
    array([[ 0.55 ,  0.55 ,  0.625],
           [ 0.55 ,  0.55 ,  1.875],
           [ 0.55 ,  1.65 ,  5.625],
           [ 0.55 ,  1.65 ,  6.875]])

    """
    mask = atleast_3d(mask)
    if mask.ndim != 3:
        raise ValueError('mask cannot be more than 3d')
    density = asarray(density, 'int')
    sp = empty(3)
    sp[:] = 1./density

    voxels = mask.nonzero()
    mg = mgrid[0:1:sp[0], 0:1:sp[1], 0:1:sp[2]]

    seeds = []
    for ii, jj, kk in zip(voxels, mg, sp):
        s = ii[:,None] + jj.ravel() + kk/2
        seeds.append(s.ravel())

    seeds = array(seeds).T
    seeds *= voxel_size
    return seeds

def target(streamlines, target_mask, voxel_size):
    """Retain tracks that pass though target_mask

    This function loops over the streamlines and returns streamlines that pass
    though target_mask.

    Parameters:
    -----------
    streamlines : iterable
        A squence of streamlines. Each streamline should be a (N, 3) array,
        where N is the length of the streamline.
    target_mask : array-like
        A mask used as a target
    voxel_size
        Size of the voxels in the target_mask

    Returns:
    streamlines : generator
        A sequence of streamlines that pass though target_mask
    
    Raises:
    -------
    IndexError
        When the points of the streamlines lie outside of the target_mask

    See Also:
    ---------
    streamline_counts

    """
    voxel_size = asarray(voxel_size, 'float')
    for sl in streamlines:
        ind = (sl/voxel_size).astype('int')
        if ind.min() < 0:
            raise IndexError('streamline has negative values, these values ' +
                             'are outside target_mask')
        i, j, k = ind.T
        try:
            state = target_mask[i, j, k]
        except IndexError:
            volume_size = tuple(voxel_size * target_mask.shape)
            raise IndexError('streamline has values greater than the size of ' +
                             'the target mask, ' + str(volume_size))
        if state.any():
            yield sl

def merge_streamlines(backward, forward):
    """Merges two sets of streamlines seeded at the same points

    Because the first point of each streamline pair should be the same, only
    one is kept

    Parameters:
    -----------
    backward : iterable
        a sequence of streamlines, will be returned in reversed order in the
        result
    forward : iterable
        a sequence of streamlines, will be returned in same order in the result

    Returns:
    --------
    streamlines : generator
        generator of merged streamlines

    Examples:
    ---------
    >>> A = [array([[0,0,0],[1,1,1],[2,2,2]])]
    >>> B = [array([[0,0,0],[-1,-1,-1],[-2,-2,-2]])]
    >>> list(merge_streamlines(A,B))
    [array([[ 2,  2,  2],
           [ 1,  1,  1],
           [ 0,  0,  0],
           [-1, -1, -1],
           [-2, -2, -2]])]
    >>> list(merge_streamlines(B,A))
    [array([[-2, -2, -2],
           [-1, -1, -1],
           [ 0,  0,  0],
           [ 1,  1,  1],
           [ 2,  2,  2]])]
    """
    B = iter(backward)
    F = iter(forward)
    while True:
        yield concatenate((B.next()[:0:-1], F.next()))
