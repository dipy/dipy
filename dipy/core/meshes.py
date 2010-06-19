''' Mesh analysis '''

import numpy as np

FLOAT64_EPS = np.finfo(np.float64).eps
FLOAT_TYPES = np.sctypes['float']


def hemisphere_vertinds(vertices,
                        hemisphere='z',
                        equator_thresh=None,
                        dist_thresh=None):
    """ Hemisphere vertex indices from sphere points `vertices` 

    Selects the vertices from a sphere that lie in one hemisphere.
    If there are pairs symmetric points on the equator, we return only
    one of each pair.

    Parameters
    ----------
    vertices : (N,3) array-like
       (x, y, z) Point coordinates of N vertices
    hemisphere : str, optional
       Which hemisphere to select.  Values of '-x', '-y', '-z' select,
       respectively negative x, y, and z hemispheres; 'x', 'y', 'z'
       select the positive x, y, and z hemispheres.  Default is 'z'
    equator_thresh : None or float, optional
       Threshold (+-0) to identify points as being on the equator of the
       sphere.   If None, generate a default based on the data type
    dist_thresh : None or float, optional
       For a vertex ``v`` on the equator, if there is a vertex
       ``v_dash`` in `vertices`, such that the Euclidean distance
       between ``v * -1`` and ``v_dash`` is <= `dist_thresh`, then ``v``
       is taken to be in the opposite hemisphere to ``v_dash``, and only
       one of (``v``, ``v_dash``) will appear in the output vertex
       indices `inds`. None results in threshold based on the input data
       type of ``vertices``
    
    Returns
    -------
    inds : (P,) array
       Indices into `vertices` giving points in hemisphere

    Notes
    -----
    We expect the sphere to be symmetric, and so there may well be
    points on the sphere equator that are both on the same diameter
    line.  The routine returns the first of the two points in the
    original order of `vertices`.
    """
    vertices = np.asarray(vertices)
    assert vertices.shape[1] == 3
    if len(hemisphere) == 2:
        sign, hemisphere = hemisphere
        if sign not in '+-':
            raise ValueError('Hemisphere sign must be + or -')
    else:
        sign = '+'
    try:
        coord = 'xyz'.index(hemisphere)
    except ValueError:
        raise ValueError('Hemisphere must be (+-) x, y or z')
    if equator_thresh is None or dist_thresh is None:
        if not vertices.dtype.type in FLOAT_TYPES:
            EPS = FLOAT64_EPS
        else:
            EPS = np.finfo(vertices.dtype.type).eps
        if equator_thresh is None:
            equator_thresh = EPS * 10
        if dist_thresh is None:
            equator_thresh = EPS * 20
    if sign == '+':
        inds = vertices[:,coord] > -equator_thresh
    else:
        inds = vertices[:,coord] < equator_thresh
    # find equator points
    return np.nonzero(inds)[0]

    
def vertinds_to_neighbors(vertex_inds, faces):
    """ Return indices of neightbors of vertices given `faces`

    Parameters
    ----------
    vertex_inds : (N,) array-like
       indices of vertices
    faces : (F, 3) array-like
       Faces given by indices of vertices for each of ``F`` faces

    Returns
    -------
    neighbors : (N, B)
       For each ``N`` vertex indicated by `vertex_inds`, the vertex
       indices that are neighbors according to the graph given by
       `faces`.  For icosohedral meshes, ``B`` will be 6.
    """
    pass


def mesh_maximae(vals, vertex_inds, adj_inds):
    """ Indices of local maximae from `vals` given adjacent points

    Parameters
    ----------
    vals : (N,) array-like
       values at all vertices referred to in either of `vertex_inds` or
       `adj_inds`'
    vertex_inds : None or (V,) array-like
       indices into `vals` giving vertices that may be local maximae.
       If None, then equivalent to ``np.arange(N)``
    adj_inds (V, 6) array-like
       For every vertex in ``vertex_inds``, the indices (into `vals`) of
       the 6 neighboring points

    Returns
    -------
    inds : (M,) array
       Indices into `vals` giving local maximae of vals, given topology
       from `adj_inds`, and restrictions from `vertex_inds`. 
    """
    vals = np.asarray(vals)
    if vertex_inds is None:
        vertex_inds = np.arange(vals.shape[0])
    else:
        vertex_inds = np.asarray(vertex_inds)
    adj_inds = np.asarray(adj_inds)
    
