''' Mesh analysis '''

import numpy as np


def hemisphere_neighbors(vertices, faces, dist_thresh=None):
    """ Hemisphere vertex indices and adjacencies from sphere

    Selects the vertices from a sphere that lie in one hemisphere, and
    return these and the indices of points surrounding each vertex.
    These provide inputs to routines to find local maximae 

    Parameters
    ----------
    vertices : (N,3) array-like
       (x, y, z) Point coordinates of N vertices
    faces : (F, 3) array-like
       One row per face.  Each row gives (a, b, c), where a, b, and c
       are indices into `vertices`, so defining the vertices connected
       by the face.
    dist_thresh : None or float, optional
       For a vertex ``v``, if there is a vertex ``v_dash`` in
       `vertices`, such that the Euclidean distance between ``v * -1``
       and ``v_dash`` is <= `dist_thresh`, then ``v`` is in the opposite
       hemisphere to ``v_dash``, and only one of (``v``, ``v_dash``)
       will appear in the output vertex indices `inds`. None results in
       threshold based on the input data type of ``vertices``
    
    Returns
    -------
    inds : (P,) array
       Indices into `vertices` giving points in hemisphere
    adj_inds : (P, 6) array
       For each index ``i`` in `inds`, the vertex indices of the 6
       points surrounding ``i``
    """
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)


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
    
