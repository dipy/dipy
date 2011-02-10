''' Mesh analysis '''

import numpy as np
from scipy import sparse

FLOAT64_EPS = np.finfo(np.float64).eps
FLOAT_TYPES = np.sctypes['float']


def sym_hemisphere(vertices,
                   hemisphere='z',
                   equator_thresh=None,
                   dist_thresh=None):
    """ Indices for hemisphere from an array of `vertices` on a sphere 

    Selects the vertices from a sphere that lie in one hemisphere.
    If there are pairs of symmetric points on the equator, we return only
    the first occurring of each pair.

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
       ``v``, not ``v_dash``, will appear in the output vertex indices
       `inds`. None results in a threshold based on the input data type
       of ``vertices``
    
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
            dist_thresh = EPS * 20
    # column with coordinates for selecting the hemisphere
    sel_col = vertices[:,coord]
    if sign == '+':
        inds = sel_col > -equator_thresh
    else:
        inds = sel_col < equator_thresh
    # find equator points
    eq_inds, = np.where(
        (sel_col < equator_thresh) & (sel_col > -equator_thresh))
    # eliminate later points that are symmetric on equator
    untested_inds = list(eq_inds)
    out_inds = []
    for ind in eq_inds:
        untested_inds.remove(ind)
        test_vert = vertices[ind,:] * -1
        test_dists = np.sum(
            (vertices[untested_inds,:] - test_vert)**2, axis=1)
        sym_inds, = np.where(test_dists < dist_thresh)
        for si in sym_inds:
            out_ind = untested_inds[si]
            untested_inds.remove(out_ind)
            out_inds.append(out_ind)
        if len(untested_inds) == 0:
            break
    inds[out_inds] = False
    return np.nonzero(inds)[0]


def vertinds_to_neighbors(vertex_inds, faces):
    """ Return indices of neighbors of vertices given `faces`

    Parameters
    ----------
    vertex_inds : sequence
       length N.  Indices of vertices
    faces : (F, 3) array-like
       Faces given by indices of vertices for each of ``F`` faces

    Returns
    -------
    adj : list
       For each ``N`` vertex indicated by `vertex_inds`, the vertex
       indices that are neighbors according to the graph given by
       `faces`.  
    """
    full_adj = neighbors(faces)
    adj = []
    for i, n in enumerate(full_adj):
        if i in vertex_inds:
            adj.append(n)
    return adj


def neighbors(faces):
    """ Return indices of neighbors for each vertex within `faces`

    Parameters
    ----------
    faces : (F, 3) array-like
       Faces given by indices of vertices for each of ``F`` faces

    Returns
    -------
    adj : list
       For each vertex found within `faces`, the vertex
       indices that are neighbors according to the graph given by
       `faces`.  We expand the list with empty lists in between
       non-empty neighbors.  
    """
    faces = np.asarray(faces)
    adj = {}
    for face in faces:
        a, b, c = face
        if a in adj:
            adj[a] += [b, c]
        else:
            adj[a] = [b, c]
        if b in adj:
            adj[b] += [a, c]
        else:
            adj[b] = [a, c]
        if c in adj:
            adj[c] += [a, b]
        else:
            adj[c] = [a, b]
    N = max(adj.keys())+1
    out = [[] for i in range(N)]
    for i in range(N):
        if i in adj:
            out[i] = np.sort(np.unique(adj[i]))
    return out


def vertinds_faces(vertex_inds, faces):
    """ Return faces containing any of `vertex_inds`

    Parameters
    ----------
    vertex_inds : sequence
       length N.  Indices of vertices
    faces : (F, 3) array-like
       Faces given by indices of vertices for each of ``F`` faces

    Returns
    ---------
    less_faces : (P, 3) array
       Only retaining rows in `faces` which contain any of `vertex_inds`
    """
    in_inds = []
    vertex_inds = set(vertex_inds)
    for ind, face in enumerate(faces):
        if vertex_inds.intersection(face):
            in_inds.append(ind)
    return faces[in_inds]

def edges(vertex_inds, faces):
    r""" Return array of starts and ends of edges from list of faces
        taking regard of direction.

    Parameters
    ----------
    vertex_inds : sequence
       length N.  Indices of vertices
    faces : (F, 3) array-like
       Faces given by indices of vertices for each of F faces

    Returns
    -------
    edgearray : (E2, 2) array
       where E2 = 2*E, twice the number of edges. If e= (a,b) is an
       edge then [a,b] and [b,a] are included in edgearray.
    """

    edgedic = {}
    for face in faces:
        edgedic[(face[0],face[1])]=1
        edgedic[(face[0],face[2])]=1
        edgedic[(face[1],face[0])]=1
        edgedic[(face[1],face[2])]=1
        edgedic[(face[2],face[0])]=1
        edgedic[(face[2],face[1])]=1
    
    start, end = zip(*edgedic)

    edgearray = np.column_stack(zip(*edgedic))

    return edgearray
 
def vertex_adjacencies(vertex_inds, faces):
    """ Return matrix which shows the adjacent vertices
        of each vertex
        
    Parameters
    ----------
    vertex_inds : sequence
       length N.  Indices of vertices
    
    faces : (F, 3) array-like
       Faces given by indices of vertices for each of F faces
    
    Returns
    -------
    """
    edgearray = edges(vertex_inds, faces)
    V = len(vertex_inds)
    a = sparse.coo_matrix((np.ones(edgearray.shape[0]),
                          (edgearray[:,0],edgearray[:,1])),
                          shape=(V,V))
    return a

    
def argmax_from_adj(vals, vertex_inds, adj_inds):
    """ Indices of local maxima from `vals` given adjacent points

    See ``reconstruction_performance`` for optimized versions of this
    routine. 
    
    Parameters
    ----------
    vals : (N,) array-like
       values at all vertices referred to in either of `vertex_inds` or
       `adj_inds`'
    vertex_inds : None or (V,) array-like
       indices into `vals` giving vertices that may be local maxima.
       If None, then equivalent to ``np.arange(N)``
    adj_inds : sequence
       For every vertex in ``vertex_inds``, the indices (into `vals`) of
       the neighboring points

    Returns
    -------
    inds : (M,) array
       Indices into `vals` giving local maxima of vals, given topology
       from `adj_inds`, and restrictions from `vertex_inds`.  Inds are
       returned sorted by value at that index - i.e. smallest value (at
       index) first.
    """
    vals = np.asarray(vals)
    if vertex_inds is None:
        vertex_inds = np.arange(vals.shape[0])
    else:
        vertex_inds = np.asarray(vertex_inds)
    maxes = []
    for i, adj in enumerate(adj_inds):
        vert_ind = vertex_inds[i]
        val = vals[vert_ind]
        if np.all(val > vals[adj]):
            maxes.append((val, vert_ind))
    if len(maxes) == 0:
        return np.array([])
    maxes.sort(cmp=lambda x, y: cmp(x[0], y[0]))
    vals, inds = zip(*maxes)
    return np.array(inds)


def peak_finding_compatible(vertices,
                            hemisphere='z',
                            equator_thresh=None,
                            dist_thresh=None):
    """ Check that a sphere mesh is compatible with ``peak_finding``

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
       ``v``, not ``v_dash``, will appear in the output vertex indices
       `inds`. None results in a threshold based on the input data type
       of ``vertices``
    
    Returns
    -------
    compatible : bool
       True if the sphere mesh is compatible with ``peak_finding``
    """
    inds = sym_hemisphere(vertices, hemisphere,
                          equator_thresh, dist_thresh)
    N = vertices.shape[0] // 2
    return np.all(inds == np.arange(N))

def euler_characteristic_check(vertices, faces, chi=2):
    r'''
    If $f$ = number of faces, $e$ = number_of_edges and $v$ = number of vertices,
    the Euler formula says $f-e+v = 2$ for a mesh
    on a sphere. Here, assuming we have a healthy triangulation every
    face is a triangle, all 3 of whose edges should belong to exactly
    two faces. So $2*e = 3*f$. To avoid integer division and consequential
    integer rounding we test whether $2*f - 3*f + 2*v == 4$ or, more generally,
    whether $2*v - f == 2*\chi$ where $\chi$ is the Euler characteristic of the mesh.

    - Open chain (track) has $\chi=1$
    - Closed chain (loop) has $\chi=0$
    - Disk has $\chi=1$
    - Sphere has $\chi=2$

    Parameters
    ----------
    vertices : (N,3) array-like
       (x, y, z) Point coordinates of N vertices
    faces : (M,3) array-like of type int
       (i1, i2, i3) Integer indices of the vertices of the (triangular) faces
    chi : int, or None
       The Euler characteristic of the mesh to be checked
       
    Returns
    -------
    check : bool
       True if the mesh has Euler characteristic chi
    '''
    
    v = vertices.shape[0]
    f = faces.shape[0]
    if 2*v-f==2*chi:
        return True
    else:
        return False


