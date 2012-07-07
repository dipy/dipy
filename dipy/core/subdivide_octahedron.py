"""Create a unit sphere by subdividing all triangles of an octahedron
recursively.

The unit sphere has a radius of 1, which also means that all points in this
sphere (assumed to have centre at [0, 0, 0]) have an absolute value (modulus)
of 1. Another feature of the unit sphere is that the unit normals of this
sphere are exactly the same as the vertices.

This recursive method will avoid the common problem of the polar singularity, 
produced by 2d (lon-lat) parameterization methods.

If you require a sphere with a radius other than that of 1, simply multiply
the vertex array by this new radius (although this will break the "vertex array
equal to unit normal array" property)
"""
import numpy as np
from .sphere import Sphere, HemiSphere
import warnings

#the vertices of an octahedron
octahedron_vertices = np.array( [
    [ 1.0, 0.0, 0.0], # 0 
    [-1.0, 0.0, 0.0], # 1
    [ 0.0, 1.0, 0.0], # 2 
    [ 0.0,-1.0, 0.0], # 3
    [ 0.0, 0.0, 1.0], # 4 
    [ 0.0, 0.0,-1.0]  # 5                                
    ] )

#each edge is a pair of neighboring vertices, the edges and triangles bellow
#follow the cycle rule. For more on the cycle rule see divide_all.
octahedron_edges = np.array( [
    [0, 4],  #0  #0
    [1, 5],  #10 #1
    [4, 2],  #1  #2
    [5, 3],  #11 #3
    [2, 0],  #2  #4
    [3, 1],  #6  #5
    [2, 1],  #3  #6
    [3, 0],  #7  #7
    [1, 4],  #4  #8
    [0, 5],  #8  #9
    [4, 3],  #5  #10
    [5, 2],  #9  #11
    ], dtype='uint16' )

#each triangle is a set of three edges, because these triangles and edges
#follow the cycle rule you can get the three vertices of a triangle by using
#octahedron_edges[octahedron_triangles, 0]. For more on the cycle rule see
#divide_all
octahedron_triangles = np.array( [
    [ 0,  2,  4],
    [ 1,  3,  5],
    [ 2,  6,  8],
    [ 3,  7,  9],
    [ 8, 10,  5],
    [ 9, 11,  4],
    [ 0, 10,  7],
    [ 1, 11,  6],
    ], dtype='uint16')

def _divide_all( vertices, edges, triangles ):
    r""" Subdivides triangles into smaller triangles

    Parameters
    ------------
    vertices : ndarray
        A Vx3 array with the x, y, and z coordinates of of each vertex.
    edges : ndarray
        An Ex2 array of edges where each edge is a pair of neighboring
        vertices.
    triangles : ndarray
        A Tx3 array of triangles, where each triangle is a set of three edges.

    Returns
    ---------
    vertices : ndarray
        like above
    edges : ndarray
        like above
    triangles : ndarray
        like above

    Important Note on Cycle Rule:
    -----------------------------
    The edges and triangles that are passed to this function must follow the
    cycle rule. If they do not the result will not be correct. The cycle
    rule states that the second vertex of each edge of each triangle must be
    the first vertex in the next edge of that triangle. For example take the
    triangle drawn below:
              1
             /\
          B /  \ C
           /____\
          0   A  2
    If the triangle is [A, B, C] the edges must be:
        A: [2, 0]
        B: [0, 1]
        C: [1, 2]
    If the triangle is [C, B, A] the edges must be:
        C: [2, 1]
        B: [1, 0]
        A: [0, 2]

    This must be true for ALL of the triangles. Such an arrangement of edges
    and triangles is not possible in general but is possible for the
    octahedron.

    Implementation Details
    ------------------------
    Using the triangle drawn above, we segment the triangle into four smaller
    triangles.
                  1
                 /\
                /  \
              b/____\c
              /\    /\
             /  \  /  \
            /____\/____\
           0      a     2

    Make new vertices at the center of each edge::
         b = (0+1)/2
         c = (1+2)/2
         a = (2+0)/2

    Normalize a, b, c, and replace each old edge with two new edges:
        [0, b]
        [1, c]
        [2, a]
        [b, 1]
        [c, 2]
        [a, 0]
    Make a new edge connecting each pair of new vertices in the triangle:
        [b, a]
        [a, c]
        [c, b]

    Construct new triangles, notice that each edge was created in such a way
    so that our new triangles follow the cycle rule:
        t1 [0b,ba,a0]
        t2 [1c,cb,b1]
        t3 [2a,ac,c2]
        t4 [ba,ac,cb]

    Code was adjusted from dlampetest website
    http://sites.google.com/site/dlampetest/python/triangulating-a-sphere-recursively
    """
    num_vertices = len(vertices)
    num_edges = len(edges)
    num_triangles = len(triangles)

    new_vertices = vertices[edges].sum(1)
    norms_new_vertices = np.sqrt((new_vertices*new_vertices).sum(-1))
    new_vertices /= norms_new_vertices[:, None]
    vertices = np.vstack((vertices, new_vertices))
    new_v_ind = np.arange(num_vertices, num_vertices+num_edges, dtype='uint16')

    v_b = new_v_ind[triangles[:,0]]
    v_c = new_v_ind[triangles[:,1]]
    v_a = new_v_ind[triangles[:,2]]
    edges = np.vstack((np.c_[edges[:,0], new_v_ind],
                       np.c_[new_v_ind, edges[:,1]],
                       np.c_[v_b, v_a],
                       np.c_[v_a, v_c],
                       np.c_[v_c, v_b],
                       ))
    E_0b = triangles[:,0]
    E_b1 = triangles[:,0] + num_edges
    E_1c = triangles[:,1]
    E_c2 = triangles[:,1] + num_edges
    E_2a = triangles[:,2]
    E_a0 = triangles[:,2] + num_edges
    S = 2*num_edges
    E_ba = np.arange(S+0*num_triangles, S+1*num_triangles, dtype='uint16')
    E_ac = np.arange(S+1*num_triangles, S+2*num_triangles, dtype='uint16')
    E_cb = np.arange(S+2*num_triangles, S+3*num_triangles, dtype='uint16')
    triangles = np.vstack((np.c_[E_0b, E_ba, E_a0],
                           np.c_[E_1c, E_cb, E_b1],
                           np.c_[E_2a, E_ac, E_c2],
                           np.c_[E_ba, E_ac, E_cb],
                           ))
    return vertices, edges, triangles

def create_unit_sphere( recursion_level=2 ):
    """ Creates a unit sphere by subdividing a unit octahedron.

    Starts with a unit octahedron and subdivides the faces, projecting the
    resulting points onto the surface of a unit sphere.

    Parameters
    ------------
    recursion_level : int
        Level of subdivision, recursion_level=1 will return an octahedron,
        anything bigger will return a more subdivided sphere. The sphere will
        have $4^recursion_level+2$ vertices.

    Returns
    ---------
    Sphere :
        The unit sphere.

    See Also
    ----------
    create_unit_hemisphere, Sphere
    """
    if recursion_level > 7 or recursion_level < 1:
        raise ValueError("recursion_level must be between 1 and 7")
    vertices = octahedron_vertices
    edges = octahedron_edges
    triangles = octahedron_triangles
    for i in range( recursion_level - 1 ):
        vertices, edges, triangles = _divide_all(vertices, edges, triangles)
    faces = edges[triangles, 0]
    return Sphere(xyz=vertices, edges=edges, faces=faces)

def create_unit_hemisphere( recursion_level=2 ):
    """Creates a unit sphere by subdividing a unit octahedron, returns half
    the sphere.

    Parameters
    -------------
    recursion_level : int
        Level of subdivision, recursion_level=1 will return an octahedron,
        anything bigger will return a more subdivided sphere. The sphere will
        have $(4^recursion_level+2)/2$ vertices.

    Returns
    ---------
    HemiSphere :
        Half of a unit sphere.

    See Also
    ----------
    create_unit_sphere, Sphere, HemiSphere
    """
    sphere = create_unit_sphere( recursion_level )
    vertices = sphere.vertices[::2]
    edges = sphere.edges[::2] // 2
    faces = sphere.faces[::2] // 2
    return HemiSphere(xyz=vertices, edges=edges, faces=faces)


