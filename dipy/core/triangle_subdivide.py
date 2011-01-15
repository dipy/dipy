'''Create a unit sphere by subdividing all triangles of an octahedron
recursively.

The unit sphere has a radius of 1, which also means that all points in this sphere
(assumed to have centre at [0, 0, 0])
have an absolute value (modulus) of 1. Another feature of the unit sphere is that the
unit normals of this sphere are exactly the same as the vertices.

This recursive method will avoid the common problem of the polar singularity, 
produced by 2d (lon-lat) parameterization methods.

If you require a sphere with another radius than that of 1, simply multiply every
single value in the vertex array by this new radius 
(although this will break the "vertex array equal to unit normal array" property)
'''
import numpy
np = numpy

octahedron_vertices = numpy.array( [ 
    [ 1.0, 0.0, 0.0], # 0 
    [-1.0, 0.0, 0.0], # 1
    [ 0.0, 1.0, 0.0], # 2 
    [ 0.0,-1.0, 0.0], # 3
    [ 0.0, 0.0, 1.0], # 4 
    [ 0.0, 0.0,-1.0]  # 5                                
    ] )

octahedron_edges = numpy.array( [
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

octahedron_triangles = numpy.array( [ 
    [ 0,  2,  4],
    [ 1,  3,  5],
    [ 2,  6,  8],
    [ 3,  7,  9],
    [ 8, 10,  5],
    [ 9, 11,  4],
    [ 0, 10,  7],
    [ 1, 11,  6],
    ], dtype='uint16')

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = numpy.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr /= lens[:,None]

def divide_all( vertices, edges, triangles ):
    r""" Subdivides a triangle

    Parameters
    ------------
    vertices : ndarray
        A Vx3 array with the x, y, and z coordinates of of each vertex
    edges : ndarray
        An Ex2 array were each pair of values is an index 

    Returns
    ---------
    vertices : array
        A 2d array with the x, y, and z coordinates of vertices
    edges : array
        A 2d array of vertex pairs for every set of neighboring vertexes
    triangles : array
        A 2d array of edge triplets representing triangles

    Notes
    -------
    Subdivide each triangle in the old approximation and normalize the new
    points thus generated to lie on the surface of the unit sphere.

    Each input triangle with vertices labelled [0,1,2], as shown below, is
    represented by a set of edges. The edges are written in such a way so that
    the second vertex in each edge is the first vertex in the next edge. For
    example::

         [0, 1]
         [1, 2]
         [2, 0]

    Make new points::

         b = (0+1)/2
         c = (1+2)/2
         a = (2+0)/2

    Construct new triangles::

        t1 [0b,ba,a0]
        t2 [1c,cb,b1]
        t3 [2a,ac,c2]
        t4 [ba,ac,cb]

    Like this::

                  1
                 /\
                /  \
              b/____\ c
              /\    /\
             /  \  /  \
            /____\/____\
           0      a     2

    Normalize a, b, c.

    When constructed this way edges[triangles,0] or edges[triangles,1] will both
    return the three vertices that make up each triangle (in a different order):
    
    Code was adjusted from dlampetest website
    http://sites.google.com/site/dlampetest/python/triangulating-a-sphere-recursively
    
    
    """
    num_vertices = len(vertices)
    num_edges = len(edges)
    num_triangles = len(triangles)

    new_vertices = vertices[edges].sum(1)
    normalize_v3(new_vertices)
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
    E_ba = np.arange(3*num_triangles, 4*num_triangles, dtype='uint16')
    E_ac = np.arange(4*num_triangles, 5*num_triangles, dtype='uint16')
    E_cb = np.arange(5*num_triangles, 6*num_triangles, dtype='uint16')
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
        anything bigger will return a more subdivided sphere.

    Returns
    ---------
    vertices : array
        A 2d array with the x, y, and z coordinates of vertices on a unit
        sphere.
    edges : array
        A 2d array of vertex pairs for every set of neighboring vertexes
        on a unit sphere.
    triangles : array
        A 2d array of edge triplets representing triangles on the surface of a
        unit sphere. 

    See Also
    ----------
    create_half_sphere, divide_all

    """
    
    vertex_array, edge_array, triangle_array = octahedron_vertices, \
                                               octahedron_edges, \
                                               octahedron_triangles
    for i in range( recursion_level - 1 ):
        vertex_array, edge_array, triangle_array = divide_all(vertex_array,
                                                              edge_array,
                                                              triangle_array)
    return vertex_array, edge_array, triangle_array

def create_half_unit_sphere( recursion_level=2 ):
    """ Creates a unit sphere and returns the top half

    Starting with a symmetric sphere of points, removes half the points so that
    for any pair of points a, -a only one is kept. Removes half the edges so
    for any pair of edges [a, b]; [-a, -b] only one is kept. The new edges are
    constructed in a way so that references to any removed point, r, is
    replaced by a reference to -r. Removes half the triangles in the same way.
    
    Parameters
    -------------
    recursion_level : int
        Level of subdivision, recursion_level=1 will return an octahedron,
        anything bigger will return a more subdivided sphere.

    Returns
    ---------
    vertices : array
        A 2d array with the x, y, and z coordinates of vertices on a unit
        sphere.
    edges : array
        A 2d array of vertex pairs for every set of neighboring vertexes
        on a unit sphere.
    triangles : array
        A 2d array of edge triplets representing triangles on the surface of a
        unit sphere. 

    See Also
    ----------
    create_half_sphere, divide_all

    """
    
    v, e, t = create_unit_sphere( recursion_level )
    return remove_half_sphere(v, e, t)
    
def remove_half_sphere(v, e, t):
    """ Returns a triangulated half sphere

    Removes half the vertices, edges, and triangles from a unit sphere created
    by create_unit_sphere
    
    """
    return v[::2], e[::2]/2, t[::2]/2

