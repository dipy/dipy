""" Testing sphere maximae finding and associated routines
"""

from os.path import join as pjoin, dirname
import numpy as np

from dipy.core.meshes import (
    hemisphere_vertices,
    vertinds_to_neighbors,
    mesh_maximae)
from dipy.core.reconstruction_performance import peak_finding

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric

# 8 faces (two square pyramids)
VERTICES = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]], dtype=np.float)
FACES = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
        [5, 1, 2],
        [5, 2, 3],
        [5, 3, 4],
        [5, 4, 1]])
N_VERTICES = VERTICES.shape[0]

DATA_PATH = pjoin(dirname(__file__), '..', 'matrices')
SPHERE_DATA = np.load(pjoin(DATA_PATH,
                            'evenly_distributed_sphere_362.npz'))


@parametric
def test_hemisphere_vertices():
    vert_inds = hemisphere_vertices(VERTICES)
    yield assert_equal(vert_inds.shape, (3,))
    verts = VERTICES[vert_inds]
    # there are no symmetrical points remanining in vertices
    for vert in verts:
        yield assert_false(np.any(np.all(
                vert * -1 == verts)))
    # Test the sphere mesh data
    vertices = sphere_data['vertices']
    n_vertices = vertices.shape[0]
    vert_inds = hemisphere_vertices(vertices)
    yield assert_equal(n_vertices / 2.0,
                       vert_inds.items)


@parametric
def test_vertinds_neighbors():
    neighbors = vertinds_to_neighbors(np.arange(6),
                                      FACES)
    yield assert_array_equal(neighbors,
                             [[1, 2, 3, 4]
                              [0, 2, 4, 5],
                              [0, 1, 3, 5],
                              [0, 2, 4, 5],
                              [0, 1, 3, 5],
                              [1, 2, 3, 4]])
    # just test right size for the real mesh
    vertices = sphere_data['vertices']
    n_vertices = vertices.shape[0]
    neighbors = vertinds_to_neighbors(np.arange(n_vertices))
    yield assert_equal(neighbors.shape,
                       (n_vertices, 6))
    

@parametric
def test_maximae():
    # test ability to find maximae on sphere
    vert_inds = hemisphere_vertices(VERTICES)
    adj_inds = vertinds_to_neighbors(vert_inds, FACES)
    # all equal, no maximae
    vert_vals = np.zeros((N_VERTICES,))
    inds = mesh_maximae(vert_vals,
                        vert_inds,
                        adj_inds)
    yield assert_equal(inds.items, 0)
    # just ome max
    for max_pos in range(3):
        vert_vals = np.zeros((N_VERTICES,))
        vert_vals[max_pos] = 1
        inds = mesh_maximae(vert_vals,
                            vert_inds,
                            adj_inds)
        yield assert_array_equal(inds, [max_pos])
    # maximae outside hemisphere don't appear
    for max_pos in range(3,6):
        vert_vals = np.zeros((N_VERTICES,))
        vert_vals[max_pos] = 1
        inds = mesh_maximae(vert_vals,
                            vert_inds,
                            adj_inds)
        yield assert_equal(inds.items, 0)


@parametric
def test_performance():
    # test this implementation against Frank Yeh implementation
    vertices = sphere_data['vertices']
    faces = sphere_data['faces']
    n_vertices = vertices.shape[0]
    vert_inds = hemisphere_vertices(vertices)
    neighbors = vertinds_to_neighbors(vert_inds, faces)
    np.random.seed(42)
    vert_vals = np.random.uniform(size=(n_vertices,))
    maxinds = mesh_maximae(vert_vals, vert_inds, neighbors)
    maxes, pfmaxinds = peak_finding(vert_vals, faces)
    yield assert_array_equal(maxinds, pfmaxinds)
    
