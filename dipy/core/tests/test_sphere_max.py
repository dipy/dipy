""" Testing sphere maximae finding and associated routines
"""

from os.path import join as pjoin, dirname
import numpy as np

from dipy.core.meshes import (
    sym_hemisphere,
    neighbors,
    vertinds_to_neighbors,
    vertinds_faces,
    argmax_from_adj)
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
def test_sym_hemisphere():
    yield assert_raises(ValueError, sym_hemisphere,
                        VERTICES, 'k')
    yield assert_raises(ValueError, sym_hemisphere,
                        VERTICES, '%z')
    for hem in ('x', 'y', 'z', '-x', '-y', '-z'):
        vert_inds = sym_hemisphere(VERTICES, hem)
        yield assert_equal(vert_inds.shape, (3,))
        verts = VERTICES[vert_inds]
        # there are no symmetrical points remanining in vertices
        for vert in verts:
            yield assert_false(np.any(np.all(
                    vert * -1 == verts)))
    # Test the sphere mesh data
    vertices = SPHERE_DATA['vertices']
    n_vertices = vertices.shape[0]
    vert_inds = sym_hemisphere(vertices)
    yield assert_array_equal(vert_inds,
                             np.arange(n_vertices / 2))


@parametric
def test_vertinds_neighbors():
    adj = neighbors(FACES)
    yield assert_equal(adj,
                       [[1, 2, 3, 4],
                        [0, 2, 4, 5],
                        [0, 1, 3, 5],
                        [0, 2, 4, 5],
                        [0, 1, 3, 5],
                        [1, 2, 3, 4]])
    adj = vertinds_to_neighbors(np.arange(6),
                                FACES)
    yield assert_equal(adj,
                       [[1, 2, 3, 4],
                        [0, 2, 4, 5],
                        [0, 1, 3, 5],
                        [0, 2, 4, 5],
                        [0, 1, 3, 5],
                        [1, 2, 3, 4]])
    # subset of inds gives subset of faces
    adj = vertinds_to_neighbors(np.arange(3),
                                      FACES)
    yield assert_equal(adj,
                       [[1, 2, 3, 4],
                        [0, 2, 4, 5],
                        [0, 1, 3, 5]])
    # can be any subset
    adj = vertinds_to_neighbors(np.arange(3,6),
                                      FACES)
    yield assert_equal(adj,
                       [[0, 2, 4, 5],
                        [0, 1, 3, 5],
                        [1, 2, 3, 4]])
    # just test right size for the real mesh
    vertices = SPHERE_DATA['vertices']
    faces = SPHERE_DATA['faces']
    n_vertices = vertices.shape[0]
    adj = vertinds_to_neighbors(np.arange(n_vertices),
                                      faces)
    yield assert_equal(len(adj), n_vertices)
    yield assert_equal(len(adj[1]), 6)


@parametric
def test_vertinds_faces():
    # routines to strip out faces
    f2 = vertinds_faces(range(6), FACES)
    yield assert_array_equal(f2, FACES)
    f2 = vertinds_faces([0, 5], FACES)
    yield assert_array_equal(f2, FACES)
    f2 = vertinds_faces([0], FACES)
    yield assert_array_equal(f2, FACES[:4])


@parametric
def test_neighbor_max():
    # test ability to find maximae on sphere using neighbors
    vert_inds = sym_hemisphere(VERTICES)
    adj_inds = vertinds_to_neighbors(vert_inds, FACES)
    # all equal, no maximae
    vert_vals = np.zeros((N_VERTICES,))
    inds = argmax_from_adj(vert_vals,
                        vert_inds,
                        adj_inds)
    yield assert_equal(inds.size, 0)
    # just ome max
    for max_pos in range(3):
        vert_vals = np.zeros((N_VERTICES,))
        vert_vals[max_pos] = 1
        inds = argmax_from_adj(vert_vals,
                            vert_inds,
                            adj_inds)
        yield assert_array_equal(inds, [max_pos])
    # maximae outside hemisphere don't appear
    for max_pos in range(3,6):
        vert_vals = np.zeros((N_VERTICES,))
        vert_vals[max_pos] = 1
        inds = argmax_from_adj(vert_vals,
                            vert_inds,
                            adj_inds)
        yield assert_equal(inds.size, 0)
    # use whole mesh, with two maximae
    vert_inds = np.arange(6)
    adj_inds = vertinds_to_neighbors(vert_inds, FACES)
    vert_vals = [1, 0, 0, 0, 0, 2]
    inds = argmax_from_adj(vert_vals, vert_inds, adj_inds)
    yield assert_array_equal(inds, [0, 5])


@parametric
def test_performance():
    # test this implementation against Frank Yeh implementation
    vertices = SPHERE_DATA['vertices']
    faces = SPHERE_DATA['faces']
    n_vertices = vertices.shape[0]
    vert_inds = sym_hemisphere(vertices)
    adj = vertinds_to_neighbors(vert_inds, faces)
    np.random.seed(42)
    vert_vals = np.random.uniform(size=(n_vertices,))
    maxinds = argmax_from_adj(vert_vals, vert_inds, adj)
    maxes, pfmaxinds = peak_finding(vert_vals, faces)
    yield assert_array_equal(maxinds, pfmaxinds[::-1])
    
