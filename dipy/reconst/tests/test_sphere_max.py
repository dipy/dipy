""" Testing sphere maxima finding and associated routines
"""

from os.path import join as pjoin, dirname
import numpy as np
from dipy.data import get_sphere

from dipy.core.meshes import (
    sym_hemisphere,
    neighbors,
    vertinds_to_neighbors,
    vertinds_faces,
    argmax_from_adj,
    peak_finding_compatible,
    edges,
    vertex_adjacencies)

import dipy.reconst.recspeed as dcr

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

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
VERTEX_INDS = np.array([0,1,2,3,4,5])

DATA_PATH = pjoin(dirname(__file__), '..', 'matrices')
# vertex, face tuple
SPHERE_DATA = get_sphere('symmetric362')

def test_sym_hemisphere():
    assert_raises(ValueError, sym_hemisphere,
                        VERTICES, 'k')
    assert_raises(ValueError, sym_hemisphere,
                        VERTICES, '%z')
    for hem in ('x', 'y', 'z', '-x', '-y', '-z'):
        vert_inds = sym_hemisphere(VERTICES, hem)
        assert_equal(vert_inds.shape, (3,))
        verts = VERTICES[vert_inds]
        # there are no symmetrical points remanining in vertices
        for vert in verts:
            assert_false(np.any(np.all(
                    vert * -1 == verts)))
    # Test the sphere mesh data
    vertices, _ = SPHERE_DATA
    n_vertices = vertices.shape[0]
    vert_inds = sym_hemisphere(vertices)
    assert_array_equal(vert_inds,
                       np.arange(n_vertices / 2))


def test_vertinds_neighbors():
    adj = neighbors(FACES)
    assert_array_equal(adj,
                             [[1, 2, 3, 4],
                              [0, 2, 4, 5],
                              [0, 1, 3, 5],
                              [0, 2, 4, 5],
                              [0, 1, 3, 5],
                              [1, 2, 3, 4]])
    adj = vertinds_to_neighbors(np.arange(6),
                                FACES)
    assert_array_equal(adj,
                             [[1, 2, 3, 4],
                              [0, 2, 4, 5],
                              [0, 1, 3, 5],
                              [0, 2, 4, 5],
                              [0, 1, 3, 5],
                              [1, 2, 3, 4]])
    # subset of inds gives subset of faces
    adj = vertinds_to_neighbors(np.arange(3),
                                      FACES)
    assert_array_equal(adj,
                             [[1, 2, 3, 4],
                              [0, 2, 4, 5],
                              [0, 1, 3, 5]])
    # can be any subset
    adj = vertinds_to_neighbors(np.arange(3,6),
                                      FACES)
    assert_array_equal(adj,
                             [[0, 2, 4, 5],
                              [0, 1, 3, 5],
                              [1, 2, 3, 4]])
    # just test right size for the real mesh
    vertices, faces = SPHERE_DATA
    n_vertices = vertices.shape[0]
    adj = vertinds_to_neighbors(np.arange(n_vertices),
                                      faces)
    assert_equal(len(adj), n_vertices)
    assert_equal(len(adj[1]), 6)


def test_vertinds_faces():
    # routines to strip out faces
    f2 = vertinds_faces(range(6), FACES)
    assert_array_equal(f2, FACES)
    f2 = vertinds_faces([0, 5], FACES)
    assert_array_equal(f2, FACES)
    f2 = vertinds_faces([0], FACES)
    assert_array_equal(f2, FACES[:4])


def test_neighbor_max():
    # test ability to find maxima on sphere using neighbors
    vert_inds = sym_hemisphere(VERTICES)
    adj_inds = vertinds_to_neighbors(vert_inds, FACES)
    # test slow and fast routine
    for func in (argmax_from_adj, dcr.argmax_from_adj):
        # all equal, no maxima
        vert_vals = np.zeros((N_VERTICES,))
        inds = func(vert_vals,
                    vert_inds,
                    adj_inds)
        assert_equal(inds.size, 0)
        # just ome max
        for max_pos in range(3):
            vert_vals = np.zeros((N_VERTICES,))
            vert_vals[max_pos] = 1
            inds = func(vert_vals,
                        vert_inds,
                        adj_inds)
            assert_array_equal(inds, [max_pos])
        # maxima outside hemisphere don't appear
        for max_pos in range(3,6):
            vert_vals = np.zeros((N_VERTICES,))
            vert_vals[max_pos] = 1
            inds = func(vert_vals,
                        vert_inds,
                        adj_inds)
            assert_equal(inds.size, 0)
        # use whole mesh, with two maxima
        w_vert_inds = np.arange(6)
        w_adj_inds = vertinds_to_neighbors(w_vert_inds, FACES)
        vert_vals = np.array([1.0, 0, 0, 0, 0, 2])
        inds = func(vert_vals, w_vert_inds, w_adj_inds)
        assert_array_equal(inds, [0, 5])
        # check too few vals raises sensible error.  For the Cython
        # version of the routine, the test below causes odd errors and
        # segfaults with numpy SVN vintage June 2010 (sometime after
        # 1.4.0 release) - see
        # http://groups.google.com/group/cython-users/browse_thread/thread/624c696293b7fe44?pli=1
        # assert_raises(IndexError, func, vert_vals[:3],
        # w_vert_inds, w_adj_inds)


def test_performance():
    # test this implementation against Frank Yeh implementation
    vertices, faces = SPHERE_DATA
    n_vertices = vertices.shape[0]
    vert_inds = sym_hemisphere(vertices)
    adj = vertinds_to_neighbors(vert_inds, faces)
    np.random.seed(42)
    vert_vals = np.random.uniform(size=(n_vertices,))
    maxinds = argmax_from_adj(vert_vals, vert_inds, adj)
    maxes, pfmaxinds = dcr.peak_finding(vert_vals, faces)
    assert_array_equal(maxinds, pfmaxinds[::-1])


def test_sym_check():
    assert_true(peak_finding_compatible(VERTICES))
    vertices, faces = SPHERE_DATA
    assert_true(peak_finding_compatible(vertices))
    assert_false(peak_finding_compatible(vertices[::-1]))


def test_adjacencies():
    faces = FACES
    vertex_inds = VERTEX_INDS
    edgearray = edges(vertex_inds, faces)
    assert_array_equal(edgearray.shape,(24,2))
    assert_array_equal(edgearray,
                             [[3, 0], [5, 4], [2, 1], [5, 1],
                              [2, 5], [0, 3], [4, 0], [1, 2],
                              [1, 5], [0, 4], [5, 3], [4, 1],
                              [3, 2], [4, 5], [1, 4], [2, 3],
                              [1, 0], [3, 5], [0, 1], [5, 2],
                              [2, 0] ,[4, 3], [3, 4], [0, 2]])
    assert_array_equal(vertex_adjacencies(vertex_inds, faces).shape,(6,6))
