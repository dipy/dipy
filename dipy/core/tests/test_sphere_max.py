""" Testing sphere maxima
"""

import numpy as np

from dipy.core.meshes import (hemisphere_neighbors,
                              mesh_maximae)

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


@parametric
def select_hemisphere():
    vert_inds, adj_inds = hemisphere_neighbors(VERTICES, FACES)
    yield assert_equal(verts_inds.shape, (3,))
    yield assert_equal(neighbors.shape, (3, 6))
    verts = VERTICES[vert_inds]
    # there are no symmetrical points remanining in vertices
    for vert in verts:
        yield assert_false(np.any(np.all(
                vert * -1 == verts)))


def test_maximae():
    # test ability to find maximae on sphere
    vert_inds, adj_inds = hemisphere_neighbors(VERTICES, FACES)
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
