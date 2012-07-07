import numpy as np
from nose.tools import assert_true, assert_false, \
     assert_equal, assert_almost_equal, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.core.subdivide_octahedron import _divide_all, create_unit_sphere

def test_divide_all():
    vertices = np.array([[1., 0, 0],
                         [0, 1., 0],
                         [0, 0, 1.]])
    edges = np.array([[0, 1],
                      [1, 2],
                      [2, 0]])
    triangles = np.array([[0, 1, 2]])
    v, e, t = _divide_all(vertices, edges, triangles)
    assert_array_equal((vertices*vertices).sum(1), 1)
    assert_array_almost_equal((v*v).sum(1), 1)
    assert_array_equal(e[t,0], np.roll(e[t, 1], 1, 1))

def test_create_unit_sphere():
    sphere = create_unit_sphere(7)
    v, e, f = sphere.vertices, sphere.edges, sphere.faces
    assert_array_almost_equal(v[::2], -v[1::2])
    assert_array_almost_equal(v[e[::2]], -v[e[1::2]])
    assert_array_almost_equal(v[f[::2]], -v[f[1::2]])
    assert_array_almost_equal((v*v).sum(1), 1)

def create_half_unit_sphere():
    sphere = create_half_unit_sphere(7)
    v, e, f = sphere.vertices, sphere.edges, sphere.faces
    assert_array_almost_equal((v*v).sum(1), 1)

