

import numpy as np
from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_almost_equal, assert_raises)
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           run_module_suite)

from dipy.core.triangle_subdivide import (_get_forces, disperse_charges,
                                          _divide_all, create_unit_sphere)

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
    v = sphere.vertices
    e = sphere.edges
    f = sphere.faces
    assert_array_almost_equal(v[::2], -v[1::2])
    assert_array_almost_equal(v[e[::2]], -v[e[1::2]])
    assert_array_almost_equal(v[f[::2]], -v[f[1::2]])
    assert_array_almost_equal((v*v).sum(1), 1)

def create_half_unit_sphere():
    v, e, t = create_unit_sphere(7)
    assert_array_almost_equal((v*v).sum(1), 1)

def test_get_force():
    charges = np.array([[1., 0, 0],
                        [0, 1., 0],
                        [0, 0, 1.]])
    force, pot = _get_forces(charges)
    assert_array_almost_equal(force, 0)

    charges = np.array([[1, -.1, 0],
                        [1, 0, 0]])
    force, pot = _get_forces(charges)
    assert_array_almost_equal(force[1, [0, 2]], 0)
    assert_true(force[1, 1] > 0)

def test_disperse_charges():
    charges = np.array([[1., 0, 0],
                        [0, 1., 0],
                        [0, 0, 1.]])
    d_charges, pot = disperse_charges(charges, 10)
    assert_array_almost_equal(charges, d_charges)

    a = np.sqrt(3)/2
    charges = np.array([[3./5, 4./5, 0],
                        [4./5, 3./5, 0]])
    expected_charges =  np.array([[0, 1., 0],
                                  [1., 0, 0]])
    d_charges, pot = disperse_charges(charges, 1000, .2)
    assert_array_almost_equal(expected_charges, d_charges)
    for ii in xrange(1, len(pot)):
        #check that the potential of the system is either going down or
        #stayting almost the same
        assert_true(pot[ii] - pot[ii-1] < 1e-12)

    #check that the function seems to work with a larger number of charges
    charges = np.arange(21).reshape(7,3)
    norms = np.sqrt((charges*charges).sum(-1))
    charges = charges / norms[:, None]
    d_charges, pot = disperse_charges(charges, 1000, .05)
    for ii in xrange(1, len(pot)):
        #check that the potential of the system is either going down or
        #stayting almost the same
        assert_true(pot[ii] - pot[ii-1] < 1e-12)
    #check that the resulting charges all lie on the unit sphere
    norms = np.sqrt((d_charges*d_charges).sum(-1))
    assert_array_almost_equal(norms, 1)

if __name__ == "__main__":
    run_module_suite()
