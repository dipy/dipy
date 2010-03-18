""" Testing utility functions

"""

import numpy as np

from dipy.core.geometry import (sphere2cart, cart2sphere,
                             nearest_pos_semi_def)

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric, sphere_points


@parametric
def test_sphere_cart():
    for pt in sphere_points:
        az, zen, r = cart2sphere(*pt)
        xyz = sphere2cart(az, zen, 1)
        yield assert_array_almost_equal(xyz, pt)


@parametric    
def test_nearest_pos_semi_def():
    B = np.diag(np.array([1,2,3]))
    yield assert_array_almost_equal(B, nearest_pos_semi_def(B))
    B = np.diag(np.array([0,2,3]))
    yield assert_array_almost_equal(B, nearest_pos_semi_def(B))
    B = np.diag(np.array([0,0,3]))
    yield assert_array_almost_equal(B, nearest_pos_semi_def(B))
    B = np.diag(np.array([-1,2,3]))
    Bpsd = np.array([[0.,0.,0.],[0.,1.75,0.],[0.,0.,2.75]])
    yield assert_array_almost_equal(Bpsd, nearest_pos_semi_def(B))
    B = np.diag(np.array([-1,-2,3]))
    Bpsd = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,2.]])
    yield assert_array_almost_equal(Bpsd, nearest_pos_semi_def(B))
    B = np.diag(np.array([-1.e-11,0,1000]))
    Bpsd = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,1000.]])
    yield assert_array_almost_equal(Bpsd, nearest_pos_semi_def(B))
    B = np.diag(np.array([-1,-2,-3]))
    Bpsd = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    yield assert_array_almost_equal(Bpsd, nearest_pos_semi_def(B))


