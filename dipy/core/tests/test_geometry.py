""" Testing utility functions

"""

import numpy as np

from dipy.core.geometry import (sphere2cart, cart2sphere,
                                nearest_pos_semi_def,
                                sphere_distance,
                                cart_distance,
                                vector_cosine,
                                )

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric, sphere_points


@parametric
def test_sphere_cart():
    # test arrays of points
    rs, thetas, phis = cart2sphere(*(sphere_points.T))
    xyz = sphere2cart(rs, thetas, phis)
    yield assert_array_almost_equal(xyz, sphere_points.T)
    # test radius estimation
    big_sph_pts = sphere_points * 10.4
    rs, thetas, phis = cart2sphere(*big_sph_pts.T)
    yield assert_array_almost_equal(rs, 10.4)
    xyz = sphere2cart(rs, thetas, phis)
    yield assert_array_almost_equal(xyz, big_sph_pts.T, 6)
    # test a scalar point
    pt = sphere_points[3]
    r, theta, phi = cart2sphere(*pt)
    xyz = sphere2cart(r, theta, phi)
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


@parametric
def test_cart_distance():
    a = [0, 1]
    b = [1, 0]
    yield assert_array_almost_equal(cart_distance(a, b),
                                    np.sqrt(2))
    yield assert_array_almost_equal(cart_distance([1,0], [-1,0]),
                                    2)
    pts1 = [2, 1, 0]
    pts2 = [0, 1, -2]
    yield assert_array_almost_equal(
        cart_distance(pts1, pts2), np.sqrt(8))
    pts2 = [[0, 1, -2],
            [-2, 1, 0]]
    yield assert_array_almost_equal(
        cart_distance(pts1, pts2), [np.sqrt(8), 4])
    

@parametric
def test_sphere_distance():
    # make a circle, go around...
    radius = 3.2
    n = 5000
    n2 = n / 2
    # pi at point n2 in array
    angles = np.linspace(0, np.pi*2, n, endpoint=False)
    x = np.sin(angles) * radius
    y = np.cos(angles) * radius
    # dists around half circle, including pi
    half_x = x[:n2+1]
    half_y = y[:n2+1]
    half_dists = np.sqrt(np.diff(half_x)**2 + np.diff(half_y)**2)
    # approximate distances from 0 to pi (not including 0)
    csums = np.cumsum(half_dists)
    # concatenated with distances from pi to 0 again
    cdists = np.r_[0, csums, csums[-2::-1]]
    # check approximation close to calculated
    sph_d = sphere_distance([0,radius], np.c_[x, y])
    yield assert_array_almost_equal(cdists, sph_d, decimal=5)
    # Now check with passed radius
    sph_d = sphere_distance([0,radius], np.c_[x, y], radius=radius)
    yield assert_array_almost_equal(cdists, sph_d, decimal=5)
    # Check points not on surface raises error when asked for
    yield assert_raises(ValueError, sphere_distance, [1, 0],
                        [0, 2])
    # Not when not asked for
    sph_d = sphere_distance([1, 0], [0,2], None, False)
    # Error when radii don't match passed radius
    yield assert_raises(ValueError, sphere_distance, [1, 0],
                        [0, 1], 2.0)
    
    
@parametric
def test_vector_cosine():
    a = [0, 1]
    b = [1, 0]
    yield assert_array_almost_equal(vector_cosine(a, b), 0)
    yield assert_array_almost_equal(vector_cosine([1,0], [-1,0]),
                                    -1)
    yield assert_array_almost_equal(vector_cosine([1,0], [1,1]),
                                    1 / np.sqrt(2))
    yield assert_array_almost_equal(vector_cosine([2,0], [-4,0]),
                                    -1)
    pts1 = [2, 1, 0]
    pts2 = [-2, -1, 0]
    yield assert_array_almost_equal(
        vector_cosine(pts1, pts2), -1)
    pts2 = [[-2, -1, 0],
            [2, 1, 0]]
    yield assert_array_almost_equal(
        vector_cosine(pts1, pts2), [-1, 1])
    

