""" Testing utility functions

"""

import numpy as np

from dipy.core.geometry import (sphere2cart, cart2sphere,
                                nearest_pos_semi_def,
                                sphere_distance,
                                cart_distance,
                                vector_cosine,
                                lambert_equal_area_projection_polar,
                                circumradius,
                                vec2vec_rotmat,
                                vector_norm,
                                compose_transformations,
                                compose_matrix,
                                decompose_matrix)

from nose.tools import assert_false, assert_equal, assert_raises

from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           run_module_suite)

from dipy.testing import sphere_points
from itertools import permutations


def test_vector_norm():
    A = np.array([[1, 0, 0],
                  [3, 4, 0],
                  [0, 5, 12],
                  [1, 2, 3]])
    expected = np.array([1, 5, 13, np.sqrt(14)])
    assert_array_almost_equal(vector_norm(A), expected)
    expected.shape = (4, 1)
    assert_array_almost_equal(vector_norm(A, keepdims=True), expected)
    assert_array_almost_equal(vector_norm(A.T, axis=0, keepdims=True),
                              expected.T)


def test_sphere_cart():
    # test arrays of points
    rs, thetas, phis = cart2sphere(*(sphere_points.T))
    xyz = sphere2cart(rs, thetas, phis)
    yield assert_array_almost_equal, xyz, sphere_points.T
    # test radius estimation
    big_sph_pts = sphere_points * 10.4
    rs, thetas, phis = cart2sphere(*big_sph_pts.T)
    yield assert_array_almost_equal, rs, 10.4
    xyz = sphere2cart(rs, thetas, phis)
    yield assert_array_almost_equal, xyz, big_sph_pts.T, 6
    #test that result shapes match
    x, y, z =  big_sph_pts.T
    r, theta, phi = cart2sphere(x[:1], y[:1], z)
    yield assert_equal, r.shape, theta.shape
    yield assert_equal, r.shape, phi.shape
    x, y, z = sphere2cart(r[:1], theta[:1], phi)
    yield assert_equal, x.shape, y.shape
    yield assert_equal, x.shape, z.shape
    # test a scalar point
    pt = sphere_points[3]
    r, theta, phi = cart2sphere(*pt)
    xyz = sphere2cart(r, theta, phi)
    yield assert_array_almost_equal, xyz, pt

    # Test full circle on x=0, y=0, z=0
    x, y, z = sphere2cart(*cart2sphere(0., 0., 0.))
    yield assert_array_equal, (x, y, z), (0., 0., 0.)

def test_invert_transform():
    n = 100.
    theta = np.arange(n)/n * np.pi   # Limited to 0,pi
    phi = (np.arange(n)/n - .5) * 2 * np.pi # Limited to 0,2pi
    x, y, z = sphere2cart(1, theta, phi)  # Let's assume they're all unit vectors
    r, new_theta, new_phi = cart2sphere(x, y, z)  # Transform back

    yield assert_array_almost_equal, theta, new_theta
    yield assert_array_almost_equal, phi, new_phi

def test_nearest_pos_semi_def():
    B = np.diag(np.array([1,2,3]))
    yield assert_array_almost_equal, B, nearest_pos_semi_def(B)
    B = np.diag(np.array([0,2,3]))
    yield assert_array_almost_equal, B, nearest_pos_semi_def(B)
    B = np.diag(np.array([0,0,3]))
    yield assert_array_almost_equal, B, nearest_pos_semi_def(B)
    B = np.diag(np.array([-1,2,3]))
    Bpsd = np.array([[0.,0.,0.],[0.,1.75,0.],[0.,0.,2.75]])
    yield assert_array_almost_equal, Bpsd, nearest_pos_semi_def(B)
    B = np.diag(np.array([-1,-2,3]))
    Bpsd = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,2.]])
    yield assert_array_almost_equal, Bpsd, nearest_pos_semi_def(B)
    B = np.diag(np.array([-1.e-11,0,1000]))
    Bpsd = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,1000.]])
    yield assert_array_almost_equal, Bpsd, nearest_pos_semi_def(B)
    B = np.diag(np.array([-1,-2,-3]))
    Bpsd = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    yield assert_array_almost_equal, Bpsd, nearest_pos_semi_def(B)

def test_cart_distance():
    a = [0, 1]
    b = [1, 0]
    yield assert_array_almost_equal, cart_distance(a, b), np.sqrt(2)
    yield assert_array_almost_equal, cart_distance([1,0], [-1,0]), 2
    pts1 = [2, 1, 0]
    pts2 = [0, 1, -2]
    yield assert_array_almost_equal, cart_distance(pts1, pts2), np.sqrt(8)
    pts2 = [[0, 1, -2],
            [-2, 1, 0]]
    yield assert_array_almost_equal, cart_distance(pts1, pts2), [np.sqrt(8), 4]

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
    yield assert_array_almost_equal, cdists, sph_d
    # Now check with passed radius
    sph_d = sphere_distance([0,radius], np.c_[x, y], radius=radius)
    yield assert_array_almost_equal, cdists, sph_d
    # Check points not on surface raises error when asked for
    yield assert_raises, ValueError, sphere_distance, [1, 0], [0, 2]
    # Not when check is disabled
    sph_d = sphere_distance([1, 0], [0,2], None, False)
    # Error when radii don't match passed radius
    yield assert_raises, ValueError, sphere_distance, [1, 0], [0, 1], 2.0

def test_vector_cosine():
    a = [0, 1]
    b = [1, 0]
    yield assert_array_almost_equal, vector_cosine(a, b), 0
    yield assert_array_almost_equal, vector_cosine([1,0], [-1,0]), -1
    yield assert_array_almost_equal, vector_cosine([1,0], [1,1]), 1/np.sqrt(2)
    yield assert_array_almost_equal, vector_cosine([2,0], [-4,0]), -1
    pts1 = [2, 1, 0]
    pts2 = [-2, -1, 0]
    yield assert_array_almost_equal, vector_cosine(pts1, pts2), -1
    pts2 = [[-2, -1, 0],
            [2, 1, 0]]
    yield assert_array_almost_equal, vector_cosine(pts1, pts2), [-1, 1]
    # test relationship with correlation
    # not the same if non-zero vector mean
    a = np.random.uniform(size=(100,))
    b = np.random.uniform(size=(100,))
    cc = np.corrcoef(a, b)[0,1]
    vcos = vector_cosine(a, b)
    yield assert_false, np.allclose(cc, vcos)
    # is the same if zero vector mean
    a_dm = a - np.mean(a)
    b_dm = b - np.mean(b)
    vcos = vector_cosine(a_dm, b_dm)
    yield assert_array_almost_equal, cc, vcos

def test_lambert_equal_area_projection_polar():

    theta = np.repeat(np.pi/3,10)
    phi = np.linspace(0,2*np.pi,10)
    # points sit on circle with co-latitude pi/3 (60 degrees)
    leap = lambert_equal_area_projection_polar(theta,phi)
    yield assert_array_almost_equal, np.sqrt(np.sum(leap**2,axis=1)), \
          np.array([ 1.,1.,1.,1.,1.,1.,1.,1.,1.,1.])
    # points map onto the circle of radius 1

def test_lambert_equal_area_projection_cart():

    xyz = np.array([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]])
    # points sit on +/-1 on all 3 axes

    r,theta,phi = cart2sphere(*xyz.T)

    leap = lambert_equal_area_projection_polar(theta,phi)
    r2 = np.sqrt(2)
    yield assert_array_almost_equal, np.sqrt(np.sum(leap**2,axis=1)), \
          np.array([ r2,r2,0,r2,r2,2])
    # x and y =+/-1 map onto circle of radius sqrt(2)
    # z=1 maps to origin, and z=-1 maps to (an arbitrary point on) the
    # outer circle of radius 2

def test_circumradius():

    yield assert_array_almost_equal, np.sqrt(0.5), circumradius(np.array([0,2,0]),np.array([2,0,0]),np.array([0,0,0]))


def test_vec2vec_rotmat():
    a = np.array([1, 0, 0])
    for b in np.array([[0, 0, 1], [-1, 0, 0], [1, 0, 0]]):
        R = vec2vec_rotmat(a, b)
        assert_array_almost_equal(np.dot(R, a), b)


def test_compose_transformations():

    A = np.eye(4)
    A[0, -1] = 10

    B = np.eye(4)
    B[0, -1] = -20

    C = np.eye(4)
    C[0, -1] = 10

    CBA = compose_transformations(A, B, C)

    assert_array_equal(CBA, np.eye(4))

    assert_raises(ValueError, compose_transformations, A)


def test_compose_decompose_matrix():

    for translate in permutations(40 * np.random.rand(3), 3):
        for angles in permutations(np.deg2rad(90 * np.random.rand(3)), 3):
            for shears in permutations(3 * np.random.rand(3), 3):
                for scale in permutations(3 * np.random.rand(3), 3):

                    mat = compose_matrix(translate=translate, angles=angles,
                                         shear=shears, scale=scale)
                    sc, sh, ang, trans, _ = decompose_matrix(mat)

                    assert_array_almost_equal(translate, trans)
                    assert_array_almost_equal(angles, ang)

                    assert_array_almost_equal(shears, sh)
                    assert_array_almost_equal(scale, sc)


if __name__ == '__main__':

    run_module_suite()

