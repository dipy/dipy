""" Testing utility functions

"""

import numpy as np

from dipy.core.utils import (matlab_sph2cart, matlab_cart2sph,
                             sphere2cart, cart2sphere)

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
def test_matlab_sph_cart():
    for pt in sphere_points:
        az, zen, r = matlab_cart2sph(*pt)
        xyz = matlab_sph2cart(az, zen, 1)
        yield assert_array_almost_equal(xyz, pt)
    # result from matlab output
    # >> [x, y, z] = sph2cart(0.2, 0.4, 2.2);
    # >> fprintf('%9.8f, ', [x, y, z]); fprintf('\n');
    # 1.98594241, 0.40257046, 0.85672035,
    a_z_r = matlab_sph2cart(0.2, 0.4, 2.2)
    yield assert_array_almost_equal(a_z_r,
                                    (1.98594241, 0.40257046, 0.85672035))
