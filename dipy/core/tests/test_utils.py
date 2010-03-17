""" Testing utility functions

"""

import numpy as np

from dipy.core.utils import sph2cart, cart2sph

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric


@parametric
def test_sph_cart():
    pi = np.pi
    two_pi = pi * 2
    for theta in np.linspace(0, two_pi, 10):
        for phi in np.linspace(0, pi, 10):
            x, y, z = sph2cart(theta, phi)
            t2, p2, r = cart2sph(x, y, z)
            theta_pies = theta / pi
            yield assert_array_almost_equal(theta, t2)
            yield assert_array_almost_equal(phi, p2)
            
