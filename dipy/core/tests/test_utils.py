""" Testing utility functions

"""

import numpy as np

from dipy.core.utils import matlab_sph2cart, matlab_cart2sph

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric


@parametric
def test_sph_cart():
    pi = np.pi
    two_pi = pi * 2
    for az in np.linspace(0, two_pi, 10):
        for zen in np.linspace(0, pi, 10):
            x, y, z = matlab_sph2cart(az, zen)
            az2, zen2, r = matlab_cart2sph(x, y, z)
            yield assert_array_almost_equal(az, az2)
            yield assert_array_almost_equal(zen, zen2)
