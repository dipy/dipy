from __future__ import division, print_function, absolute_import

from dipy.utils.six.moves import xrange

from nose.tools import assert_equal, assert_raises, assert_true, assert_false
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_equal)

import numpy as np
from dipy.reconst.interpolate import (NearestNeighborInterpolator, TriLinearInterpolator,
                                      OutsideImage)


def test_NearestNeighborInterpolator():
    # Place integers values at the center of every voxel
    l, m, n, o = np.ogrid[0:6.01, 0:6.01, 0:6.01, 0:4]
    data = l + m + n + o

    nni = NearestNeighborInterpolator(data, (1, 1, 1))
    a, b, c = np.mgrid[.5:6.5:1.6, .5:6.5:2.7, .5:6.5:3.8]
    for ii in xrange(a.size):
        x = a.flat[ii]
        y = b.flat[ii]
        z = c.flat[ii]
        expected_result = int(x) + int(y) + int(z) + o.ravel()
        assert_array_equal(nni[x, y, z], expected_result)
        ind = np.array([x, y, z])
        assert_array_equal(nni[ind], expected_result)
    assert_raises(OutsideImage, nni.__getitem__, (-.1, 0, 0))
    assert_raises(OutsideImage, nni.__getitem__, (0, 8.2, 0))


def test_TriLinearInterpolator():
    # Place (0, 0, 0) at the bottom left of the image
    l, m, n, o = np.ogrid[.5:6.51, .5:6.51, .5:6.51, 0:4]
    data = l + m + n + o
    data = data.astype("float32")

    tli = TriLinearInterpolator(data, (1, 1, 1))
    a, b, c = np.mgrid[.5:6.5:1.6, .5:6.5:2.7, .5:6.5:3.8]
    for ii in xrange(a.size):
        x = a.flat[ii]
        y = b.flat[ii]
        z = c.flat[ii]
        expected_result = x + y + z + o.ravel()
        assert_array_almost_equal(tli[x, y, z], expected_result, decimal=5)
        ind = np.array([x, y, z])
        assert_array_almost_equal(tli[ind], expected_result)

    # Index at 0
    expected_value = np.arange(4) + 1.5
    assert_array_almost_equal(tli[0, 0, 0], expected_value)
    # Index at shape
    expected_value = np.arange(4) + (6.5 * 3)
    assert_array_almost_equal(tli[7, 7, 7], expected_value)

    assert_raises(OutsideImage, tli.__getitem__, (-.1, 0, 0))
    assert_raises(OutsideImage, tli.__getitem__, (0, 7.01, 0))
