from nose.tools import assert_equal, assert_raises, assert_true, assert_false
from numpy.testing import assert_array_equal, assert_array_almost_equal

import numpy as np
from dipy.reconst.interpolate import NearestNeighborInterpolator

def test_NearestNeighborInterpolator():
    a, b, c = np.ogrid[0:6,0:6,0:6]
    data = a+b+c

    nni = NearestNeighborInterpolator(data, (1,1,1))
    a, b, c = np.mgrid[0:6:.6, 0:6:.6, .0:6:.6]
    for ii in xrange(a.size):
        x = a.flat[ii]
        y = b.flat[ii]
        z = c.flat[ii]
        expected_result = int(x) + int(y) + int(z)
        assert nni[x, y, z] == expected_result
        ind = np.array([x, y, z])
        assert nni[ind] == expected_result
    assert_raises(IndexError, nni.__getitem__, (-.1, 0, 0))
