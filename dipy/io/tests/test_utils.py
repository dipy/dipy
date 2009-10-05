''' Testing utils module '''

import numpy as np

from nose.tools import assert_true, assert_false, assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.io.utils import rec2dict

def test_rec2dict():
    dt = [('x', 'i4'), ('yz', 'f8', 2), ('s', 'S10')]
    ra = np.zeros((), dt)
    ra['x'] = 10
    ra['yz'] = [1,2]
    ra['s'] = 'string'
    d = rec2dict(ra)
    yield assert_equal, type(d), type({})
    yield assert_equal, d['x'], 10
    yield assert_array_equal, d['yz'], [1, 2]
    yield assert_equal, d['s'], 'string'


