''' Testing utils module '''

import numpy as np

from nose.tools import assert_true, assert_false, assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.io.utils import rec2dict

def test_rec2dict():
    dt = [('x', 'i4'), ('s', 'S10')]
    ra = np.zeros((), dt)
    ra['x'] = 10
    ra['s'] = 'string'
    d = rec2dict(ra)
    yield assert_equal, d, {'x': 10, 's': 'string'}
    ra = np.zeros((2,), dt)
    yield assert_raises, ValueError, rec2dict, ra
    
