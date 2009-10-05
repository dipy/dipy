''' Testing trackvis module '''

from StringIO import StringIO

import numpy as np

from nose.tools import assert_true, assert_false, assert_equal

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.core.streamlines import StreamLine


def test_streamline():
    # testing streamline object
    # object has .xyz, .x .y .z .scalars .properties .n_points
    s = StreamLine(np.zeros((10,3)))
    yield assert_equal, s.x.shape, (10,)
    yield assert_equal, s.y.shape, (10,)
    yield assert_equal, s.z.shape, (10,)
    yield assert_equal, s.scalars, None
    yield assert_array_equal, s.properties, None
    

def test_steam_iter():
    s = StreamLine(np.zeros((10,3)))
    s_tup = tuple(s)
    yield assert_array_equal, s_tup[0], np.zeros((10,3))
    yield assert_equal, s_tup[1:], (None, None)
