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
    


def test_stream_iter():
    s = StreamLine(np.zeros((10,3)))
    s_tup = tuple(s)
    #s_tup[0] returns (np.array([0.0, 0.0, 0.0]), None, None)
    yield assert_equal, s_tup[0][0][0],0.0
    yield assert_equal, s_tup[0][1], None
    


def test_stream_midpoint():
    s = StreamLine(np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4]]))
    s2 = StreamLine(np.array([[0,0,0],[1,1,0],[2,1,0],[3,0,0]]))       
    s3 = StreamLine(np.array([[1,1,1],[2,2,2],[3,3,3]]))
    
    yield assert_array_almost_equal, s.midpoint(), np.array([2.5,2.5,2.5])
    yield assert_array_almost_equal, s2.midpoint(), np.array([1.5,1.0,0.0])
    yield assert_array_almost_equal, s3.midpoint(), np.array([2,2,2])


 
