""" Testing reconstruction utilities
"""

import numpy as np

from dipy.core.reconstruction_performance import adj_to_countarrs

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric


@parametric
def test_adj_countarrs():
    adj = [[0, 1, 2],
           [2, 3],
           [4, 5, 6, 7]]
    counts, inds = adj_to_countarrs(adj)
    yield assert_array_equal(counts, [3, 2, 4])
    yield assert_equal(counts.dtype.type, np.uint32)
    yield assert_array_equal(inds, [0, 1, 2, 2, 3, 4, 5, 6, 7])
    yield assert_equal(inds.dtype.type, np.uint32)
    
