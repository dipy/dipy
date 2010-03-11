""" Testing diffusion parameter processing

"""

import numpy as np

from dipy.io.dwiparams import B2q

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric


@parametric
def test_b2q():
    # conversion of b matrix to q
    q = np.array([1,2,3])
    B = np.outer(q, q)
    yield assert_array_almost_equal(q, B2q(B))
    
