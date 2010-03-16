""" Testing qball 

"""

import numpy as np

import dipy.core.qball as qball

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric


@parametric
def test_real_sph_harm():
    real_sh = qball.real_sph_harm(0, 0, 0, 0)
    yield assert_true(True)
    yield assert_false(True)


