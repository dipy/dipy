""" Testing qball 

"""

import numpy as np

import dipy.core.qball as qball

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric

@parametric
def test_sph_harm_ind_list():
    m_list, n_list = qball.sph_harm_ind_list(8)
    yield assert_equal(m_list.shape, n_list.shape)
    yield assert_equal(m_list.ndim, 2)
    yield assert_equal(m_list.shape, (45,1))
    yield assert_true(np.all(np.abs(m_list) <= n_list))
    yield assert_array_equal(n_list % 2, 0)
    yield assert_raises(ValueError, qball.sph_harm_ind_list, 1)

@parametric
def test_real_sph_harm():
    real_sh = qball.real_sph_harm(0, 0, 0, 0)
    yield assert_true(True)


