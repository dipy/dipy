""" Testing diffusion parameter processing

"""

import numpy as np

from dipy.io.dwiparams import B2q
from dipy.io.vectors import vector_norm
from dipy.io.dwiparams import nearest_positive_semi_definite

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric


@parametric
def test_b2q():
    # conversion of b matrix to q
    q = np.array([1,2,3])
    B = np.outer(q, q) / vector_norm(q)
    yield assert_array_almost_equal(q, B2q(B))
    q = np.array([1,2,3])
    # check that the sign of the vector as positive x convention
    B = np.outer(-q, -q) / vector_norm(q)
    yield assert_array_almost_equal(q, B2q(B))
    q = np.array([-1, 2, 3])
    B = np.outer(q, q) / vector_norm(q)
    yield assert_array_almost_equal(-q, B2q(B))
    B = np.eye(3) * -1
    yield assert_raises(ValueError, B2q, B)
    # no error if we up the tolerance
    q = B2q(B, tol=1)


@parametric    
def test_nearest_positive_semi_definite():
    B = np.diag(np.array([1,2,3]))
    yield assert_array_almost_equal(B, nearest_positive_semi_definite(B))
    B = np.diag(np.array([0,2,3]))
    yield assert_array_almost_equal(B, nearest_positive_semi_definite(B))
    B = np.diag(np.array([0,0,3]))
    yield assert_array_almost_equal(B, nearest_positive_semi_definite(B))
    B = np.diag(np.array([-1,2,3]))
    Bpsd = np.array([[0.,0.,0.],[0.,1.75,0.],[0.,0.,2.75]])
    yield assert_array_almost_equal(Bpsd, nearest_positive_semi_definite(B))
    B = np.diag(np.array([-1,-2,3]))
    Bpsd = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,2.]])
    yield assert_array_almost_equal(Bpsd, nearest_positive_semi_definite(B))
    B = np.diag(np.array([-1.e-10,0,1000]))
    Bpsd = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,1000.]])
    yield assert_array_almost_equal(Bpsd, nearest_positive_semi_definite(B))
    B = np.diag(np.array([-1,-2,-3]))
    Bpsd = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    yield assert_array_almost_equal(Bpsd, nearest_positive_semi_definite(B))
