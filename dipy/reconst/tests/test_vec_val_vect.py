import numpy as np
from numpy.random import randn
from numpy.testing import assert_almost_equal

from dipy.reconst.vec_val_sum import vec_val_vect


def make_vecs_vals(shape):
    return randn(*shape), randn(*(shape[:-2] + shape[-1:]))


def test_vec_val_vect():
    for shape0 in ((10,), (100,), (10, 12), (12, 10, 5)):
        for shape1 in ((3, 3), (4, 3), (3, 4)):
            shape = shape0 + shape1
            evecs, evals = make_vecs_vals(shape)
            res1 = np.einsum('...ij,...j,...kj->...ik', evecs, evals, evecs)
            assert_almost_equal(res1, vec_val_vect(evecs, evals))


def dumb_sum(vecs, vals):
    N, rows, cols = vecs.shape
    res2 = np.zeros((N, rows, rows))
    for i in range(N):
        Q = vecs[i]
        L = vals[i]
        res2[i] = np.dot(Q, np.dot(np.diag(L), Q.T))
    return res2


def test_vec_val_vect_dumber():
    for shape0 in ((10,), (100,)):
        for shape1 in ((3, 3), (4, 3), (3, 4)):
            shape = shape0 + shape1
            evecs, evals = make_vecs_vals(shape)
            res1 = dumb_sum(evecs, evals)
            assert_almost_equal(res1, vec_val_vect(evecs, evals))
