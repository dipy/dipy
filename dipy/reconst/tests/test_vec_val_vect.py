import numpy as np
from numpy.random import randn
from numpy.testing import assert_array_equal, dec

from ..vec_val_sum import vec_val_vect

def make_vecs_vals(shape):
    return randn(*(shape + (3, 3))), randn(*(shape + (3,)))


try:
    np.einsum
except AttributeError:
    with_einsum = dec.skipif(True, "Need einsum for benchmark")
else:
    with_einsum = lambda f : f


@with_einsum
def test_vec_val_vect():
    for shape in ((10,), (100,), (10, 12), (12, 10, 5)):
        evecs, evals = make_vecs_vals(shape)
        res1 = np.einsum('...ij,...j,...kj->...ik', evecs, evals, evecs)
        assert_array_equal(res1, vec_val_vect(evecs, evals))


def dumb_sum(vecs, vals):
    N = vecs.shape[0]
    res2 = np.zeros(vecs.shape)
    for i in range(N):
        Q = vecs[i]
        L = vals[i]
        res2[i] = np.dot(Q * L, Q.T)
    return res2


def test_vec_val_vect_dumber():
    for shape in ((10,), (100,)):
        evecs, evals = make_vecs_vals(shape)
        res1 = dumb_sum(evecs, evals)
        assert_array_equal(res1, vec_val_vect(evecs, evals))
