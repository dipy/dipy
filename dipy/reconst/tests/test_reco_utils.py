"""Testing reconstruction utilities."""

import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_raises

from dipy.reconst.recspeed import adj_to_countarrs, argmax_from_countarrs, fmls_afd


def test_adj_countarrs():
    adj = [[0, 1, 2], [2, 3], [4, 5, 6, 7]]
    counts, inds = adj_to_countarrs(adj)
    assert_array_equal(counts, [3, 2, 4])
    assert_equal(counts.dtype.type, np.uint32)
    assert_array_equal(inds, [0, 1, 2, 2, 3, 4, 5, 6, 7])
    assert_equal(inds.dtype.type, np.uint32)


def test_argmax_from_countarrs():
    # basic case
    vals = np.arange(10, dtype=float)
    vertinds = np.arange(10, dtype=np.uint32)
    adj_counts = np.ones((10,), dtype=np.uint32)
    adj_inds_raw = np.arange(10, dtype=np.uint32)[::-1]
    # when contiguous - OK
    adj_inds = adj_inds_raw.copy()
    argmax_from_countarrs(vals, vertinds, adj_counts, adj_inds)
    # yield assert_array_equal(inds, [5, 6, 7, 8, 9])

    # test for errors - first - not contiguous
    with assert_raises(ValueError):
        argmax_from_countarrs(vals, vertinds, adj_counts, adj_inds_raw)

    # too few vertices
    with assert_raises(ValueError):
        argmax_from_countarrs(vals, vertinds[:-1], adj_counts, adj_inds)

    # adj_inds too short
    with assert_raises(IndexError):
        argmax_from_countarrs(vals, vertinds, adj_counts, adj_inds[:-1])

    # vals too short
    with assert_raises(IndexError):
        argmax_from_countarrs(vals[:-1], vertinds, adj_counts, adj_inds)


def test_fmls_afd():
    # Path graph 0-1-2-3 with direction 4 attached to 1 and 5 isolated
    adj_indptr = np.array([0, 1, 4, 6, 7, 8, 8], dtype=np.intp)
    adj_indices = np.array([1, 0, 2, 4, 1, 3, 2, 1], dtype=np.intp)
    weights = np.array([1, 2, 1, 1, 1, 1], dtype=float)
    odf = np.array(
        [
            [3.0, 1.0, 2.0, 0.5, 0.9, -0.2],
            # The amplitude of largest magnitude is negative: no lobes
            [3.0, 1.0, 2.0, 0.5, 0.9, -3.5],
        ]
    )
    order = np.argsort(-odf, axis=-1, kind="stable")

    # Direction 1 borders the lobes of 0 and 2: it adds to the lobe with the
    # largest peak but does not join it, so direction 4 starts a third lobe.
    afd = fmls_afd(odf, order, weights, adj_indptr, adj_indices, 4, 0, 0)
    assert_array_equal(afd, [[5.0, 2.5, 0.9, 0], [0, 0, 0, 0]])

    afd = fmls_afd(odf[:1], order[:1], weights, adj_indptr, adj_indices, 4, 1, 0)
    assert_array_equal(afd, [[5.0, 2.5, 0, 0]])
    afd = fmls_afd(odf[:1], order[:1], weights, adj_indptr, adj_indices, 4, 0, 3)
    assert_array_equal(afd, [[5.0, 0, 0, 0]])
    afd = fmls_afd(odf[:1], order[:1], weights, adj_indptr, adj_indices, 2, 0, 0)
    assert_array_equal(afd, [[5.0, 2.5]])

    with assert_raises(IndexError):
        fmls_afd(odf, order + 1, weights, adj_indptr, adj_indices, 4, 0, 0)
    with assert_raises(IndexError):
        fmls_afd(odf, order, weights, adj_indptr, adj_indices + 2, 4, 0, 0)
    with assert_raises(ValueError):
        fmls_afd(odf, order, weights[:-1], adj_indptr, adj_indices, 4, 0, 0)
