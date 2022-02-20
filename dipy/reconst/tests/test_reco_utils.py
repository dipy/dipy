"""Testing reconstruction utilities."""

import numpy as np

from dipy.reconst.recspeed import (adj_to_countarrs,
                                   argmax_from_countarrs)

from numpy.testing import assert_array_equal, assert_equal


def test_adj_countarrs():
    adj = [[0, 1, 2],
           [2, 3],
           [4, 5, 6, 7]]
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
    #
    # The tests below cause odd errors and segfaults with numpy SVN
    # vintage June 2010 (sometime after 1.4.0 release) - see
    # http://groups.google.com/group/cython-users/browse_thread/thread/624c696293b7fe44?pli=1
    """
    yield assert_raises(ValueError,
                        argmax_from_countarrs,
                        vals,
                        vertinds,
                        adj_counts,
                        adj_inds_raw)
    # too few vertices
    yield assert_raises(ValueError,
                        argmax_from_countarrs,
                        vals,
                        vertinds[:-1],
                        adj_counts,
                        adj_inds)
    # adj_inds too short
    yield assert_raises(IndexError,
                        argmax_from_countarrs,
                        vals,
                        vertinds,
                        adj_counts,
                        adj_inds[:-1])
    # vals too short
    yield assert_raises(IndexError,
                        argmax_from_countarrs,
                        vals[:-1],
                        vertinds,
                        adj_counts,
                        adj_inds)
                        """
