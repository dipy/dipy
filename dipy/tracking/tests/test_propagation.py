import numpy as np

from dipy.data import default_sphere
from dipy.tracking.propspeed import ndarray_offset, eudx_both_directions

from numpy.testing import assert_equal, assert_raises


def stepped_1d(arr_1d):
    # Make a version of `arr_1d` which is not contiguous
    return np.vstack((arr_1d, arr_1d)).ravel(order='F')[::2]


def test_offset():
    # Test ndarray_offset function
    for dt in (np.int32, np.float64):
        index = np.array([1, 1], dtype=np.intp)
        A = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=dt)
        strides = np.array(A.strides, np.intp)
        i_size = A.dtype.itemsize
        assert_equal(ndarray_offset(index, strides, 2, i_size), 4)
        assert_equal(A.ravel()[4], A[1, 1])
        # Index and strides arrays must be C-continuous. Test this is enforced
        # by using non-contiguous versions of the input arrays.
        assert_raises(ValueError, ndarray_offset,
                      stepped_1d(index), strides, 2, i_size)
        assert_raises(ValueError, ndarray_offset,
                      index, stepped_1d(strides), 2, i_size)


def test_eudx_both_directions_errors():
    # Test error conditions for both directions function
    sphere = default_sphere
    seed = np.zeros(3, np.float64)
    qa = np.zeros((4, 5, 6, 7), np.float64)
    ind = qa.copy()
    # All of seed, qa, ind, odf_vertices must be C-contiguous.  Check by
    # passing in versions that aren't C contiguous
    assert_raises(ValueError, eudx_both_directions,
                  stepped_1d(seed), 0, qa, ind, sphere.vertices, 0.5, 0.1,
                  1., 1., 2)
    assert_raises(ValueError, eudx_both_directions,
                  seed, 0, qa[..., ::2], ind, sphere.vertices, 0.5, 0.1,
                  1., 1., 2)
    assert_raises(ValueError, eudx_both_directions,
                  seed, 0, qa, ind[..., ::2], sphere.vertices, 0.5, 0.1,
                  1., 1., 2)
    assert_raises(ValueError, eudx_both_directions,
                  seed, 0, qa, ind, sphere.vertices[::2], 0.5, 0.1,
                  1., 1., 2)
