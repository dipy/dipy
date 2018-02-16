
import numpy as np
import numpy.testing as npt

from dipy.tracking.local.interpolation import trilinear_interpolate4d


def test_trilinear_interpolate():
    """This tests that the trilinear interpolation returns the correct values.
    """
    a, b, c = np.random.random(3)

    def linear_function(x, y, z):
        return a * x + b * y + c * z

    N = 6
    x, y, z = np.mgrid[:N, :N, :N]
    data = np.empty((N, N, N, 2))
    data[..., 0] = linear_function(x, y, z)
    data[..., 1] = 99.

    # Use a point not near the edges
    point = np.array([2.1, 4.8, 3.3])
    out = trilinear_interpolate4d(data, point)
    expected = [linear_function(*point), 99.]
    npt.assert_array_almost_equal(out, expected)

    # Pass in out ourselves
    out[:] = -1
    trilinear_interpolate4d(data, point, out)
    npt.assert_array_almost_equal(out, expected)

    # use a point close to an edge
    point = np.array([-.1, -.1, -.1])
    expected = [0., 99.]
    out = trilinear_interpolate4d(data, point)
    npt.assert_array_almost_equal(out, expected)

    # different edge
    point = np.array([2.4, 5.4, 3.3])
    # On the edge 5.4 get treated as the max y value, 5.
    expected = [linear_function(point[0], 5., point[2]), 99.]
    out = trilinear_interpolate4d(data, point)
    npt.assert_array_almost_equal(out, expected)

    # Test index errors
    point = np.array([2.4, 5.5, 3.3])
    npt.assert_raises(IndexError, trilinear_interpolate4d, data, point)
    point = np.array([2.4, -1., 3.3])
    npt.assert_raises(IndexError, trilinear_interpolate4d, data, point)


if __name__ == "__main__":
    npt.run_module_suite()
