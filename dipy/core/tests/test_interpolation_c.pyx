import numpy as np
import numpy.testing as npt

from dipy.core.interpolation import trilinear_interpolate4d
from dipy.core.interpolation cimport trilinear_interpolate3d_c


def test_trilinear_interpolate3d_c():
    rng = np.random.default_rng(1234)
    data64 = rng.random((4, 5, 6))
    data32 = data64.astype(np.float32)
    cdef double[:, :, :] d64 = data64
    cdef float[:, :, :] d32 = data32
    cdef double[::1] point = np.empty(3)
    cdef double res64 = 0, res32 = 0

    # inside, including within half a voxel of the borders
    points = rng.uniform(-0.5, np.array(data64.shape) - 0.5, (200, 3))
    for p in points:
        point[0], point[1], point[2] = p
        npt.assert_equal(trilinear_interpolate3d_c(d64, &point[0], &res64), 0)
        npt.assert_equal(trilinear_interpolate3d_c(d32, &point[0], &res32), 0)
        expected = trilinear_interpolate4d(data64[..., None], p)[0]
        npt.assert_almost_equal(res64, expected)
        expected = trilinear_interpolate4d(
            data32.astype(float)[..., None], p)[0]
        npt.assert_almost_equal(res32, expected)

    # outside
    for p in ([-0.6, 0, 0], [0, 4.5, 0], [0, 0, 5.5]):
        point[0], point[1], point[2] = p
        npt.assert_equal(trilinear_interpolate3d_c(d64, &point[0], &res64), -1)
        npt.assert_equal(trilinear_interpolate3d_c(d32, &point[0], &res32), -1)
