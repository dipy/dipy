import numpy as np
import numpy.testing as npt
import warnings

from scipy.ndimage import map_coordinates
from dipy.core.subdivide_octahedron import create_unit_sphere
from dipy.core.interpolation import (trilinear_interpolate4d,
                                     interpolate_scalar_2d,
                                     interpolate_scalar_3d,
                                     interpolate_vector_2d,
                                     interpolate_vector_3d,
                                     interpolate_scalar_nn_2d,
                                     interpolate_scalar_nn_3d,
                                     NearestNeighborInterpolator,
                                     TriLinearInterpolator,
                                     OutsideImage,
                                     map_coordinates_trilinear_iso,
                                     interp_rbf)
from dipy.align import floating
from dipy.testing.decorators import set_random_number_generator


@set_random_number_generator()
def test_trilinear_interpolate(rng):
    """This tests that the trilinear interpolation returns the correct values.
    """
    a, b, c = rng.random(3)

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


@set_random_number_generator(5324989)
def test_interpolate_scalar_2d(rng):
    sz = 64
    target_shape = (sz, sz)
    image = np.empty(target_shape, dtype=floating)
    image[...] = rng.integers(0, 10, np.size(image)).reshape(target_shape)

    extended_image = np.zeros((sz + 2, sz + 2), dtype=floating)
    extended_image[1:sz + 1, 1:sz + 1] = image[...]

    # Select some coordinates inside the image to interpolate at
    nsamples = 200
    locations =\
        rng.random(2 * nsamples).reshape((nsamples, 2)) * (sz + 2) - 1.0
    extended_locations = locations + 1.0  # shift coordinates one voxel

    # Call the implementation under test
    interp, inside = interpolate_scalar_2d(image, locations)

    # Call the reference implementation
    expected = map_coordinates(extended_image, extended_locations.transpose(),
                               order=1)

    npt.assert_array_almost_equal(expected, interp)

    # Test interpolation stability along the boundary
    epsilon = 5e-8
    for k in range(2):
        for offset in [0, sz - 1]:
            delta = ((rng.random(nsamples) * 2) - 1) * epsilon
            locations[:, k] = delta + offset
            locations[:, (k + 1) % 2] = rng.random(nsamples) * (sz - 1)
            interp, inside = interpolate_scalar_2d(image, locations)

            locations[:, k] = offset
            expected = map_coordinates(image, locations.transpose(), order=1)
            npt.assert_array_almost_equal(expected, interp)
            if offset == 0:
                expected_flag = np.array(delta >= 0, dtype=np.int32)
            else:
                expected_flag = np.array(delta <= 0, dtype=np.int32)
            npt.assert_array_almost_equal(expected_flag, inside)


@set_random_number_generator(1924781)
def test_interpolate_scalar_nn_2d(rng):
    sz = 64
    target_shape = (sz, sz)
    image = np.empty(target_shape, dtype=floating)
    image[...] = rng.integers(0, 10, np.size(image)).reshape(target_shape)
    # Select some coordinates to interpolate at
    nsamples = 200
    locations =\
        rng.random(2 * nsamples).reshape((nsamples, 2)) * (sz + 2) - 1.0

    # Call the implementation under test
    interp, inside = interpolate_scalar_nn_2d(image, locations)

    # Call the reference implementation
    expected = map_coordinates(image, locations.transpose(), order=0)

    npt.assert_array_almost_equal(expected, interp)

    # Test the 'inside' flag
    for i in range(nsamples):
        if (locations[i, 0] < 0 or locations[i, 0] > (sz - 1)) or\
           (locations[i, 1] < 0 or locations[i, 1] > (sz - 1)):
            npt.assert_equal(inside[i], 0)
        else:
            npt.assert_equal(inside[i], 1)


@set_random_number_generator(3121121)
def test_interpolate_scalar_nn_3d(rng):
    sz = 64
    target_shape = (sz, sz, sz)
    image = np.empty(target_shape, dtype=floating)
    image[...] = rng.integers(0, 10, np.size(image)).reshape(target_shape)
    # Select some coordinates to interpolate at
    nsamples = 200
    locations =\
        rng.random(3 * nsamples).reshape((nsamples, 3)) * (sz + 2) - 1.0

    # Call the implementation under test
    interp, inside = interpolate_scalar_nn_3d(image, locations)

    # Call the reference implementation
    expected = map_coordinates(image, locations.transpose(), order=0)

    npt.assert_array_almost_equal(expected, interp)

    # Test the 'inside' flag
    for i in range(nsamples):
        expected_inside = 1
        for axis in range(3):
            if locations[i, axis] < 0 or locations[i, axis] > (sz - 1):
                expected_inside = 0
                break
        npt.assert_equal(inside[i], expected_inside)


@set_random_number_generator(9216326)
def test_interpolate_scalar_3d(rng):
    sz = 64
    target_shape = (sz, sz, sz)
    image = np.empty(target_shape, dtype=floating)
    image[...] = rng.integers(0, 10, np.size(image)).reshape(target_shape)

    extended_image = np.zeros((sz + 2, sz + 2, sz + 2), dtype=floating)
    extended_image[1:sz + 1, 1:sz + 1, 1:sz + 1] = image[...]

    # Select some coordinates inside the image to interpolate at
    nsamples = 800
    locations =\
        rng.random(3 * nsamples).reshape((nsamples, 3)) * (sz + 2) - 1.0
    extended_locations = locations + 1.0  # shift coordinates one voxel

    # Call the implementation under test
    interp, inside = interpolate_scalar_3d(image, locations)

    # Call the reference implementation
    expected = map_coordinates(extended_image, extended_locations.transpose(),
                               order=1)

    npt.assert_array_almost_equal(expected, interp)

    # Test interpolation stability along the boundary
    epsilon = 5e-8
    for k in range(3):
        for offset in [0, sz - 1]:
            delta = ((rng.random(nsamples) * 2) - 1) * epsilon
            locations[:, k] = delta + offset
            locations[:, (k + 1) % 3] = rng.random(nsamples) * (sz - 1)
            locations[:, (k + 2) % 3] = rng.random(nsamples) * (sz - 1)
            interp, inside = interpolate_scalar_3d(image, locations)

            locations[:, k] = offset
            expected = map_coordinates(image, locations.transpose(), order=1)
            npt.assert_array_almost_equal(expected, interp)

            if offset == 0:
                expected_flag = np.array(delta >= 0, dtype=np.int32)
            else:
                expected_flag = np.array(delta <= 0, dtype=np.int32)
            npt.assert_array_almost_equal(expected_flag, inside)


@set_random_number_generator(7711219)
def test_interpolate_vector_3d(rng):
    sz = 64
    target_shape = (sz, sz, sz)
    field = np.empty(target_shape + (3,), dtype=floating)
    field[...] =\
        rng.integers(0, 10, np.size(field)).reshape(target_shape + (3,))

    extended_field = np.zeros((sz + 2, sz + 2, sz + 2, 3), dtype=floating)
    extended_field[1:sz + 1, 1:sz + 1, 1:sz + 1] = field
    # Select some coordinates to interpolate at
    nsamples = 800
    locations =\
        rng.random(3 * nsamples).reshape((nsamples, 3)) * (sz + 2) - 1.0
    extended_locations = locations + 1

    # Call the implementation under test
    interp, inside = interpolate_vector_3d(field, locations)

    # Call the reference implementation
    expected = np.zeros_like(interp)
    for i in range(3):
        expected[..., i] = map_coordinates(extended_field[..., i],
                                           extended_locations.transpose(),
                                           order=1)

    npt.assert_array_almost_equal(expected, interp)

    # Test interpolation stability along the boundary
    epsilon = 5e-8
    for k in range(3):
        for offset in [0, sz - 1]:
            delta = ((rng.random(nsamples) * 2) - 1) * epsilon
            locations[:, k] = delta + offset
            locations[:, (k + 1) % 3] = rng.random(nsamples) * (sz - 1)
            locations[:, (k + 2) % 3] = rng.random(nsamples) * (sz - 1)
            interp, inside = interpolate_vector_3d(field, locations)

            locations[:, k] = offset
            for i in range(3):
                expected[..., i] = map_coordinates(field[..., i],
                                                   locations.transpose(),
                                                   order=1)
            npt.assert_array_almost_equal(expected, interp)

            if offset == 0:
                expected_flag = np.array(delta >= 0, dtype=np.int32)
            else:
                expected_flag = np.array(delta <= 0, dtype=np.int32)
            npt.assert_array_almost_equal(expected_flag, inside)


@set_random_number_generator(1271244)
def test_interpolate_vector_2d(rng):
    sz = 64
    target_shape = (sz, sz)
    field = np.empty(target_shape + (2,), dtype=floating)
    field[...] =\
        rng.integers(0, 10, np.size(field)).reshape(target_shape + (2,))
    extended_field = np.zeros((sz + 2, sz + 2, 2), dtype=floating)
    extended_field[1:sz + 1, 1:sz + 1] = field
    # Select some coordinates to interpolate at
    nsamples = 200
    locations =\
        rng.random(2 * nsamples).reshape((nsamples, 2)) * (sz + 2) - 1.0
    extended_locations = locations + 1

    # Call the implementation under test
    interp, inside = interpolate_vector_2d(field, locations)

    # Call the reference implementation
    expected = np.zeros_like(interp)
    for i in range(2):
        expected[..., i] = map_coordinates(extended_field[..., i],
                                           extended_locations.transpose(),
                                           order=1)

    npt.assert_array_almost_equal(expected, interp)

    # Test interpolation stability along the boundary
    epsilon = 5e-8
    for k in range(2):
        for offset in [0, sz - 1]:
            delta = ((rng.random(nsamples) * 2) - 1) * epsilon
            locations[:, k] = delta + offset
            locations[:, (k + 1) % 2] = rng.random(nsamples) * (sz - 1)
            interp, inside = interpolate_vector_2d(field, locations)

            locations[:, k] = offset
            for i in range(2):
                expected[..., i] = map_coordinates(field[..., i],
                                                   locations.transpose(),
                                                   order=1)
            npt.assert_array_almost_equal(expected, interp)

            if offset == 0:
                expected_flag = np.array(delta >= 0, dtype=np.int32)
            else:
                expected_flag = np.array(delta <= 0, dtype=np.int32)
            npt.assert_array_almost_equal(expected_flag, inside)


def test_NearestNeighborInterpolator():
    # Place integers values at the center of every voxel
    l, m, n, o = np.ogrid[0:6.01, 0:6.01, 0:6.01, 0:4]
    data = l + m + n + o

    nni = NearestNeighborInterpolator(data, (1, 1, 1))
    a, b, c = np.mgrid[.5:6.5:1.6, .5:6.5:2.7, .5:6.5:3.8]
    for ii in range(a.size):
        x = a.flat[ii]
        y = b.flat[ii]
        z = c.flat[ii]
        expected_result = int(x) + int(y) + int(z) + o.ravel()
        npt.assert_array_equal(nni[x, y, z], expected_result)
        ind = np.array([x, y, z])
        npt.assert_array_equal(nni[ind], expected_result)
    npt.assert_raises(OutsideImage, nni.__getitem__, (-.1, 0, 0))
    npt.assert_raises(OutsideImage, nni.__getitem__, (0, 8.2, 0))


def test_TriLinearInterpolator():
    # Place (0, 0, 0) at the bottom left of the image
    l, m, n, o = np.ogrid[.5:6.51, .5:6.51, .5:6.51, 0:4]
    data = l + m + n + o
    data = data.astype("float32")

    tli = TriLinearInterpolator(data, (1, 1, 1))
    a, b, c = np.mgrid[.5:6.5:1.6, .5:6.5:2.7, .5:6.5:3.8]
    for ii in range(a.size):
        x = a.flat[ii]
        y = b.flat[ii]
        z = c.flat[ii]
        expected_result = x + y + z + o.ravel()
        npt.assert_array_almost_equal(tli[x, y, z], expected_result, decimal=5)
        ind = np.array([x, y, z])
        npt.assert_array_almost_equal(tli[ind], expected_result)

    # Index at 0
    expected_value = np.arange(4) + 1.5
    npt.assert_array_almost_equal(tli[0, 0, 0], expected_value)
    # Index at shape
    expected_value = np.arange(4) + (6.5 * 3)
    npt.assert_array_almost_equal(tli[7, 7, 7], expected_value)

    npt.assert_raises(OutsideImage, tli.__getitem__, (-.1, 0, 0))
    npt.assert_raises(OutsideImage, tli.__getitem__, (0, 7.01, 0))


def test_trilinear_interp_cubic_voxels():

    def stepped_1d(arr_1d):
        # Make a version of `arr_1d` which is not contiguous
        return np.vstack((arr_1d, arr_1d)).ravel(order='F')[::2]

    A = np.ones((17, 17, 17))
    B = np.zeros(3)
    strides = np.array(A.strides, np.intp)
    A[7, 7, 7] = 2
    points = np.array([[0, 0, 0], [7., 7.5, 7.], [3.5, 3.5, 3.5]])
    map_coordinates_trilinear_iso(A, points, strides, 3, B)
    npt.assert_array_almost_equal(B, np.array([1., 1.5, 1.]))
    # All of the input array, points array, strides array and output array must
    # be C-contiguous.  Check by passing in versions that aren't C contiguous
    npt.assert_raises(ValueError, map_coordinates_trilinear_iso,
                      A.copy(order='F'), points, strides, 3, B)
    npt.assert_raises(ValueError, map_coordinates_trilinear_iso,
                      A, points.copy(order='F'), strides, 3, B)
    npt.assert_raises(ValueError, map_coordinates_trilinear_iso,
                      A, points, stepped_1d(strides), 3, B)
    npt.assert_raises(ValueError, map_coordinates_trilinear_iso,
                      A, points, strides, 3, stepped_1d(B))


def test_interp_rbf():
    def data_func(s, a, b):
        return a * np.cos(s.theta) + b * np.sin(s.phi)

    s0 = create_unit_sphere(3)
    s1 = create_unit_sphere(4)
    for a, b in zip([1, 2, 0.5], [1, 0.5, 2]):
        data = data_func(s0, a, b)
        expected = data_func(s1, a, b)
        interp_data_a = interp_rbf(data, s0, s1, norm="angle")
        npt.assert_(np.mean(np.abs(interp_data_a - expected)) < 0.1)

    # Test that using the euclidean norm raises a warning
    # (following
    # https://docs.python.org/2/library/warnings.html#testing-warnings)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        interp_rbf(data, s0, s1, norm="euclidean_norm")
        npt.assert_(len(w) == 1)
        npt.assert_(issubclass(w[-1].category, PendingDeprecationWarning))
        npt.assert_("deprecated" in str(w[-1].message))
