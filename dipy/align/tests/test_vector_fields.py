import numpy as np
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           assert_almost_equal,
                           assert_equal,
                           assert_raises)
from scipy.ndimage import map_coordinates
from nibabel.affines import apply_affine, from_matvec
from dipy.core import geometry
from dipy.align import floating
from dipy.align import imwarp
from dipy.align import vector_fields as vfu
from dipy.align.transforms import regtransforms
from dipy.align.parzenhist import sample_domain_regular
from dipy.testing.decorators import set_random_number_generator


@set_random_number_generator(3921116)
def test_random_displacement_field_2d(rng):
    from_shape = (25, 32)
    to_shape = (33, 29)

    # Create grid coordinates
    x_0 = np.asarray(range(from_shape[0]))
    x_1 = np.asarray(range(from_shape[1]))
    X = np.empty((3,) + from_shape, dtype=np.float64)
    O = np.ones(from_shape)
    X[0, ...] = x_0[:, None] * O
    X[1, ...] = x_1[None, :] * O
    X[2, ...] = 1

    # Create an arbitrary image-to-space transform
    t = 0.15  # translation factor

    trans = np.array([[1, 0, -t * from_shape[0]],
                      [0, 1, -t * from_shape[1]],
                      [0, 0, 1]])
    trans_inv = np.linalg.inv(trans)

    for theta in [-1 * np.pi / 6.0, 0.0, np.pi / 5.0]:  # rotation angle
        for s in [0.83, 1.3, 2.07]:  # scale
            ct = np.cos(theta)
            st = np.sin(theta)

            rot = np.array([[ct, -st, 0],
                            [st, ct, 0],
                            [0, 0, 1]])

            scale = np.array([[1 * s, 0, 0],
                              [0, 1 * s, 0],
                              [0, 0, 1]])

            from_grid2world = trans_inv.dot(scale.dot(rot.dot(trans)))
            to_grid2world = from_grid2world.dot(scale)
            to_world2grid = np.linalg.inv(to_grid2world)

            field, assignment = vfu.create_random_displacement_2d(
                np.array(from_shape, dtype=np.int32), from_grid2world,
                np.array(to_shape, dtype=np.int32), to_grid2world, rng=rng)
            field = np.array(field, dtype=floating)
            assignment = np.array(assignment)
            # Verify the assignments are inside the requested region
            assert_equal(0, (assignment < 0).sum())
            for i in range(2):
                assert_equal(0, (assignment[..., i] >= to_shape[i]).sum())

            # Compute the warping coordinates (see warp_2d documentation)
            Y = np.apply_along_axis(from_grid2world.dot, 0, X)[0:2, ...]
            Z = np.zeros_like(X)
            Z[0, ...] = Y[0, ...] + field[..., 0]
            Z[1, ...] = Y[1, ...] + field[..., 1]
            Z[2, ...] = 1
            W = np.apply_along_axis(to_world2grid.dot, 0, Z)[0:2, ...]

            # Verify the claimed assignments are correct
            assert_array_almost_equal(W[0, ...], assignment[..., 0], 5)
            assert_array_almost_equal(W[1, ...], assignment[..., 1], 5)

    # Test exception is raised when the affine transform matrix is not valid
    valid = np.zeros((2, 3), dtype=np.float64)
    invalid = np.zeros((2, 2), dtype=np.float64)
    shape = np.array(from_shape, dtype=np.int32)
    assert_raises(ValueError, vfu.create_random_displacement_2d,
                  shape, invalid, shape, valid, rng=rng)
    assert_raises(ValueError, vfu.create_random_displacement_2d,
                  shape, valid, shape, invalid, rng=rng)


@set_random_number_generator(7127562)
def test_random_displacement_field_3d(rng):
    from_shape = (25, 32, 31)
    to_shape = (33, 29, 35)

    # Create grid coordinates
    x_0 = np.asarray(range(from_shape[0]))
    x_1 = np.asarray(range(from_shape[1]))
    x_2 = np.asarray(range(from_shape[2]))
    X = np.empty((4,) + from_shape, dtype=np.float64)
    O = np.ones(from_shape)
    X[0, ...] = x_0[:, None, None] * O
    X[1, ...] = x_1[None, :, None] * O
    X[2, ...] = x_2[None, None, :] * O
    X[3, ...] = 1

    # Select an arbitrary rotation axis
    axis = np.array([.5, 2.0, 1.5])

    # Create an arbitrary image-to-space transform
    t = 0.15  # translation factor

    trans = np.array([[1, 0, 0, -t * from_shape[0]],
                      [0, 1, 0, -t * from_shape[1]],
                      [0, 0, 1, -t * from_shape[2]],
                      [0, 0, 0, 1]])
    trans_inv = np.linalg.inv(trans)

    for theta in [-1 * np.pi / 6.0, 0.0, np.pi / 5.0]:  # rotation angle
        for s in [0.83, 1.3, 2.07]:  # scale
            rot = np.zeros(shape=(4, 4))
            rot[:3, :3] = geometry.rodrigues_axis_rotation(axis, theta)
            rot[3, 3] = 1.0

            scale = np.array([[1 * s, 0, 0, 0],
                              [0, 1 * s, 0, 0],
                              [0, 0, 1 * s, 0],
                              [0, 0, 0, 1]])

            from_grid2world = trans_inv.dot(scale.dot(rot.dot(trans)))
            to_grid2world = from_grid2world.dot(scale)
            to_world2grid = np.linalg.inv(to_grid2world)

            field, assignment = vfu.create_random_displacement_3d(
                np.array(from_shape, dtype=np.int32), from_grid2world,
                np.array(to_shape, dtype=np.int32), to_grid2world, rng=rng)
            field = np.array(field, dtype=floating)
            assignment = np.array(assignment)
            # Verify the assignments are inside the requested region
            assert_equal(0, (assignment < 0).sum())
            for i in range(3):
                assert_equal(0, (assignment[..., i] >= to_shape[i]).sum())

            # Compute the warping coordinates (see warp_2d documentation)
            Y = np.apply_along_axis(from_grid2world.dot, 0, X)[0:3, ...]
            Z = np.zeros_like(X)
            Z[0, ...] = Y[0, ...] + field[..., 0]
            Z[1, ...] = Y[1, ...] + field[..., 1]
            Z[2, ...] = Y[2, ...] + field[..., 2]
            Z[3, ...] = 1
            W = np.apply_along_axis(to_world2grid.dot, 0, Z)[0:3, ...]

            # Verify the claimed assignments are correct
            assert_array_almost_equal(W[0, ...], assignment[..., 0], 5)
            assert_array_almost_equal(W[1, ...], assignment[..., 1], 5)
            assert_array_almost_equal(W[2, ...], assignment[..., 2], 5)

    # Test exception is raised when the affine transform matrix is not valid
    valid = np.zeros((3, 4), dtype=np.float64)
    invalid = np.zeros((3, 3), dtype=np.float64)
    shape = np.array(from_shape, dtype=np.int32)
    assert_raises(ValueError, vfu.create_random_displacement_2d,
                  shape, invalid, shape, valid, rng=rng)
    assert_raises(ValueError, vfu.create_random_displacement_2d,
                  shape, valid, shape, invalid, rng=rng)


def test_harmonic_fields_2d():
    nrows = 64
    ncols = 67
    mid_row = nrows // 2
    mid_col = ncols // 2
    expected_d = np.empty(shape=(nrows, ncols, 2))
    expected_d_inv = np.empty(shape=(nrows, ncols, 2))
    for b in [0.1, 0.3, 0.7]:
        for m in [2, 4, 7]:
            for i in range(nrows):
                for j in range(ncols):
                    ii = i - mid_row
                    jj = j - mid_col
                    theta = np.arctan2(ii, jj)
                    expected_d[i, j, 0] =\
                        ii * (1.0 / (1 + b * np.cos(m * theta)) - 1.0)
                    expected_d[i, j, 1] =\
                        jj * (1.0 / (1 + b * np.cos(m * theta)) - 1.0)
                    expected_d_inv[i, j, 0] = b * np.cos(m * theta) * ii
                    expected_d_inv[i, j, 1] = b * np.cos(m * theta) * jj

            actual_d, actual_d_inv =\
                vfu.create_harmonic_fields_2d(nrows, ncols, b, m)
            assert_array_almost_equal(expected_d, actual_d)
            assert_array_almost_equal(expected_d_inv, expected_d_inv)


def test_harmonic_fields_3d():
    nslices = 25
    nrows = 34
    ncols = 37
    mid_slice = nslices // 2
    mid_row = nrows // 2
    mid_col = ncols // 2
    expected_d = np.empty(shape=(nslices, nrows, ncols, 3))
    expected_d_inv = np.empty(shape=(nslices, nrows, ncols, 3))
    for b in [0.3, 0.7]:
        for m in [2, 5]:
            for k in range(nslices):
                for i in range(nrows):
                    for j in range(ncols):
                        kk = k - mid_slice
                        ii = i - mid_row
                        jj = j - mid_col
                        theta = np.arctan2(ii, jj)
                        expected_d[k, i, j, 0] =\
                            kk * (1.0 / (1 + b * np.cos(m * theta)) - 1.0)
                        expected_d[k, i, j, 1] =\
                            ii * (1.0 / (1 + b * np.cos(m * theta)) - 1.0)
                        expected_d[k, i, j, 2] =\
                            jj * (1.0 / (1 + b * np.cos(m * theta)) - 1.0)
                        expected_d_inv[k, i, j, 0] = b * np.cos(m * theta) * kk
                        expected_d_inv[k, i, j, 1] = b * np.cos(m * theta) * ii
                        expected_d_inv[k, i, j, 2] = b * np.cos(m * theta) * jj

            actual_d, actual_d_inv =\
                vfu.create_harmonic_fields_3d(nslices, nrows, ncols, b, m)
            assert_array_almost_equal(expected_d, actual_d)
            assert_array_almost_equal(expected_d_inv, expected_d_inv)


def test_circle():
    sh = (64, 61)
    cr = sh[0] // 2
    cc = sh[1] // 2
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    X = np.empty((2,) + sh, dtype=np.float64)
    O = np.ones(sh)
    X[0, ...] = x_0[:, None] * O - cr
    X[1, ...] = x_1[None, :] * O - cc
    nrm = np.sqrt(np.sum(X ** 2, axis=0))
    for radius in [0, 7, 17, 32]:
        expected = nrm <= radius
        actual = vfu.create_circle(sh[0], sh[1], radius)
        assert_array_almost_equal(actual, expected)


def test_sphere():
    sh = (64, 61, 57)
    cs = sh[0] // 2
    cr = sh[1] // 2
    cc = sh[2] // 2
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    x_2 = np.asarray(range(sh[2]))
    X = np.empty((3,) + sh, dtype=np.float64)
    O = np.ones(sh)
    X[0, ...] = x_0[:, None, None] * O - cs
    X[1, ...] = x_1[None, :, None] * O - cr
    X[2, ...] = x_2[None, None, :] * O - cc
    nrm = np.sqrt(np.sum(X ** 2, axis=0))
    for radius in [0, 7, 17, 32]:
        expected = nrm <= radius
        actual = vfu.create_sphere(sh[0], sh[1], sh[2], radius)
        assert_array_almost_equal(actual, expected)

def test_warping_2d():
    r"""
    Tests the cython implementation of the 2d warpings against scipy
    """
    sh = (64, 64)
    nr = sh[0]
    nc = sh[1]

    # Create an image of a circle
    radius = 24
    circle = vfu.create_circle(nr, nc, radius)
    circle = np.array(circle, dtype=floating)

    # Create a displacement field for warping
    d, dinv = vfu.create_harmonic_fields_2d(nr, nc, 0.2, 8)
    d = np.asarray(d).astype(floating)

    # Create grid coordinates
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    X = np.empty((3,) + sh, dtype=np.float64)
    O = np.ones(sh)
    X[0, ...] = x_0[:, None] * O
    X[1, ...] = x_1[None, :] * O
    X[2, ...] = 1

    # Select an arbitrary translation matrix
    t = 0.1
    trans = np.array([[1, 0, -t * nr],
                      [0, 1, -t * nc],
                      [0, 0, 1]])
    trans_inv = np.linalg.inv(trans)

    # Select arbitrary rotation and scaling matrices
    for theta in [-1 * np.pi / 6.0, 0.0, np.pi / 6.0]:  # rotation angle
        for s in [0.42, 1.3, 2.15]:  # scale
            ct = np.cos(theta)
            st = np.sin(theta)

            rot = np.array([[ct, -st, 0],
                            [st, ct, 0],
                            [0, 0, 1]])

            scale = np.array([[1 * s, 0, 0],
                              [0, 1 * s, 0],
                              [0, 0, 1]])

            aff = trans_inv.dot(scale.dot(rot.dot(trans)))

            # Select arbitrary (but different) grid-to-space transforms
            sampling_grid2world = scale
            field_grid2world = aff
            field_world2grid = np.linalg.inv(field_grid2world)
            image_grid2world = aff.dot(scale)
            image_world2grid = np.linalg.inv(image_grid2world)

            A = field_world2grid.dot(sampling_grid2world)
            B = image_world2grid.dot(sampling_grid2world)
            C = image_world2grid

            # Reorient the displacement field according to its grid-to-space
            # transform
            dcopy = np.copy(d)
            vfu.reorient_vector_field_2d(dcopy, field_grid2world)
            extended_dcopy = np.zeros((nr + 2, nc + 2, 2), dtype=floating)
            extended_dcopy[1:nr + 1, 1:nc + 1, :] = dcopy

            # Compute the warping coordinates (see warp_2d documentation)
            Y = np.apply_along_axis(A.dot, 0, X)[0:2, ...]
            Z = np.zeros_like(X)
            Z[0, ...] = map_coordinates(extended_dcopy[..., 0], Y + 1, order=1)
            Z[1, ...] = map_coordinates(extended_dcopy[..., 1], Y + 1, order=1)
            Z[2, ...] = 0
            Z = np.apply_along_axis(C.dot, 0, Z)[0:2, ...]
            T = np.apply_along_axis(B.dot, 0, X)[0:2, ...]
            W = T + Z

            # Test bilinear interpolation
            expected = map_coordinates(circle, W, order=1)
            warped = vfu.warp_2d(circle, dcopy, A, B, C,
                                 np.array(sh, dtype=np.int32))
            assert_array_almost_equal(warped, expected)

            # Test nearest neighbor interpolation
            expected = map_coordinates(circle, W, order=0)
            warped = vfu.warp_2d_nn(circle, dcopy, A, B, C,
                                    np.array(sh, dtype=np.int32))
            assert_array_almost_equal(warped, expected)

    # Test exception is raised when the affine transform matrix is not valid
    val = np.zeros((2, 3), dtype=np.float64)
    inval = np.zeros((2, 2), dtype=np.float64)
    sh = np.array(sh, dtype=np.int32)
    # Exceptions from warp_2d
    assert_raises(ValueError, vfu.warp_2d, circle, d, inval, val, val, sh)
    assert_raises(ValueError, vfu.warp_2d, circle, d, val, inval, val, sh)
    assert_raises(ValueError, vfu.warp_2d, circle, d, val, val, inval, sh)
    # Exceptions from warp_2d_nn
    assert_raises(ValueError, vfu.warp_2d_nn, circle, d, inval, val, val, sh)
    assert_raises(ValueError, vfu.warp_2d_nn, circle, d, val, inval, val, sh)
    assert_raises(ValueError, vfu.warp_2d_nn, circle, d, val, val, inval, sh)


def test_warping_3d():
    r"""
    Tests the cython implementation of the 2d warpings against scipy
    """
    sh = (64, 64, 64)
    ns = sh[0]
    nr = sh[1]
    nc = sh[2]

    # Create an image of a sphere
    radius = 24
    sphere = vfu.create_sphere(ns, nr, nc, radius)
    sphere = np.array(sphere, dtype=floating)

    # Create a displacement field for warping
    d, dinv = vfu.create_harmonic_fields_3d(ns, nr, nc, 0.2, 8)
    d = np.asarray(d).astype(floating)

    # Create grid coordinates
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    x_2 = np.asarray(range(sh[2]))
    X = np.empty((4,) + sh, dtype=np.float64)
    O = np.ones(sh)
    X[0, ...] = x_0[:, None, None] * O
    X[1, ...] = x_1[None, :, None] * O
    X[2, ...] = x_2[None, None, :] * O
    X[3, ...] = 1

    # Select an arbitrary rotation axis
    axis = np.array([.5, 2.0, 1.5])
    # Select an arbitrary translation matrix
    t = 0.1
    trans = np.array([[1, 0, 0, -t * ns],
                      [0, 1, 0, -t * nr],
                      [0, 0, 1, -t * nc],
                      [0, 0, 0, 1]])
    trans_inv = np.linalg.inv(trans)

    # Select arbitrary rotation and scaling matrices
    for theta in [-1 * np.pi / 5.0, 0.0, np.pi / 5.0]:  # rotation angle
        for s in [0.45, 1.1, 2.0]:  # scale
            rot = np.zeros(shape=(4, 4))
            rot[:3, :3] = geometry.rodrigues_axis_rotation(axis, theta)
            rot[3, 3] = 1.0

            scale = np.array([[1 * s, 0, 0, 0],
                              [0, 1 * s, 0, 0],
                              [0, 0, 1 * s, 0],
                              [0, 0, 0, 1]])

            aff = trans_inv.dot(scale.dot(rot.dot(trans)))

            # Select arbitrary (but different) grid-to-space transforms
            sampling_grid2world = scale
            field_grid2world = aff
            field_world2grid = np.linalg.inv(field_grid2world)
            image_grid2world = aff.dot(scale)
            image_world2grid = np.linalg.inv(image_grid2world)

            A = field_world2grid.dot(sampling_grid2world)
            B = image_world2grid.dot(sampling_grid2world)
            C = image_world2grid

            # Reorient the displacement field according to its grid-to-space
            # transform
            dcopy = np.copy(d)
            vfu.reorient_vector_field_3d(dcopy, field_grid2world)

            extended_dcopy = np.zeros(
                (ns + 2, nr + 2, nc + 2, 3), dtype=floating)
            extended_dcopy[1:ns + 1, 1:nr + 1, 1:nc + 1, :] = dcopy

            # Compute the warping coordinates (see warp_2d documentation)
            Y = np.apply_along_axis(A.dot, 0, X)[0:3, ...]
            Z = np.zeros_like(X)
            Z[0, ...] = map_coordinates(extended_dcopy[..., 0], Y + 1, order=1)
            Z[1, ...] = map_coordinates(extended_dcopy[..., 1], Y + 1, order=1)
            Z[2, ...] = map_coordinates(extended_dcopy[..., 2], Y + 1, order=1)
            Z[3, ...] = 0
            Z = np.apply_along_axis(C.dot, 0, Z)[0:3, ...]
            T = np.apply_along_axis(B.dot, 0, X)[0:3, ...]
            W = T + Z

            # Test bilinear interpolation
            expected = map_coordinates(sphere, W, order=1)
            warped = vfu.warp_3d(sphere, dcopy, A, B, C,
                                 np.array(sh, dtype=np.int32))
            assert_array_almost_equal(warped, expected, decimal=5)

            # Test nearest neighbor interpolation
            expected = map_coordinates(sphere, W, order=0)
            warped = vfu.warp_3d_nn(sphere, dcopy, A, B, C,
                                    np.array(sh, dtype=np.int32))
            assert_array_almost_equal(warped, expected, decimal=5)

    # Test exception is raised when the affine transform matrix is not valid
    val = np.zeros((3, 4), dtype=np.float64)
    inval = np.zeros((3, 3), dtype=np.float64)
    sh = np.array(sh, dtype=np.int32)
    # Exceptions from warp_3d
    assert_raises(ValueError, vfu.warp_3d, sphere, d, inval, val, val, sh)
    assert_raises(ValueError, vfu.warp_3d, sphere, d, val, inval, val, sh)
    assert_raises(ValueError, vfu.warp_3d, sphere, d, val, val, inval, sh)
    # Exceptions from warp_3d_nn
    assert_raises(ValueError, vfu.warp_3d_nn, sphere, d, inval, val, val, sh)
    assert_raises(ValueError, vfu.warp_3d_nn, sphere, d, val, inval, val, sh)
    assert_raises(ValueError, vfu.warp_3d_nn, sphere, d, val, val, inval, sh)


def test_affine_transforms_2d():
    r"""
    Tests 2D affine transform functions against scipy implementation
    """
    # Create a simple invertible affine transform
    d_shape = (64, 64)
    codomain_shape = (80, 80)
    nr = d_shape[0]
    nc = d_shape[1]

    # Create an image of a circle
    radius = 16
    circle = vfu.create_circle(codomain_shape[0], codomain_shape[1], radius)
    circle = np.array(circle, dtype=floating)

    # Create grid coordinates
    x_0 = np.asarray(range(d_shape[0]))
    x_1 = np.asarray(range(d_shape[1]))
    X = np.empty((3,) + d_shape, dtype=np.float64)
    O = np.ones(d_shape)
    X[0, ...] = x_0[:, None] * O
    X[1, ...] = x_1[None, :] * O
    X[2, ...] = 1

    # Generate affine transforms
    t = 0.3
    trans = np.array([[1, 0, -t * nr],
                      [0, 1, -t * nc],
                      [0, 0, 1]])
    trans_inv = np.linalg.inv(trans)
    for theta in [-1 * np.pi / 5.0, 0.0, np.pi / 5.0]:  # rotation angle
        for s in [0.5, 1.0, 2.0]:  # scale
            ct = np.cos(theta)
            st = np.sin(theta)

            rot = np.array([[ct, -st, 0],
                            [st, ct, 0],
                            [0, 0, 1]])

            scale = np.array([[1 * s, 0, 0],
                              [0, 1 * s, 0],
                              [0, 0, 1]])

            gt_affine = trans_inv.dot(scale.dot(rot.dot(trans)))

            # Apply the affine transform to the grid coordinates
            Y = np.apply_along_axis(gt_affine.dot, 0, X)[0:2, ...]

            expected = map_coordinates(circle, Y, order=1)
            warped = vfu.transform_2d_affine(
                circle, np.array(
                    d_shape, dtype=np.int32), gt_affine)
            assert_array_almost_equal(warped, expected)

            # Test affine warping with nearest-neighbor interpolation
            expected = map_coordinates(circle, Y, order=0)
            warped = vfu.transform_2d_affine_nn(
                circle, np.array(d_shape, dtype=np.int32), gt_affine)
            assert_array_almost_equal(warped, expected)

    # Test the affine = None case
    warped = vfu.transform_2d_affine(
        circle, np.array(
            codomain_shape, dtype=np.int32), None)
    assert_array_equal(warped, circle)

    warped = vfu.transform_2d_affine_nn(
        circle, np.array(
            codomain_shape, dtype=np.int32), None)
    assert_array_equal(warped, circle)

    # Test exception is raised when the affine transform matrix is not valid
    invalid = np.zeros((2, 2), dtype=np.float64)
    invalid_nan = np.zeros((3, 3), dtype=np.float64)
    invalid_nan[1, 1] = np.nan
    shape = np.array(codomain_shape, dtype=np.int32)
    # Exceptions from transform_2d
    assert_raises(ValueError, vfu.transform_2d_affine, circle, shape, invalid)
    assert_raises(
        ValueError,
        vfu.transform_2d_affine,
        circle,
        shape,
        invalid_nan)
    # Exceptions from transform_2d_nn
    assert_raises(
        ValueError,
        vfu.transform_2d_affine_nn,
        circle,
        shape,
        invalid)
    assert_raises(
        ValueError,
        vfu.transform_2d_affine_nn,
        circle,
        shape,
        invalid_nan)


def test_affine_transforms_3d():
    r"""
    Tests 3D affine transform functions against scipy implementation
    """
    # Create a simple invertible affine transform
    d_shape = (64, 64, 64)
    codomain_shape = (80, 80, 80)
    ns = d_shape[0]
    nr = d_shape[1]
    nc = d_shape[2]

    # Create an image of a sphere
    radius = 16
    sphere = vfu.create_sphere(codomain_shape[0], codomain_shape[1],
                               codomain_shape[2], radius)
    sphere = np.array(sphere, dtype=floating)

    # Create grid coordinates
    x_0 = np.asarray(range(d_shape[0]))
    x_1 = np.asarray(range(d_shape[1]))
    x_2 = np.asarray(range(d_shape[2]))
    X = np.empty((4,) + d_shape, dtype=np.float64)
    O = np.ones(d_shape)
    X[0, ...] = x_0[:, None, None] * O
    X[1, ...] = x_1[None, :, None] * O
    X[2, ...] = x_2[None, None, :] * O
    X[3, ...] = 1

    # Generate affine transforms
    # Select an arbitrary rotation axis
    axis = np.array([.5, 2.0, 1.5])
    t = 0.3
    trans = np.array([[1, 0, 0, -t * ns],
                      [0, 1, 0, -t * nr],
                      [0, 0, 1, -t * nc],
                      [0, 0, 0, 1]])
    trans_inv = np.linalg.inv(trans)
    for theta in [-1 * np.pi / 5.0, 0.0, np.pi / 5.0]:  # rotation angle
        for s in [0.45, 1.1, 2.3]:  # scale
            rot = np.zeros(shape=(4, 4))
            rot[:3, :3] = geometry.rodrigues_axis_rotation(axis, theta)
            rot[3, 3] = 1.0

            scale = np.array([[1 * s, 0, 0, 0],
                              [0, 1 * s, 0, 0],
                              [0, 0, 1 * s, 0],
                              [0, 0, 0, 1]])

            gt_affine = trans_inv.dot(scale.dot(rot.dot(trans)))

            # Apply the affine transform to the grid coordinates
            Y = np.apply_along_axis(gt_affine.dot, 0, X)[0:3, ...]

            expected = map_coordinates(sphere, Y, order=1)
            transformed = vfu.transform_3d_affine(
                sphere, np.array(d_shape, dtype=np.int32), gt_affine)
            assert_array_almost_equal(transformed, expected)

            # Test affine transform with nearest-neighbor interpolation
            expected = map_coordinates(sphere, Y, order=0)
            transformed = vfu.transform_3d_affine_nn(
                sphere, np.array(d_shape, dtype=np.int32), gt_affine)
            assert_array_almost_equal(transformed, expected)

    # Test the affine = None case
    transformed = vfu.transform_3d_affine(
        sphere, np.array(codomain_shape, dtype=np.int32), None)
    assert_array_equal(transformed, sphere)

    transformed = vfu.transform_3d_affine_nn(
        sphere, np.array(codomain_shape, dtype=np.int32), None)
    assert_array_equal(transformed, sphere)

    # Test exception is raised when the affine transform matrix is not valid
    invalid = np.zeros((3, 3), dtype=np.float64)
    invalid_nan = np.zeros((4, 4), dtype=np.float64)
    invalid_nan[1, 1] = np.nan
    shape = np.array(codomain_shape, dtype=np.int32)
    # Exceptions from transform_3d_affine
    assert_raises(ValueError, vfu.transform_3d_affine, sphere, shape, invalid)
    assert_raises(
        ValueError,
        vfu.transform_3d_affine,
        sphere,
        shape,
        invalid_nan)
    # Exceptions from transform_3d_affine_nn
    assert_raises(
        ValueError,
        vfu.transform_3d_affine_nn,
        sphere,
        shape,
        invalid)
    assert_raises(
        ValueError,
        vfu.transform_3d_affine_nn,
        sphere,
        shape,
        invalid_nan)


@set_random_number_generator(8315759)
def test_compose_vector_fields_2d(rng):
    r"""
    Creates two random displacement field that exactly map pixels from an input
    image to an output image. The resulting displacements and their
    composition, although operating in physical space, map the points exactly
    (up to numerical precision).
    """
    input_shape = (10, 10)
    tgt_sh = (10, 10)
    # create a simple affine transformation
    nr = input_shape[0]
    nc = input_shape[1]
    s = 1.5
    t = 2.5
    trans = np.array([[1, 0, -t * nr],
                      [0, 1, -t * nc],
                      [0, 0, 1]])
    trans_inv = np.linalg.inv(trans)
    scale = np.array([[1 * s, 0, 0],
                      [0, 1 * s, 0],
                      [0, 0, 1]])
    gt_affine = trans_inv.dot(scale.dot(trans))

    # create two random displacement fields
    input_grid2world = gt_affine
    target_grid2world = gt_affine

    disp1, assign1 = vfu.create_random_displacement_2d(
        np.array(input_shape, dtype=np.int32),
        input_grid2world,
        np.array(tgt_sh, dtype=np.int32),
        target_grid2world, rng=rng)
    disp1 = np.array(disp1, dtype=floating)
    assign1 = np.array(assign1)

    disp2, assign2 = vfu.create_random_displacement_2d(
        np.array(input_shape, dtype=np.int32),
        input_grid2world,
        np.array(tgt_sh, dtype=np.int32),
        target_grid2world, rng=rng)
    disp2 = np.array(disp2, dtype=floating)
    assign2 = np.array(assign2)

    # create a random image (with decimal digits) to warp
    moving_image = np.empty(tgt_sh, dtype=floating)
    moving_image[...] =\
        rng.integers(0, 10, np.size(moving_image)).reshape(tuple(tgt_sh))
    # set boundary values to zero so we don't test wrong interpolation due to
    # floating point precision
    moving_image[0, :] = 0
    moving_image[-1, :] = 0
    moving_image[:, 0] = 0
    moving_image[:, -1] = 0

    # evaluate the composed warping using the exact assignments
    # (first 1 then 2)
    warp1 = moving_image[(assign2[..., 0], assign2[..., 1])]
    expected = warp1[(assign1[..., 0], assign1[..., 1])]

    # compose the displacement fields
    target_world2grid = np.linalg.inv(target_grid2world)
    premult_index = target_world2grid.dot(input_grid2world)
    premult_disp = target_world2grid

    for time_scaling in [0.25, 0.5, 1.0, 2.0, 4.0]:
        composition, stats = vfu.compose_vector_fields_2d(disp1,
                                                          disp2 / time_scaling,
                                                          premult_index,
                                                          premult_disp,
                                                          time_scaling, None)
        # apply the implementation under test
        warped = np.array(vfu.warp_2d(moving_image, composition, None,
                                      premult_index, premult_disp))
        assert_array_almost_equal(warped, expected)

        # test also using nearest neighbor interpolation
        warped = np.array(vfu.warp_2d_nn(moving_image, composition, None,
                                         premult_index, premult_disp))
        assert_array_almost_equal(warped, expected)

        # test updating the displacement field instead of creating a new one
        composition = disp1.copy()
        vfu.compose_vector_fields_2d(composition, disp2 / time_scaling,
                                     premult_index, premult_disp, time_scaling,
                                     composition)
        # apply the implementation under test
        warped = np.array(vfu.warp_2d(moving_image, composition, None,
                                      premult_index, premult_disp))
        assert_array_almost_equal(warped, expected)

        # test also using nearest neighbor interpolation
        warped = np.array(vfu.warp_2d_nn(moving_image, composition, None,
                                         premult_index, premult_disp))
        assert_array_almost_equal(warped, expected)

    # Test non-overlapping case
    x_0 = np.asarray(range(input_shape[0]))
    x_1 = np.asarray(range(input_shape[1]))
    X = np.empty(input_shape + (2,), dtype=np.float64)
    O = np.ones(input_shape)
    X[..., 0] = x_0[:, None] * O
    X[..., 1] = x_1[None, :] * O
    random_labels = rng.integers(
        0, 2, input_shape[0] * input_shape[1] * 2)
    random_labels = random_labels.reshape(input_shape + (2,))
    values = np.array([-1, tgt_sh[0]])
    disp1 = (values[random_labels] - X).astype(floating)
    composition, stats = vfu.compose_vector_fields_2d(disp1,
                                                      disp2,
                                                      None,
                                                      None,
                                                      1.0, None)
    assert_array_almost_equal(composition, np.zeros_like(composition))

    # test updating the displacement field instead of creating a new one
    composition = disp1.copy()
    vfu.compose_vector_fields_2d(composition, disp2, None, None, 1.0,
                                 composition)
    assert_array_almost_equal(composition, np.zeros_like(composition))

    # Test exception is raised when the affine transform matrix is not valid
    valid = np.zeros((2, 3), dtype=np.float64)
    invalid = np.zeros((2, 2), dtype=np.float64)
    assert_raises(ValueError, vfu.compose_vector_fields_2d, disp1, disp2,
                  invalid, valid, 1.0, None)
    assert_raises(ValueError, vfu.compose_vector_fields_2d, disp1, disp2,
                  valid, invalid, 1.0, None)


@set_random_number_generator(8315759)
def test_compose_vector_fields_3d(rng):
    r"""
    Creates two random displacement field that exactly map pixels from an input
    image to an output image. The resulting displacements and their
    composition, although operating in physical space, map the points exactly
    (up to numerical precision).
    """
    input_shape = (10, 10, 10)
    tgt_sh = (10, 10, 10)
    # create a simple affine transformation
    ns = input_shape[0]
    nr = input_shape[1]
    nc = input_shape[2]
    s = 1.5
    t = 2.5
    trans = np.array([[1, 0, 0, -t * ns],
                      [0, 1, 0, -t * nr],
                      [0, 0, 1, -t * nc],
                      [0, 0, 0, 1]])
    trans_inv = np.linalg.inv(trans)
    scale = np.array([[1 * s, 0, 0, 0],
                      [0, 1 * s, 0, 0],
                      [0, 0, 1 * s, 0],
                      [0, 0, 0, 1]])
    gt_affine = trans_inv.dot(scale.dot(trans))

    # create two random displacement fields
    input_grid2world = gt_affine
    target_grid2world = gt_affine

    disp1, assign1 = vfu.create_random_displacement_3d(
        np.array(input_shape, dtype=np.int32),
        input_grid2world, np.array(tgt_sh, dtype=np.int32),
        target_grid2world, rng=rng)
    disp1 = np.array(disp1, dtype=floating)
    assign1 = np.array(assign1)

    disp2, assign2 = vfu.create_random_displacement_3d(
        np.array(
            input_shape, dtype=np.int32), input_grid2world, np.array(
            tgt_sh, dtype=np.int32), target_grid2world, rng=rng)
    disp2 = np.array(disp2, dtype=floating)
    assign2 = np.array(assign2)

    # create a random image (with decimal digits) to warp
    moving_image = np.empty(tgt_sh, dtype=floating)
    moving_image[...] =\
        rng.integers(0, 10, np.size(moving_image)).reshape(tuple(tgt_sh))
    # set boundary values to zero so we don't test wrong interpolation due to
    # floating point precision
    moving_image[0, :, :] = 0
    moving_image[-1, :, :] = 0
    moving_image[:, 0, :] = 0
    moving_image[:, -1, :] = 0
    moving_image[:, :, 0] = 0
    moving_image[:, :, -1] = 0

    # evaluate the composed warping using the exact assignments
    # (first 1 then 2)

    warp1 = moving_image[(assign2[..., 0], assign2[..., 1], assign2[..., 2])]
    expected = warp1[(assign1[..., 0], assign1[..., 1], assign1[..., 2])]

    # compose the displacement fields
    target_world2grid = np.linalg.inv(target_grid2world)
    premult_index = target_world2grid.dot(input_grid2world)
    premult_disp = target_world2grid

    for time_scaling in [0.25, 0.5, 1.0, 2.0, 4.0]:
        composition, stats = vfu.compose_vector_fields_3d(disp1,
                                                          disp2 / time_scaling,
                                                          premult_index,
                                                          premult_disp,
                                                          time_scaling, None)
        # apply the implementation under test
        warped = np.array(vfu.warp_3d(moving_image, composition, None,
                                      premult_index, premult_disp))
        assert_array_almost_equal(warped, expected)

        # test also using nearest neighbor interpolation
        warped = np.array(vfu.warp_3d_nn(moving_image, composition, None,
                                         premult_index, premult_disp))
        assert_array_almost_equal(warped, expected)

        # test updating the displacement field instead of creating a new one
        composition = disp1.copy()
        vfu.compose_vector_fields_3d(composition, disp2 / time_scaling,
                                     premult_index, premult_disp,
                                     time_scaling, composition)
        # apply the implementation under test
        warped = np.array(vfu.warp_3d(moving_image, composition, None,
                                      premult_index, premult_disp))
        assert_array_almost_equal(warped, expected)

        # test also using nearest neighbor interpolation
        warped = np.array(vfu.warp_3d_nn(moving_image, composition, None,
                                         premult_index, premult_disp))
        assert_array_almost_equal(warped, expected)

    # Test non-overlapping case
    x_0 = np.asarray(range(input_shape[0]))
    x_1 = np.asarray(range(input_shape[1]))
    x_2 = np.asarray(range(input_shape[2]))
    X = np.empty(input_shape + (3,), dtype=np.float64)
    O = np.ones(input_shape)
    X[..., 0] = x_0[:, None, None] * O
    X[..., 1] = x_1[None, :, None] * O
    X[..., 2] = x_2[None, None, :] * O
    sz = input_shape[0] * input_shape[1] * input_shape[2] * 3
    random_labels = rng.integers(0, 2, sz)
    random_labels = random_labels.reshape(input_shape + (3,))
    values = np.array([-1, tgt_sh[0]])
    disp1 = (values[random_labels] - X).astype(floating)
    composition, stats = vfu.compose_vector_fields_3d(disp1,
                                                      disp2,
                                                      None,
                                                      None,
                                                      1.0, None)
    assert_array_almost_equal(composition, np.zeros_like(composition))

    # test updating the displacement field instead of creating a new one
    composition = disp1.copy()
    vfu.compose_vector_fields_3d(composition, disp2, None, None, 1.0,
                                 composition)
    assert_array_almost_equal(composition, np.zeros_like(composition))

    # Test exception is raised when the affine transform matrix is not valid
    valid = np.zeros((3, 4), dtype=np.float64)
    invalid = np.zeros((3, 3), dtype=np.float64)
    assert_raises(ValueError, vfu.compose_vector_fields_3d, disp1, disp2,
                  invalid, valid, 1.0, None)
    assert_raises(ValueError, vfu.compose_vector_fields_3d, disp1, disp2,
                  valid, invalid, 1.0, None)


def test_invert_vector_field_2d():
    r"""
    Inverts a synthetic, analytically invertible, displacement field
    """
    shape = (64, 64)
    nr = shape[0]
    nc = shape[1]
    # Create an arbitrary image-to-space transform
    t = 2.5  # translation factor

    trans = np.array([[1, 0, -t * nr],
                      [0, 1, -t * nc],
                      [0, 0, 1]])
    trans_inv = np.linalg.inv(trans)

    d, _ = vfu.create_harmonic_fields_2d(nr, nc, 0.2, 8)
    d = np.asarray(d).astype(floating)

    for theta in [-1 * np.pi / 5.0, 0.0, np.pi / 5.0]:  # rotation angle
        for s in [0.5, 1.0, 2.0]:  # scale
            ct = np.cos(theta)
            st = np.sin(theta)

            rot = np.array([[ct, -st, 0],
                            [st, ct, 0],
                            [0, 0, 1]])

            scale = np.array([[1 * s, 0, 0],
                              [0, 1 * s, 0],
                              [0, 0, 1]])

            gt_affine = trans_inv.dot(scale.dot(rot.dot(trans)))
            gt_affine_inv = np.linalg.inv(gt_affine)
            dcopy = np.copy(d)

            # make sure the field remains invertible after the re-mapping
            vfu.reorient_vector_field_2d(dcopy, gt_affine)

            inv_approx =\
                vfu.invert_vector_field_fixed_point_2d(dcopy, gt_affine_inv,
                                                       np.array([s, s]),
                                                       40, 1e-7)

            mapping = imwarp.DiffeomorphicMap(2, (nr, nc), gt_affine)
            mapping.forward = dcopy
            mapping.backward = inv_approx
            residual, stats = mapping.compute_inversion_error()
            assert_almost_equal(stats[1], 0, decimal=4)
            assert_almost_equal(stats[2], 0, decimal=4)

    # Test exception is raised when the affine transform matrix is not valid
    invalid = np.zeros((2, 2), dtype=np.float64)
    spacing = np.array([1.0, 1.0])
    assert_raises(ValueError, vfu.invert_vector_field_fixed_point_2d,
                  d, invalid, spacing, 40, 1e-7, None)


def test_invert_vector_field_3d():
    r"""
    Inverts a synthetic, analytically invertible, displacement field
    """
    shape = (64, 64, 64)
    ns = shape[0]
    nr = shape[1]
    nc = shape[2]

    # Create an arbitrary image-to-space transform

    # Select an arbitrary rotation axis
    axis = np.array([2.0, 0.5, 1.0])
    t = 2.5  # translation factor

    trans = np.array([[1, 0, 0, -t * ns],
                      [0, 1, 0, -t * nr],
                      [0, 0, 1, -t * nc],
                      [0, 0, 0, 1]])
    trans_inv = np.linalg.inv(trans)

    d, _ = vfu.create_harmonic_fields_3d(ns, nr, nc, 0.2, 8)
    d = np.asarray(d).astype(floating)

    for theta in [-1 * np.pi / 5.0, 0.0, np.pi / 5.0]:  # rotation angle
        for s in [0.5, 1.0, 2.0]:  # scale
            rot = np.zeros(shape=(4, 4))
            rot[:3, :3] = geometry.rodrigues_axis_rotation(axis, theta)
            rot[3, 3] = 1.0
            scale = np.array([[1 * s, 0, 0, 0],
                              [0, 1 * s, 0, 0],
                              [0, 0, 1 * s, 0],
                              [0, 0, 0, 1]])

            gt_affine = trans_inv.dot(scale.dot(rot.dot(trans)))
            gt_affine_inv = np.linalg.inv(gt_affine)
            dcopy = np.copy(d)

            # make sure the field remains invertible after the re-mapping
            vfu.reorient_vector_field_3d(dcopy, gt_affine)

            # Note: the spacings are used just to check convergence, so they
            # don't need to be very accurate. Here we are passing (0.5 * s) to
            # force the algorithm to make more iterations: in ANTS, there is a
            # hard-coded bound on the maximum residual, that's why we cannot
            # force more iteration by changing the parameters.
            # We will investigate this issue with more detail in the future.

            inv_approx = vfu.invert_vector_field_fixed_point_3d(
                dcopy, gt_affine_inv, np.array([s, s, s]) * 0.5, 40, 1e-7)

            mapping = imwarp.DiffeomorphicMap(3, (nr, nc), gt_affine)
            mapping.forward = dcopy
            mapping.backward = inv_approx
            residual, stats = mapping.compute_inversion_error()
            assert_almost_equal(stats[1], 0, decimal=3)
            assert_almost_equal(stats[2], 0, decimal=3)

    # Test exception is raised when the affine transform matrix is not valid
    invalid = np.zeros((3, 3), dtype=np.float64)
    spacing = np.array([1.0, 1.0, 1.0])
    assert_raises(ValueError, vfu.invert_vector_field_fixed_point_3d,
                  d, invalid, spacing, 40, 1e-7, None)


def test_resample_vector_field_2d():
    r"""
    Expand a vector field by 2, then subsample by 2, the resulting
    field should be the original one
    """
    domain_shape = np.array((64, 64), dtype=np.int32)
    reduced_shape = np.array((32, 32), dtype=np.int32)
    factors = np.array([0.5, 0.5])
    d, dinv = vfu.create_harmonic_fields_2d(reduced_shape[0], reduced_shape[1],
                                            0.3, 6)
    d = np.array(d, dtype=floating)

    expanded = vfu.resample_displacement_field_2d(d, factors, domain_shape)
    subsampled = expanded[::2, ::2, :]

    assert_array_almost_equal(d, subsampled)


def test_resample_vector_field_3d():
    r"""
    Expand a vector field by 2, then subsample by 2, the resulting
    field should be the original one
    """
    domain_shape = np.array((64, 64, 64), dtype=np.int32)
    reduced_shape = np.array((32, 32, 32), dtype=np.int32)
    factors = np.array([0.5, 0.5, 0.5])
    d, dinv = vfu.create_harmonic_fields_3d(reduced_shape[0], reduced_shape[1],
                                            reduced_shape[2], 0.3, 6)
    d = np.array(d, dtype=floating)

    expanded = vfu.resample_displacement_field_3d(d, factors, domain_shape)
    subsampled = expanded[::2, ::2, ::2, :]

    assert_array_almost_equal(d, subsampled)


@set_random_number_generator(8315759)
def test_downsample_scalar_field_2d(rng):
    size = 32
    sh = (size, size)
    for reduce_r in [True, False]:
        nr = size - 1 if reduce_r else size
        for reduce_c in [True, False]:
            nc = size - 1 if reduce_c else size
            image = np.empty((size, size), dtype=floating)
            image[...] = rng.integers(0, 10, np.size(image)).reshape(sh)

            if reduce_r:
                image[-1, :] = 0
            if reduce_c:
                image[:, -1] = 0

            a = image[::2, ::2]
            b = image[1::2, ::2]
            c = image[::2, 1::2]
            d = image[1::2, 1::2]

            expected = 0.25 * (a + b + c + d)

            if reduce_r:
                expected[-1, :] *= 2
            if reduce_c:
                expected[:, -1] *= 2

            actual = np.array(vfu.downsample_scalar_field_2d(image[:nr, :nc]))
            assert_array_almost_equal(expected, actual)


@set_random_number_generator(2115556)
def test_downsample_displacement_field_2d(rng):
    size = 32
    sh = (size, size, 2)
    for reduce_r in [True, False]:
        nr = size - 1 if reduce_r else size
        for reduce_c in [True, False]:
            nc = size - 1 if reduce_c else size
            field = np.empty((size, size, 2), dtype=floating)
            field[...] = rng.integers(0, 10, np.size(field)).reshape(sh)

            if reduce_r:
                field[-1, :, :] = 0
            if reduce_c:
                field[:, -1, :] = 0

            a = field[::2, ::2, :]
            b = field[1::2, ::2, :]
            c = field[::2, 1::2, :]
            d = field[1::2, 1::2, :]

            expected = 0.25 * (a + b + c + d)

            if reduce_r:
                expected[-1, :, :] *= 2
            if reduce_c:
                expected[:, -1, :] *= 2

            actual = vfu.downsample_displacement_field_2d(field[:nr, :nc, :])
            assert_array_almost_equal(expected, actual)


@set_random_number_generator(8315759)
def test_downsample_scalar_field_3d(rng):
    size = 32
    sh = (size, size, size)
    for reduce_s in [True, False]:
        ns = size - 1 if reduce_s else size
        for reduce_r in [True, False]:
            nr = size - 1 if reduce_r else size
            for reduce_c in [True, False]:
                nc = size - 1 if reduce_c else size
                image = np.empty((size, size, size), dtype=floating)
                image[...] =\
                    rng.integers(0, 10, np.size(image)).reshape(sh)

                if reduce_s:
                    image[-1, :, :] = 0
                if reduce_r:
                    image[:, -1, :] = 0
                if reduce_c:
                    image[:, :, -1] = 0

                a = image[::2, ::2, ::2]
                b = image[1::2, ::2, ::2]
                c = image[::2, 1::2, ::2]
                d = image[1::2, 1::2, ::2]
                aa = image[::2, ::2, 1::2]
                bb = image[1::2, ::2, 1::2]
                cc = image[::2, 1::2, 1::2]
                dd = image[1::2, 1::2, 1::2]

                expected = 0.125 * (a + b + c + d + aa + bb + cc + dd)

                if reduce_s:
                    expected[-1, :, :] *= 2
                if reduce_r:
                    expected[:, -1, :] *= 2
                if reduce_c:
                    expected[:, :, -1] *= 2

                actual = vfu.downsample_scalar_field_3d(image[:ns, :nr, :nc])
                assert_array_almost_equal(expected, actual)


@set_random_number_generator(8315759)
def test_downsample_displacement_field_3d(rng):
    size = 32
    sh = (size, size, size, 3)
    for reduce_s in [True, False]:
        ns = size - 1 if reduce_s else size
        for reduce_r in [True, False]:
            nr = size - 1 if reduce_r else size
            for reduce_c in [True, False]:
                nc = size - 1 if reduce_c else size
                field = np.empty((size, size, size, 3), dtype=floating)
                field[...] =\
                    rng.integers(0, 10, np.size(field)).reshape(sh)

                if reduce_s:
                    field[-1, :, :] = 0
                if reduce_r:
                    field[:, -1, :] = 0
                if reduce_c:
                    field[:, :, -1] = 0

                a = field[::2, ::2, ::2, :]
                b = field[1::2, ::2, ::2, :]
                c = field[::2, 1::2, ::2, :]
                d = field[1::2, 1::2, ::2, :]
                aa = field[::2, ::2, 1::2, :]
                bb = field[1::2, ::2, 1::2, :]
                cc = field[::2, 1::2, 1::2, :]
                dd = field[1::2, 1::2, 1::2, :]

                expected = 0.125 * (a + b + c + d + aa + bb + cc + dd)

                if reduce_s:
                    expected[-1, :, :, :] *= 2
                if reduce_r:
                    expected[:, -1, :, :] *= 2
                if reduce_c:
                    expected[:, :, -1, :] *= 2

                actual =\
                    vfu.downsample_displacement_field_3d(field[:ns, :nr, :nc])
                assert_array_almost_equal(expected, actual)


def test_reorient_vector_field_2d():
    shape = (16, 16)
    d, dinv = vfu.create_harmonic_fields_2d(shape[0], shape[1], 0.2, 4)
    d = np.array(d, dtype=floating)

    # the vector field rotated 90 degrees
    expected = np.empty(shape=shape + (2,), dtype=floating)
    expected[..., 0] = -1 * d[..., 1]
    expected[..., 1] = d[..., 0]

    # rotate 45 degrees twice
    c = np.sqrt(0.5)
    affine = np.array([[c, -c, 0.0], [c, c, 0.0]])
    vfu.reorient_vector_field_2d(d, affine)
    vfu.reorient_vector_field_2d(d, affine)

    # verify almost equal
    assert_array_almost_equal(d, expected)

    # Test exception is raised when the affine transform matrix is not valid
    invalid = np.zeros((2, 2), dtype=np.float64)
    assert_raises(ValueError, vfu.reorient_vector_field_2d, d, invalid)


def test_reorient_vector_field_3d():
    sh = (16, 16, 16)
    d, dinv = vfu.create_harmonic_fields_3d(sh[0], sh[1], sh[2], 0.2, 4)
    d = np.array(d, dtype=floating)
    dinv = np.array(dinv, dtype=floating)

    # the vector field rotated 90 degrees around the last axis
    expected = np.empty(shape=sh + (3,), dtype=floating)
    expected[..., 0] = -1 * d[..., 1]
    expected[..., 1] = d[..., 0]
    expected[..., 2] = d[..., 2]

    # rotate 45 degrees twice around the last axis
    c = np.sqrt(0.5)
    affine = np.array([[c, -c, 0, 0], [c, c, 0, 0], [0, 0, 1, 0]])
    vfu.reorient_vector_field_3d(d, affine)
    vfu.reorient_vector_field_3d(d, affine)

    # verify almost equal
    assert_array_almost_equal(d, expected)

    # the vector field rotated 90 degrees around the first axis
    expected[..., 0] = dinv[..., 0]
    expected[..., 1] = -1 * dinv[..., 2]
    expected[..., 2] = dinv[..., 1]

    # rotate 45 degrees twice around the first axis
    affine = np.array([[1, 0, 0, 0], [0, c, -c, 0], [0, c, c, 0]])
    vfu.reorient_vector_field_3d(dinv, affine)
    vfu.reorient_vector_field_3d(dinv, affine)

    # verify almost equal
    assert_array_almost_equal(dinv, expected)

    # Test exception is raised when the affine transform matrix is not valid
    invalid = np.zeros((3, 3), dtype=np.float64)
    assert_raises(ValueError, vfu.reorient_vector_field_3d, d, invalid)


@set_random_number_generator(1134781)
def test_reorient_random_vector_fields(rng):
    # Test reorienting vector field
    for n_dims, func in ((2, vfu.reorient_vector_field_2d),
                         (3, vfu.reorient_vector_field_3d)):
        size = [20, 30, 40][:n_dims] + [n_dims]
        arr = rng.normal(size=size)
        arr_32 = arr.astype(floating)
        affine = from_matvec(rng.normal(size=(n_dims, n_dims)),
                             np.zeros(n_dims))
        func(arr_32, affine)
        assert_almost_equal(arr_32, apply_affine(affine, arr), 6)
        # Reorient reorients without translation
        trans = np.arange(n_dims) + 2
        affine[:-1, -1] = trans
        arr_32 = arr.astype(floating)
        func(arr_32, affine)
        assert_almost_equal(arr_32, apply_affine(affine, arr) - trans, 6)

        # Test exception is raised when the affine transform is not valid
        invalid = np.eye(n_dims)
        assert_raises(ValueError, func, arr_32, invalid)


@set_random_number_generator(3921116)
def test_gradient_2d(rng):
    sh = (25, 32)
    # Create grid coordinates
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    X = np.empty(sh + (3,), dtype=np.float64)
    O = np.ones(sh)
    X[..., 0] = x_0[:, None] * O
    X[..., 1] = x_1[None, :] * O
    X[..., 2] = 1

    transform = regtransforms[('RIGID', 2)]
    theta = np.array([0.1, 5.0, 2.5])
    T = transform.param_to_matrix(theta)
    TX = X.dot(T.T)
    # Eval an arbitrary (known) function at TX
    # f(x, y) = ax^2 + bxy + cy^{2}
    # df/dx = 2ax + by
    # df/dy = 2cy + bx
    a = 2e-3
    b = 5e-3
    c = 7e-3
    img = a * TX[..., 0] ** 2 +\
        b * TX[..., 0] * TX[..., 1] +\
        c * TX[..., 1] ** 2
    img = img.astype(floating)
    # img is an image sampled at X with grid-to-space transform T

    # Test sparse gradient: choose some sample points (in space)
    sample = sample_domain_regular(20, np.array(sh, dtype=np.int32),
                                   T, rng=rng)
    sample = np.array(sample)
    # Compute the analytical gradient at all points
    expected = np.empty((sample.shape[0], 2), dtype=floating)
    expected[..., 0] = 2 * a * sample[:, 0] + b * sample[:, 1]
    expected[..., 1] = 2 * c * sample[:, 1] + b * sample[:, 0]
    # Get the numerical gradient with the implementation under test
    sp_to_grid = np.linalg.inv(T)
    img_spacing = np.ones(2)
    actual, inside = vfu.sparse_gradient(img, sp_to_grid, img_spacing, sample)
    diff = np.abs(expected - actual).mean(1) * inside
    # The finite differences are really not accurate, especially with float32
    assert_equal(diff.max() < 1e-3, True)
    # Verify exception is raised when passing invalid affine or spacings
    invalid_affine = np.eye(2)
    invalid_spacings = np.ones(1)
    assert_raises(ValueError, vfu.sparse_gradient, img, invalid_affine,
                  img_spacing, sample)
    assert_raises(ValueError, vfu.sparse_gradient, img, sp_to_grid,
                  invalid_spacings, sample)

    # Test dense gradient
    # Compute the analytical gradient at all points
    expected = np.empty(sh + (2,), dtype=floating)
    expected[..., 0] = 2 * a * TX[..., 0] + b * TX[..., 1]
    expected[..., 1] = 2 * c * TX[..., 1] + b * TX[..., 0]
    # Get the numerical gradient with the implementation under test
    sp_to_grid = np.linalg.inv(T)
    img_spacing = np.ones(2)
    actual, inside = vfu.gradient(img, sp_to_grid, img_spacing, sh, T)
    diff = np.abs(expected - actual).mean(2) * inside
    # In the dense case, we are evaluating at the exact points (sample points
    # are not slightly moved like in the sparse case) so we have more precision
    assert_equal(diff.max() < 1e-5, True)
    # Verify exception is raised when passing invalid affine or spacings
    assert_raises(ValueError, vfu.gradient, img, invalid_affine, img_spacing,
                  sh, T)
    assert_raises(ValueError, vfu.gradient, img, sp_to_grid, img_spacing,
                  sh, invalid_affine)
    assert_raises(ValueError, vfu.gradient, img, sp_to_grid, invalid_spacings,
                  sh, T)


@set_random_number_generator(3921116)
def test_gradient_3d(rng):
    shape = (25, 32, 15)
    # Create grid coordinates
    x_0 = np.asarray(range(shape[0]))
    x_1 = np.asarray(range(shape[1]))
    x_2 = np.asarray(range(shape[2]))
    X = np.zeros(shape + (4,), dtype=np.float64)
    O = np.ones(shape)
    X[..., 0] = x_0[:, None, None] * O
    X[..., 1] = x_1[None, :, None] * O
    X[..., 2] = x_2[None, None, :] * O
    X[..., 3] = 1

    transform = regtransforms[('RIGID', 3)]
    theta = np.array([0.1, 0.05, 0.12, -12.0, -15.5, -7.2])
    T = transform.param_to_matrix(theta)

    TX = X.dot(T.T)
    # Eval an arbitrary (known) function at TX
    # f(x, y, z) = ax^2 + by^2 + cz^2 + dxy + exz + fyz
    # df/dx = 2ax + dy + ez
    # df/dy = 2by + dx + fz
    # df/dz = 2cz + ex + fy
    a, b, c = 2e-3, 3e-3, 1e-3
    d, e, f = 1e-3, 2e-3, 3e-3
    img = a * TX[..., 0] ** 2 + b * TX[..., 1] ** 2 +\
        c * TX[..., 2] ** 2 + d * TX[..., 0] * TX[..., 1] +\
        e * TX[..., 0] * TX[..., 2] + f * TX[..., 1] * TX[..., 2]

    img = img.astype(floating)
    # Test sparse gradient: choose some sample points (in space)
    sample =\
        sample_domain_regular(100, np.array(shape, dtype=np.int32),
                              T, rng=rng)
    sample = np.array(sample)
    # Compute the analytical gradient at all points
    expected = np.empty((sample.shape[0], 3), dtype=floating)
    expected[..., 0] =\
        2 * a * sample[:, 0] + d * sample[:, 1] + e * sample[:, 2]
    expected[..., 1] =\
        2 * b * sample[:, 1] + d * sample[:, 0] + f * sample[:, 2]
    expected[..., 2] =\
        2 * c * sample[:, 2] + e * sample[:, 0] + f * sample[:, 1]
    # Get the numerical gradient with the implementation under test
    sp_to_grid = np.linalg.inv(T)
    img_spacing = np.ones(3)
    actual, inside = vfu.sparse_gradient(img, sp_to_grid, img_spacing, sample)
    # Discard points outside the image domain
    diff = np.abs(expected - actual).mean(1) * inside
    # The finite differences are really not accurate, especially with float32
    assert_equal(diff.max() < 1e-3, True)
    # Verify exception is raised when passing invalid affine or spacings
    invalid_affine = np.eye(3)
    invalid_spacings = np.ones(2)
    assert_raises(ValueError, vfu.sparse_gradient, img, invalid_affine,
                  img_spacing, sample)
    assert_raises(ValueError, vfu.sparse_gradient, img, sp_to_grid,
                  invalid_spacings, sample)

    # Test dense gradient
    # Compute the analytical gradient at all points
    expected = np.empty(shape + (3,), dtype=floating)
    expected[..., 0] = 2 * a * TX[..., 0] + d * TX[..., 1] + e * TX[..., 2]
    expected[..., 1] = 2 * b * TX[..., 1] + d * TX[..., 0] + f * TX[..., 2]
    expected[..., 2] = 2 * c * TX[..., 2] + e * TX[..., 0] + f * TX[..., 1]
    # Get the numerical gradient with the implementation under test
    sp_to_grid = np.linalg.inv(T)
    img_spacing = np.ones(3)
    actual, inside = vfu.gradient(img, sp_to_grid, img_spacing, shape, T)
    diff = np.abs(expected - actual).mean(3) * inside
    # In the dense case, we are evaluating at the exact points (sample points
    # are not slightly moved like in the sparse case) so we have more precision
    assert_equal(diff.max() < 1e-5, True)
    # Verify exception is raised when passing invalid affine or spacings
    assert_raises(ValueError, vfu.gradient, img, invalid_affine, img_spacing,
                  shape, T)
    assert_raises(ValueError, vfu.gradient, img, sp_to_grid, img_spacing,
                  shape, invalid_affine)
    assert_raises(ValueError, vfu.gradient, img, sp_to_grid, invalid_spacings,
                  shape, T)
