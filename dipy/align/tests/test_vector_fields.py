import numpy as np
from dipy.align import floating
import dipy.align.vector_fields as vfu
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           assert_almost_equal,
                           assert_equal)
import dipy.align.imwarp as imwarp
from nibabel.affines import apply_affine, from_matvec
from scipy.ndimage.interpolation import map_coordinates
import dipy.core.geometry as geometry


def test_random_displacement_field_2d():
    np.random.seed(3921116)
    from_shape = (25, 32)
    to_shape = (33, 29)

    # Create grid coordinates
    x_0 = np.asarray(range(from_shape[0]))
    x_1 = np.asarray(range(from_shape[1]))
    X = np.ndarray((3,)+from_shape, dtype = np.float64)
    O = np.ones(from_shape)
    X[0, ...]= x_0[:, None] * O
    X[1, ...]= x_1[None, :] * O
    X[2, ...]= 1

    # Create an arbitrary image-to-space transform
    t = 0.15 #translation factor

    trans = np.array([[1, 0, -t*from_shape[0]],
                      [0, 1, -t*from_shape[1]],
                      [0, 0, 1]])
    trans_inv = np.linalg.inv(trans)

    for theta in [-1 * np.pi/6.0, 0.0, np.pi/5.0]: #rotation angle
        for s in [0.83,  1.3, 2.07]: #scale
            ct = np.cos(theta)
            st = np.sin(theta)

            rot = np.array([[ct, -st, 0],
                            [st, ct, 0],
                            [0, 0, 1]])

            scale = np.array([[1*s, 0, 0],
                              [0, 1*s, 0],
                              [0, 0, 1]])

            from_affine = trans_inv.dot(scale.dot(rot.dot(trans)))
            to_affine = from_affine.dot(scale)
            to_affine_inv = np.linalg.inv(to_affine)

            field, assignment = vfu.create_random_displacement_2d(np.array(from_shape, dtype=np.int32),
                                                              from_affine,
                                                              np.array(to_shape, dtype=np.int32),
                                                              to_affine)
            field = np.array(field, dtype=floating)
            assignment = np.array(assignment)
            # Verify the assignments are inside the requested region
            assert_equal(0, (assignment<0).sum())
            for i in range(2):
                assert_equal(0, (assignment[...,i]>=to_shape[i]).sum())

            # Compute the warping coordinates (see warp_2d documentation)
            Y = np.apply_along_axis(from_affine.dot, 0, X)[0:2,...]
            Z = np.zeros_like(X)
            Z[0,...] = Y[0,...] + field[...,0]
            Z[1,...] = Y[1,...] + field[...,1]
            Z[2,...] = 1
            W = np.apply_along_axis(to_affine_inv.dot, 0, Z)[0:2,...]

            # Verify the claimed assignments are correct
            assert_array_almost_equal(W[0,...], assignment[...,0], 5)
            assert_array_almost_equal(W[1,...], assignment[...,1], 5)


def test_random_displacement_field_3d():
    np.random.seed(7127562)
    from_shape = (25, 32, 31)
    to_shape = (33, 29, 35)

    # Create grid coordinates
    x_0 = np.asarray(range(from_shape[0]))
    x_1 = np.asarray(range(from_shape[1]))
    x_2 = np.asarray(range(from_shape[2]))
    X = np.ndarray((4,)+from_shape, dtype = np.float64)
    O = np.ones(from_shape)
    X[0, ...]= x_0[:, None, None] * O
    X[1, ...]= x_1[None, :, None] * O
    X[2, ...]= x_2[None, None, :] * O
    X[3, ...]= 1

    # Select an arbitrary rotation axis
    axis = np.array([.5, 2.0, 1.5])

    # Create an arbitrary image-to-space transform
    t = 0.15 #translation factor

    trans = np.array([[1, 0, 0, -t*from_shape[0]],
                      [0, 1, 0, -t*from_shape[1]],
                      [0, 0, 1, -t*from_shape[2]],
                      [0, 0, 0, 1]])
    trans_inv = np.linalg.inv(trans)

    for theta in [-1 * np.pi/6.0, 0.0, np.pi/5.0]: #rotation angle
        for s in [0.83,  1.3, 2.07]: #scale
            rot = np.zeros(shape=(4,4))
            rot[:3, :3] = geometry.rodrigues_axis_rotation(axis, theta)
            rot[3,3] = 1.0

            scale = np.array([[1*s, 0, 0, 0],
                              [0, 1*s, 0, 0],
                              [0, 0, 1*s, 0],
                              [0, 0, 0, 1]])

            from_affine = trans_inv.dot(scale.dot(rot.dot(trans)))
            to_affine = from_affine.dot(scale)
            to_affine_inv = np.linalg.inv(to_affine)

            field, assignment = vfu.create_random_displacement_3d(np.array(from_shape, dtype=np.int32),
                                                              from_affine,
                                                              np.array(to_shape, dtype=np.int32),
                                                              to_affine)
            field = np.array(field, dtype=floating)
            assignment = np.array(assignment)
            # Verify the assignments are inside the requested region
            assert_equal(0, (assignment<0).sum())
            for i in range(3):
                assert_equal(0, (assignment[...,i]>=to_shape[i]).sum())

            # Compute the warping coordinates (see warp_2d documentation)
            Y = np.apply_along_axis(from_affine.dot, 0, X)[0:3,...]
            Z = np.zeros_like(X)
            Z[0,...] = Y[0,...] + field[...,0]
            Z[1,...] = Y[1,...] + field[...,1]
            Z[2,...] = Y[2,...] + field[...,2]
            Z[3,...] = 1
            W = np.apply_along_axis(to_affine_inv.dot, 0, Z)[0:3,...]

            # Verify the claimed assignments are correct
            assert_array_almost_equal(W[0,...], assignment[...,0], 5)
            assert_array_almost_equal(W[1,...], assignment[...,1], 5)
            assert_array_almost_equal(W[2,...], assignment[...,2], 5)


def test_harmonic_fields_2d():
    nrows = 64
    ncols = 67
    mid_row = nrows//2
    mid_col = ncols//2
    expected_d = np.ndarray(shape = (nrows, ncols, 2))
    expected_d_inv = np.ndarray(shape = (nrows, ncols, 2))
    for b in [0.1, 0.3, 0.7]:
        for m in [2, 4, 7]:
            for i in range(nrows):
                for j in range(ncols):
                    ii = i - mid_row
                    jj = j - mid_col
                    theta = np.arctan2(ii, jj)
                    expected_d[i, j, 0]=ii * (1.0 / (1 + b * np.cos(m * theta)) - 1.0)
                    expected_d[i, j, 1]=jj * (1.0 / (1 + b * np.cos(m * theta)) - 1.0)
                    expected_d_inv[i,j,0] = b * np.cos(m * theta) * ii
                    expected_d_inv[i,j,1] = b * np.cos(m * theta) * jj

            actual_d, actual_d_inv = vfu.create_harmonic_fields_2d(nrows, ncols, b, m)
            assert_array_almost_equal(expected_d, actual_d)
            assert_array_almost_equal(expected_d_inv, expected_d_inv)


def test_harmonic_fields_3d():
    nslices = 25
    nrows = 34
    ncols = 37
    mid_slice = nslices//2
    mid_row = nrows//2
    mid_col = ncols//2
    expected_d = np.ndarray(shape = (nslices, nrows, ncols, 3))
    expected_d_inv = np.ndarray(shape = (nslices, nrows, ncols, 3))
    for b in [0.3, 0.7]:
        for m in [2, 5]:
            for k in range(nslices):
                for i in range(nrows):
                    for j in range(ncols):
                        kk = k - mid_slice
                        ii = i - mid_row
                        jj = j - mid_col
                        theta = np.arctan2(ii, jj)
                        expected_d[k, i, j, 0]=kk * (1.0 / (1 + b * np.cos(m * theta)) - 1.0)
                        expected_d[k, i, j, 1]=ii * (1.0 / (1 + b * np.cos(m * theta)) - 1.0)
                        expected_d[k, i, j, 2]=jj * (1.0 / (1 + b * np.cos(m * theta)) - 1.0)
                        expected_d_inv[k, i, j, 0] = b * np.cos(m * theta) * kk
                        expected_d_inv[k, i, j, 1] = b * np.cos(m * theta) * ii
                        expected_d_inv[k, i, j, 2] = b * np.cos(m * theta) * jj

            actual_d, actual_d_inv = vfu.create_harmonic_fields_3d(nslices, nrows, ncols, b, m)
            assert_array_almost_equal(expected_d, actual_d)
            assert_array_almost_equal(expected_d_inv, expected_d_inv)


def test_circle():
    sh = (64, 61)
    cr = sh[0]//2
    cc = sh[1]//2
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    X = np.ndarray((2,)+sh, dtype = np.float64)
    O = np.ones(sh)
    X[0, ...]= x_0[:, None] * O - cr
    X[1, ...]= x_1[None, :] * O - cc
    nrm = np.sqrt(np.sum(X**2,axis = 0))
    for radius in [0, 7, 17, 32]:
        expected = nrm<=radius
        actual = vfu.create_circle(sh[0], sh[1], radius)
        assert_array_almost_equal(actual, expected)


def test_sphere():
    sh = (64, 61, 57)
    cs = sh[0]//2
    cr = sh[1]//2
    cc = sh[2]//2
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    x_2 = np.asarray(range(sh[2]))
    X = np.ndarray((3,)+sh, dtype = np.float64)
    O = np.ones(sh)
    X[0, ...]= x_0[:, None, None] * O - cs
    X[1, ...]= x_1[None, :, None] * O - cr
    X[2, ...]= x_2[None, None, :] * O - cc
    nrm = np.sqrt(np.sum(X**2,axis = 0))
    for radius in [0, 7, 17, 32]:
        expected = nrm<=radius
        actual = vfu.create_sphere(sh[0], sh[1], sh[2], radius)
        assert_array_almost_equal(actual, expected)


def test_interpolate_scalar_2d():
    np.random.seed(5324989)
    sz = 64
    target_shape = (sz, sz)
    image = np.ndarray(target_shape, dtype=floating)
    image[...] = np.random.randint(0, 10, np.size(image)).reshape(target_shape)

    extended_image = np.zeros((sz+2, sz+2), dtype=floating)
    extended_image[1:sz+1, 1:sz+1] = image[...]

    #Select some coordinates inside the image to interpolate at
    nsamples = 200
    locations = np.random.ranf(2 * nsamples).reshape((nsamples,2)) * (sz+2) - 1.0
    extended_locations = locations + 1.0 # shift coordinates one voxel

    #Call the implementation under test
    interp, inside = vfu.interpolate_scalar_2d(image, locations)

    #Call the reference implementation
    expected = map_coordinates(extended_image, extended_locations.transpose(), order=1)

    assert_array_almost_equal(expected, interp)

    #Test interpolation stability along the boundary
    epsilon = 5e-8
    for k in range(2):
        for offset in [0, sz-1]:
            delta = ((np.random.ranf(nsamples) * 2) -1) * epsilon
            locations[:, k] = delta + offset
            locations[:, (k+1)%2] = np.random.ranf(nsamples) * (sz-1)
            interp, inside = vfu.interpolate_scalar_2d(image, locations)

            locations[:, k] = offset
            expected = map_coordinates(image, locations.transpose(), order=1)
            assert_array_almost_equal(expected, interp)
            if offset == 0:
                expected_flag = np.array(delta>=0, dtype = np.int32)
            else:
                expected_flag = np.array(delta<=0, dtype = np.int32)
            assert_array_almost_equal(expected_flag, inside)


def test_interpolate_scalar_nn_2d():
    np.random.seed(1924781)
    sz = 64
    target_shape = (sz, sz)
    image = np.ndarray(target_shape, dtype=floating)
    image[...] = np.random.randint(0, 10, np.size(image)).reshape(target_shape)
    #Select some coordinates to interpolate at
    nsamples = 200
    locations = np.random.ranf(2 * nsamples).reshape((nsamples,2)) * (sz+2) - 1.0

    #Call the implementation under test
    interp, inside = vfu.interpolate_scalar_nn_2d(image, locations)

    #Call the reference implementation
    expected = map_coordinates(image, locations.transpose(), order=0)

    assert_array_almost_equal(expected, interp)

    #Test the 'inside' flag
    for i in range(nsamples):
        if (locations[i, 0]<0 or locations[i, 0]>(sz-1)) or (locations[i, 1]<0 or locations[i, 1]>(sz-1)):
            assert_equal(inside[i], 0)
        else:
            assert_equal(inside[i], 1)


def test_interpolate_scalar_nn_3d():
    np.random.seed(3121121)
    sz = 64
    target_shape = (sz, sz, sz)
    image = np.ndarray(target_shape, dtype=floating)
    image[...] = np.random.randint(0, 10, np.size(image)).reshape(target_shape)
    #Select some coordinates to interpolate at
    nsamples = 200
    locations = np.random.ranf(3 * nsamples).reshape((nsamples,3)) * (sz+2) - 1.0

    #Call the implementation under test
    interp, inside = vfu.interpolate_scalar_nn_3d(image, locations)

    #Call the reference implementation
    expected = map_coordinates(image, locations.transpose(), order=0)

    assert_array_almost_equal(expected, interp)

    #Test the 'inside' flag
    for i in range(nsamples):
        expected_inside = 1
        for axis in range(3):
            if (locations[i, axis]<0 or locations[i, axis]>(sz-1)):
                expected_inside = 0
                break
        assert_equal(inside[i], expected_inside)


def test_interpolate_scalar_3d():
    np.random.seed(9216326)
    sz = 64
    target_shape = (sz, sz, sz)
    image = np.ndarray(target_shape, dtype=floating)
    image[...] = np.random.randint(0, 10, np.size(image)).reshape(target_shape)

    extended_image = np.zeros((sz+2, sz+2, sz+2), dtype=floating)
    extended_image[1:sz+1, 1:sz+1, 1:sz+1] = image[...]

    #Select some coordinates inside the image to interpolate at
    nsamples = 800
    locations = np.random.ranf(3 * nsamples).reshape((nsamples,3)) * (sz+2) - 1.0
    extended_locations = locations + 1.0 # shift coordinates one voxel

    #Call the implementation under test
    interp, inside = vfu.interpolate_scalar_3d(image, locations)

    #Call the reference implementation
    expected = map_coordinates(extended_image, extended_locations.transpose(), order=1)

    assert_array_almost_equal(expected, interp)

    #Test interpolation stability along the boundary
    epsilon = 5e-8
    for k in range(3):
        for offset in [0, sz-1]:
            delta = ((np.random.ranf(nsamples) * 2) -1) * epsilon
            locations[:, k] = delta + offset
            locations[:, (k+1)%3] = np.random.ranf(nsamples) * (sz-1)
            locations[:, (k+2)%3] = np.random.ranf(nsamples) * (sz-1)
            interp, inside = vfu.interpolate_scalar_3d(image, locations)

            locations[:, k] = offset
            expected = map_coordinates(image, locations.transpose(), order=1)
            assert_array_almost_equal(expected, interp)

            if offset == 0:
                expected_flag = np.array(delta>=0, dtype = np.int32)
            else:
                expected_flag = np.array(delta<=0, dtype = np.int32)
            assert_array_almost_equal(expected_flag, inside)


def test_interpolate_vector_3d():
    np.random.seed(7711219)
    sz = 64
    target_shape = (sz, sz, sz)
    field = np.ndarray(target_shape+(3,), dtype=floating)
    field[...] = np.random.randint(0, 10, np.size(field)).reshape(target_shape+(3,))

    extended_field = np.zeros((sz+2, sz+2, sz+2, 3), dtype=floating)
    extended_field[1:sz+1, 1:sz+1, 1:sz+1] = field
    #Select some coordinates to interpolate at
    nsamples = 800
    locations = np.random.ranf(3 * nsamples).reshape((nsamples,3)) * (sz+2) - 1.0
    extended_locations = locations + 1

    #Call the implementation under test
    interp, inside = vfu.interpolate_vector_3d(field, locations)

    #Call the reference implementation
    expected = np.zeros_like(interp)
    for i in range(3):
        expected[...,i] = map_coordinates(extended_field[...,i], extended_locations.transpose(), order=1)

    assert_array_almost_equal(expected, interp)

    #Test interpolation stability along the boundary
    epsilon = 5e-8
    for k in range(3):
        for offset in [0, sz-1]:
            delta = ((np.random.ranf(nsamples) * 2) -1) * epsilon
            locations[:, k] = delta + offset
            locations[:, (k+1)%3] = np.random.ranf(nsamples) * (sz-1)
            locations[:, (k+2)%3] = np.random.ranf(nsamples) * (sz-1)
            interp, inside = vfu.interpolate_vector_3d(field, locations)

            locations[:, k] = offset
            for i in range(3):
                expected[...,i] = map_coordinates(field[...,i], locations.transpose(), order=1)
            assert_array_almost_equal(expected, interp)

            if offset == 0:
                expected_flag = np.array(delta>=0, dtype = np.int32)
            else:
                expected_flag = np.array(delta<=0, dtype = np.int32)
            assert_array_almost_equal(expected_flag, inside)


def test_interpolate_vector_2d():
    np.random.seed(1271244)
    sz = 64
    target_shape = (sz, sz)
    field = np.ndarray(target_shape+(2,), dtype=floating)
    field[...] = np.random.randint(0, 10, np.size(field)).reshape(target_shape+(2,))
    extended_field = np.zeros((sz+2, sz+2, 2), dtype=floating)
    extended_field[1:sz+1, 1:sz+1] = field
    #Select some coordinates to interpolate at
    nsamples = 200
    locations = np.random.ranf(2 * nsamples).reshape((nsamples,2)) * (sz+2) - 1.0
    extended_locations = locations + 1

    #Call the implementation under test
    interp, inside = vfu.interpolate_vector_2d(field, locations)

    #Call the reference implementation
    expected = np.zeros_like(interp)
    for i in range(2):
        expected[...,i] = map_coordinates(extended_field[...,i], extended_locations.transpose(), order=1)

    assert_array_almost_equal(expected, interp)

    #Test interpolation stability along the boundary
    epsilon = 5e-8
    for k in range(2):
        for offset in [0, sz-1]:
            delta = ((np.random.ranf(nsamples) * 2) -1) * epsilon
            locations[:, k] = delta + offset
            locations[:, (k+1)%2] = np.random.ranf(nsamples) * (sz-1)
            interp, inside = vfu.interpolate_vector_2d(field, locations)

            locations[:, k] = offset
            for i in range(2):
                expected[...,i] = map_coordinates(field[...,i], locations.transpose(), order=1)
            assert_array_almost_equal(expected, interp)

            if offset == 0:
                expected_flag = np.array(delta>=0, dtype = np.int32)
            else:
                expected_flag = np.array(delta<=0, dtype = np.int32)
            assert_array_almost_equal(expected_flag, inside)



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
    circle = np.array(circle, dtype = floating)

    # Create a displacement field for warping
    d, dinv = vfu.create_harmonic_fields_2d(nr, nc, 0.2, 8)
    d = np.asarray(d).astype(floating)
    dinv = np.asarray(dinv).astype(floating)

    # Create grid coordinates
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    X = np.ndarray((3,)+sh, dtype = np.float64)
    O = np.ones(sh)
    X[0, ...]= x_0[:, None] * O
    X[1, ...]= x_1[None, :] * O
    X[2, ...]= 1

    # Select an arbitrary translation matrix
    t = 0.1
    trans = np.array([[1, 0, -t*nr],
                      [0, 1, -t*nc],
                      [0, 0, 1]])
    trans_inv = np.linalg.inv(trans)

    # Select arbitrary rotation and scaling matrices
    for theta in [-1 * np.pi/6.0, 0.0, np.pi/6.0]: #rotation angle
        for s in [0.42,  1.3, 2.15]: #scale
            ct = np.cos(theta)
            st = np.sin(theta)

            rot = np.array([[ct, -st, 0],
                            [st, ct, 0],
                            [0, 0, 1]])

            scale = np.array([[1*s, 0, 0],
                              [0, 1*s, 0],
                              [0, 0, 1]])

            aff = trans_inv.dot(scale.dot(rot.dot(trans)))

            # Select arbitrary (but different) grid-to-space transforms
            sampling_affine = scale
            field_affine =  aff
            field_affine_inv = np.linalg.inv(field_affine)
            image_affine = aff.dot(scale)
            image_affine_inv = np.linalg.inv(image_affine)

            A = field_affine_inv.dot(sampling_affine)
            B = image_affine_inv.dot(sampling_affine)
            C = image_affine_inv

            # Reorient the displacement field according to its grid-to-space transform
            dcopy = np.copy(d)
            vfu.reorient_vector_field_2d(dcopy, field_affine)
            extended_dcopy = np.zeros((nr+2, nc+2, 2), dtype=floating)
            extended_dcopy[1:nr+1, 1:nc+1, :] = dcopy

            # Compute the warping coordinates (see warp_2d documentation)
            Y = np.apply_along_axis(A.dot, 0, X)[0:2,...]
            Z = np.zeros_like(X)
            Z[0,...] = map_coordinates(extended_dcopy[...,0], Y + 1, order=1)
            Z[1,...] = map_coordinates(extended_dcopy[...,1], Y + 1, order=1)
            Z[2,...] = 0
            Z = np.apply_along_axis(C.dot, 0, Z)[0:2,...]
            T = np.apply_along_axis(B.dot, 0, X)[0:2,...]
            W = T + Z

            #Test bilinear interpolation
            expected = map_coordinates(circle, W, order=1)
            warped = vfu.warp_2d(circle, dcopy, A, B, C, np.array(sh, dtype=np.int32))
            assert_array_almost_equal(warped, expected)

            #Test nearest neighbor interpolation
            expected = map_coordinates(circle, W, order=0)
            warped = vfu.warp_2d_nn(circle, dcopy, A, B, C, np.array(sh, dtype=np.int32))
            assert_array_almost_equal(warped, expected)


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
    sphere = np.array(sphere, dtype = floating)

    # Create a displacement field for warping
    d, dinv = vfu.create_harmonic_fields_3d(ns, nr, nc, 0.2, 8)
    d = np.asarray(d).astype(floating)
    dinv = np.asarray(dinv).astype(floating)

    # Create grid coordinates
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    x_2 = np.asarray(range(sh[2]))
    X = np.ndarray((4,)+sh, dtype = np.float64)
    O = np.ones(sh)
    X[0, ...]= x_0[:, None, None] * O
    X[1, ...]= x_1[None, :, None] * O
    X[2, ...]= x_2[None, None, :] * O
    X[3, ...]= 1

    # Select an arbitrary rotation axis
    axis = np.array([.5, 2.0, 1.5])
    # Select an arbitrary translation matrix
    t = 0.1
    trans = np.array([[1, 0, 0, -t*ns],
                      [0, 1, 0, -t*nr],
                      [0, 0, 1, -t*nc],
                      [0, 0, 0, 1]])
    trans_inv = np.linalg.inv(trans)

    # Select arbitrary rotation and scaling matrices
    for theta in [-1 * np.pi/5.0, 0.0, np.pi/5.0]: #rotation angle
        for s in [0.45,  1.1, 2.0]: #scale
            rot = np.zeros(shape=(4,4))
            rot[:3, :3] = geometry.rodrigues_axis_rotation(axis, theta)
            rot[3,3] = 1.0

            scale = np.array([[1*s, 0, 0, 0],
                              [0, 1*s, 0, 0],
                              [0, 0, 1*s, 0],
                              [0, 0, 0, 1]])

            aff = trans_inv.dot(scale.dot(rot.dot(trans)))

            # Select arbitrary (but different) grid-to-space transforms
            sampling_affine = scale
            field_affine =  aff
            field_affine_inv = np.linalg.inv(field_affine)
            image_affine = aff.dot(scale)
            image_affine_inv = np.linalg.inv(image_affine)

            A = field_affine_inv.dot(sampling_affine)
            B = image_affine_inv.dot(sampling_affine)
            C = image_affine_inv

            # Reorient the displacement field according to its grid-to-space transform
            dcopy = np.copy(d)
            vfu.reorient_vector_field_3d(dcopy, field_affine)

            extended_dcopy = np.zeros((ns+2, nr+2, nc+2, 3), dtype=floating)
            extended_dcopy[1:ns+1, 1:nr+1, 1:nc+1, :] = dcopy

            # Compute the warping coordinates (see warp_2d documentation)
            Y = np.apply_along_axis(A.dot, 0, X)[0:3,...]
            Z = np.zeros_like(X)
            Z[0,...] = map_coordinates(extended_dcopy[...,0], Y + 1, order=1)
            Z[1,...] = map_coordinates(extended_dcopy[...,1], Y + 1, order=1)
            Z[2,...] = map_coordinates(extended_dcopy[...,2], Y + 1, order=1)
            Z[3,...] = 0
            Z = np.apply_along_axis(C.dot, 0, Z)[0:3,...]
            T = np.apply_along_axis(B.dot, 0, X)[0:3,...]
            W = T + Z

            #Test bilinear interpolation
            expected = map_coordinates(sphere, W, order=1)
            warped = vfu.warp_3d(sphere, dcopy, A, B, C, np.array(sh, dtype=np.int32))
            assert_array_almost_equal(warped, expected, decimal=5)

            #Test nearest neighbor interpolation
            expected = map_coordinates(sphere, W, order=0)
            warped = vfu.warp_3d_nn(sphere, dcopy, A, B, C, np.array(sh, dtype=np.int32))
            assert_array_almost_equal(warped, expected, decimal=5)


def test_affine_warping_2d():
    r"""
    Tests 2D affine warping functions against scipy implementation
    """
    # Create a simple invertible affine transform
    domain_shape = (64, 64)
    codomain_shape = (80, 80)
    nr = domain_shape[0]
    nc = domain_shape[1]

    # Create an image of a circle
    radius = 16
    circle = vfu.create_circle(codomain_shape[0], codomain_shape[1], radius)
    circle = np.array(circle, dtype = floating)

    # Create grid coordinates
    x_0 = np.asarray(range(domain_shape[0]))
    x_1 = np.asarray(range(domain_shape[1]))
    X = np.ndarray((3,)+domain_shape, dtype = np.float64)
    O = np.ones(domain_shape)
    X[0, ...]= x_0[:, None] * O
    X[1, ...]= x_1[None, :] * O
    X[2, ...]= 1

    # Generate affine transforms
    t = 0.3
    trans = np.array([[1, 0, -t*nr],
                      [0, 1, -t*nc],
                      [0, 0, 1]])
    trans_inv = np.linalg.inv(trans)
    for theta in [-1 * np.pi/5.0, 0.0, np.pi/5.0]: #rotation angle
        for s in [0.5,  1.0, 2.0]: #scale
            ct = np.cos(theta)
            st = np.sin(theta)

            rot = np.array([[ct, -st, 0],
                            [st, ct, 0],
                            [0, 0, 1]])

            scale = np.array([[1*s, 0, 0],
                              [0, 1*s, 0],
                              [0, 0, 1]])

            gt_affine = trans_inv.dot(scale.dot(rot.dot(trans)))

            # Apply the affine transform to the grid coordinates
            Y = np.apply_along_axis(gt_affine.dot, 0, X)[0:2,...]

            expected = map_coordinates(circle, Y, order=1)
            warped = vfu.warp_2d_affine(circle, np.array(domain_shape, dtype = np.int32), gt_affine)
            assert_array_almost_equal(warped, expected)

            # Test affine warping with nearest-neighbor interpolation
            expected = map_coordinates(circle, Y, order=0)
            warped = vfu.warp_2d_affine_nn(circle, np.array(domain_shape, dtype = np.int32), gt_affine)
            assert_array_almost_equal(warped, expected)

    #Test the affine = None case
    warped = vfu.warp_2d_affine(circle, np.array(codomain_shape, dtype = np.int32), None)
    assert_array_equal(warped, circle)

    warped = vfu.warp_2d_affine_nn(circle, np.array(codomain_shape, dtype = np.int32), None)
    assert_array_equal(warped, circle)


def test_affine_warping_3d():
    r"""
    Tests 3D affine warping functions against scipy implementation
    """
    # Create a simple invertible affine transform
    domain_shape = (64, 64, 64)
    codomain_shape = (80, 80, 80)
    ns = domain_shape[0]
    nr = domain_shape[1]
    nc = domain_shape[2]

    # Create an image of a sphere
    radius = 16
    sphere = vfu.create_sphere(codomain_shape[0], codomain_shape[1], codomain_shape[2], radius)
    sphere = np.array(sphere, dtype = floating)

    # Create grid coordinates
    x_0 = np.asarray(range(domain_shape[0]))
    x_1 = np.asarray(range(domain_shape[1]))
    x_2 = np.asarray(range(domain_shape[2]))
    X = np.ndarray((4,)+domain_shape, dtype = np.float64)
    O = np.ones(domain_shape)
    X[0, ...]= x_0[:, None, None] * O
    X[1, ...]= x_1[None, :, None] * O
    X[2, ...]= x_2[None, None, :] * O
    X[3, ...]= 1

    # Generate affine transforms
    # Select an arbitrary rotation axis
    axis = np.array([.5, 2.0, 1.5])
    t = 0.3
    trans = np.array([[1, 0, 0, -t*ns],
                      [0, 1, 0, -t*nr],
                      [0, 0, 1, -t*nc],
                      [0, 0, 0, 1]])
    trans_inv = np.linalg.inv(trans)
    for theta in [-1 * np.pi/5.0, 0.0, np.pi/5.0]: #rotation angle
        for s in [0.45,  1.1, 2.3]: #scale
            rot = np.zeros(shape=(4,4))
            rot[:3, :3] = geometry.rodrigues_axis_rotation(axis, theta)
            rot[3,3] = 1.0

            scale = np.array([[1*s, 0, 0, 0],
                              [0, 1*s, 0, 0],
                              [0, 0, 1*s, 0],
                              [0, 0, 0, 1]])

            gt_affine = trans_inv.dot(scale.dot(rot.dot(trans)))

            # Apply the affine transform to the grid coordinates
            Y = np.apply_along_axis(gt_affine.dot, 0, X)[0:3,...]

            expected = map_coordinates(sphere, Y, order=1)
            warped = vfu.warp_3d_affine(sphere, np.array(domain_shape, dtype = np.int32), gt_affine)
            assert_array_almost_equal(warped, expected)

            # Test affine warping with nearest-neighbor interpolation
            expected = map_coordinates(sphere, Y, order=0)
            warped = vfu.warp_3d_affine_nn(sphere, np.array(domain_shape, dtype = np.int32), gt_affine)
            assert_array_almost_equal(warped, expected)

    #Test the affine = None case
    warped = vfu.warp_3d_affine(sphere, np.array(codomain_shape, dtype = np.int32), None)
    assert_array_equal(warped, sphere)

    warped = vfu.warp_3d_affine_nn(sphere, np.array(codomain_shape, dtype = np.int32), None)
    assert_array_equal(warped, sphere)



def test_compose_vector_fields_2d():
    r"""
    Creates two random displacement field that exactly map pixels from an input
    image to an output image. The resulting displacements and their composition,
    although operating in physical space, map the points exactly (up to
    numerical precision).
    """
    np.random.seed(8315759)
    input_shape = (10, 10)
    target_shape = (10, 10)
    #create a simple affine transformation
    nr = input_shape[0]
    nc = input_shape[1]
    s = 1.5
    t = 2.5
    trans = np.array([[1, 0, -t*nr],
                      [0, 1, -t*nc],
                      [0, 0, 1]])
    trans_inv = np.linalg.inv(trans)
    scale = np.array([[1*s, 0, 0],
                      [0, 1*s, 0],
                      [0, 0, 1]])
    gt_affine = trans_inv.dot(scale.dot(trans))

    #create two random displacement fields
    input_affine = gt_affine
    target_affine = gt_affine

    disp1, assign1 = vfu.create_random_displacement_2d(np.array(input_shape,
                                                       dtype=np.int32),
                                                       input_affine,
                                                       np.array(target_shape,
                                                       dtype=np.int32),
                                                       target_affine)
    disp1 = np.array(disp1, dtype=floating)
    assign1 = np.array(assign1)

    disp2, assign2 = vfu.create_random_displacement_2d(np.array(input_shape,
                                                       dtype=np.int32),
                                                       input_affine,
                                                       np.array(target_shape,
                                                       dtype=np.int32),
                                                       target_affine)
    disp2 = np.array(disp2, dtype=floating)
    assign2 = np.array(assign2)

    #create a random image (with decimal digits) to warp
    moving_image = np.ndarray(target_shape, dtype=floating)
    moving_image[...] = np.random.randint(0, 10, np.size(moving_image)).reshape(tuple(target_shape))
    #set boundary values to zero so we don't test wrong interpolation due to
    #floating point precision
    moving_image[0,:] = 0
    moving_image[-1,:] = 0
    moving_image[:,0] = 0
    moving_image[:,-1] = 0

    #evaluate the composed warping using the exact assignments (first 1 then 2)
    warp1 = moving_image[(assign2[...,0], assign2[...,1])]
    expected = warp1[(assign1[...,0], assign1[...,1])]

    #compose the displacement fields
    target_affine_inv = np.linalg.inv(target_affine)

    target_affine_inv = np.linalg.inv(target_affine)
    premult_index = target_affine_inv.dot(input_affine)
    premult_disp = target_affine_inv

    for time_scaling in [0.25, 0.5, 1.0, 2.0, 4.0]:
        composition, stats = vfu.compose_vector_fields_2d(disp1,
                                                          disp2/time_scaling,
                                                          premult_index,
                                                          premult_disp,
                                                          time_scaling, None)
        #apply the implementation under test
        warped = np.array(vfu.warp_2d(moving_image, composition, None,
                                         premult_index, premult_disp))
        assert_array_almost_equal(warped, expected)

        #test also using nearest neighbor interpolation
        warped = np.array(vfu.warp_2d_nn(moving_image, composition, None,
                                            premult_index, premult_disp))
        assert_array_almost_equal(warped, expected)

        #test updating the displacement field instead of creating a new one
        composition = disp1.copy()
        vfu.compose_vector_fields_2d(composition,disp2/time_scaling, premult_index,
                                     premult_disp, time_scaling, composition)
        #apply the implementation under test
        warped = np.array(vfu.warp_2d(moving_image, composition, None,
                                         premult_index, premult_disp))
        assert_array_almost_equal(warped, expected)

        #test also using nearest neighbor interpolation
        warped = np.array(vfu.warp_2d_nn(moving_image, composition, None,
                                            premult_index, premult_disp))
        assert_array_almost_equal(warped, expected)

    # Test non-overlapping case
    x_0 = np.asarray(range(input_shape[0]))
    x_1 = np.asarray(range(input_shape[1]))
    X = np.ndarray(input_shape + (2,), dtype = np.float64)
    O = np.ones(input_shape)
    X[...,0]= x_0[:, None] * O
    X[...,1]= x_1[None, :] * O
    random_labels = np.random.randint(0, 2, input_shape[0]*input_shape[1]*2)
    random_labels = random_labels.reshape(input_shape+(2,))
    values = np.array([-1, target_shape[0]])
    disp1 = (values[random_labels] - X).astype(floating)
    composition, stats = vfu.compose_vector_fields_2d(disp1,
                                                      disp2,
                                                      None,
                                                      None,
                                                      1.0, None)
    assert_array_almost_equal(composition, np.zeros_like(composition))

    #test updating the displacement field instead of creating a new one
    composition = disp1.copy()
    vfu.compose_vector_fields_2d(composition, disp2, None, None, 1.0, composition)
    assert_array_almost_equal(composition, np.zeros_like(composition))


def test_compose_vector_fields_3d():
    r"""
    Creates two random displacement field that exactly map pixels from an input
    image to an output image. The resulting displacements and their composition,
    although operating in physical space, map the points exactly (up to
    numerical precision).
    """
    np.random.seed(8315759)
    input_shape = (10, 10, 10)
    target_shape = (10, 10, 10)
    #create a simple affine transformation
    ns = input_shape[0]
    nr = input_shape[1]
    nc = input_shape[2]
    s = 1.5
    t = 2.5
    trans = np.array([[1, 0, 0, -t*ns],
                      [0, 1, 0, -t*nr],
                      [0, 0, 1, -t*nc],
                      [0, 0, 0, 1]])
    trans_inv = np.linalg.inv(trans)
    scale = np.array([[1*s, 0, 0, 0],
                      [0, 1*s, 0, 0],
                      [0, 0, 1*s, 0],
                      [0, 0, 0, 1]])
    gt_affine = trans_inv.dot(scale.dot(trans))

    #create two random displacement fields
    input_affine = gt_affine
    target_affine = gt_affine

    disp1, assign1 = vfu.create_random_displacement_3d(np.array(input_shape,
                                                       dtype=np.int32),
                                                       input_affine,
                                                       np.array(target_shape,
                                                       dtype=np.int32),
                                                       target_affine)
    disp1 = np.array(disp1, dtype=floating)
    assign1 = np.array(assign1)

    disp2, assign2 = vfu.create_random_displacement_3d(np.array(input_shape,
                                                       dtype=np.int32),
                                                       input_affine,
                                                       np.array(target_shape,
                                                       dtype=np.int32),
                                                       target_affine)
    disp2 = np.array(disp2, dtype=floating)
    assign2 = np.array(assign2)

    #create a random image (with decimal digits) to warp
    moving_image = np.ndarray(target_shape, dtype=floating)
    moving_image[...] = np.random.randint(0, 10, np.size(moving_image)).reshape(tuple(target_shape))
    #set boundary values to zero so we don't test wrong interpolation due to
    #floating point precision
    moving_image[0,:,:] = 0
    moving_image[-1,:,:] = 0
    moving_image[:,0,:] = 0
    moving_image[:,-1,:] = 0
    moving_image[:,:,0] = 0
    moving_image[:,:,-1] = 0

    #evaluate the composed warping using the exact assignments (first 1 then 2)

    warp1 = moving_image[(assign2[...,0], assign2[...,1], assign2[...,2])]
    expected = warp1[(assign1[...,0], assign1[...,1], assign1[...,2])]

    #compose the displacement fields
    target_affine_inv = np.linalg.inv(target_affine)

    target_affine_inv = np.linalg.inv(target_affine)
    premult_index = target_affine_inv.dot(input_affine)
    premult_disp = target_affine_inv

    for time_scaling in [0.25, 0.5, 1.0, 2.0, 4.0]:
        composition, stats = vfu.compose_vector_fields_3d(disp1,
                                                          disp2/time_scaling,
                                                          premult_index,
                                                          premult_disp,
                                                          time_scaling, None)
        #apply the implementation under test
        warped = np.array(vfu.warp_3d(moving_image, composition, None,
                                          premult_index, premult_disp))
        assert_array_almost_equal(warped, expected)

        #test also using nearest neighbor interpolation
        warped = np.array(vfu.warp_3d_nn(moving_image, composition, None,
                                             premult_index, premult_disp))
        assert_array_almost_equal(warped, expected)

        #test updating the displacement field instead of creating a new one
        composition = disp1.copy()
        vfu.compose_vector_fields_3d(composition, disp2/time_scaling,
                                     premult_index, premult_disp,
                                     time_scaling, composition)
        #apply the implementation under test
        warped = np.array(vfu.warp_3d(moving_image, composition, None,
                                          premult_index, premult_disp))
        assert_array_almost_equal(warped, expected)

        #test also using nearest neighbor interpolation
        warped = np.array(vfu.warp_3d_nn(moving_image, composition, None,
                                             premult_index, premult_disp))
        assert_array_almost_equal(warped, expected)

    # Test non-overlapping case
    x_0 = np.asarray(range(input_shape[0]))
    x_1 = np.asarray(range(input_shape[1]))
    x_2 = np.asarray(range(input_shape[2]))
    X = np.ndarray(input_shape + (3,), dtype = np.float64)
    O = np.ones(input_shape)
    X[...,0]= x_0[:, None, None] * O
    X[...,1]= x_1[None, :, None] * O
    X[...,2]= x_2[None, None, :] * O
    random_labels = np.random.randint(0, 2, input_shape[0]*input_shape[1]*input_shape[2]*3)
    random_labels = random_labels.reshape(input_shape+(3,))
    values = np.array([-1, target_shape[0]])
    disp1 = (values[random_labels] - X).astype(floating)
    composition, stats = vfu.compose_vector_fields_3d(disp1,
                                                      disp2,
                                                      None,
                                                      None,
                                                      1.0, None)
    assert_array_almost_equal(composition, np.zeros_like(composition))

    #test updating the displacement field instead of creating a new one
    composition = disp1.copy()
    vfu.compose_vector_fields_3d(composition, disp2, None, None, 1.0, composition)
    assert_array_almost_equal(composition, np.zeros_like(composition))


def test_invert_vector_field_2d():
    r"""
    Inverts a synthetic, analytically invertible, displacement field
    """
    shape = (64, 64)
    nr = shape[0]
    nc = shape[1]
    # Create an arbitrary image-to-space transform
    t = 2.5 #translation factor

    trans = np.array([[1, 0, -t*nr],
                      [0, 1, -t*nc],
                      [0, 0, 1]])
    trans_inv = np.linalg.inv(trans)

    d, dinv = vfu.create_harmonic_fields_2d(nr, nc, 0.2, 8)
    d = np.asarray(d).astype(floating)
    dinv = np.asarray(dinv).astype(floating)

    for theta in [-1 * np.pi/5.0, 0.0, np.pi/5.0]: #rotation angle
        for s in [0.5,  1.0, 2.0]: #scale
            ct = np.cos(theta)
            st = np.sin(theta)

            rot = np.array([[ct, -st, 0],
                            [st, ct, 0],
                            [0, 0, 1]])

            scale = np.array([[1*s, 0, 0],
                              [0, 1*s, 0],
                              [0, 0, 1]])

            gt_affine = trans_inv.dot(scale.dot(rot.dot(trans)))
            gt_affine_inv = np.linalg.inv(gt_affine)
            dcopy = np.copy(d)

            #make sure the field remains invertible after the re-mapping
            vfu.reorient_vector_field_2d(dcopy, gt_affine)

            inv_approx = vfu.invert_vector_field_fixed_point_2d(dcopy, gt_affine_inv,
                                                                np.array([s, s]),
                                                                40, 1e-7)

            mapping = imwarp.DiffeomorphicMap(2, (nr,nc), gt_affine)
            mapping.forward = dcopy
            mapping.backward = inv_approx
            residual, stats = mapping.compute_inversion_error()
            assert_almost_equal(stats[1], 0, decimal=4)
            assert_almost_equal(stats[2], 0, decimal=4)


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
    t = 2.5 #translation factor

    trans = np.array([[1, 0, 0, -t*ns],
                      [0, 1, 0, -t*nr],
                      [0, 0, 1, -t*nc],
                      [0, 0, 0, 1]])
    trans_inv = np.linalg.inv(trans)

    d, dinv = vfu.create_harmonic_fields_3d(ns, nr, nc, 0.2, 8)
    d = np.asarray(d).astype(floating)
    dinv = np.asarray(dinv).astype(floating)

    for theta in [-1 * np.pi/5.0, 0.0, np.pi/5.0]: #rotation angle
        for s in [0.5,  1.0, 2.0]: #scale
            rot = np.zeros(shape=(4,4))
            rot[:3, :3] = geometry.rodrigues_axis_rotation(axis, theta)
            rot[3,3] = 1.0
            scale = np.array([[1*s, 0, 0, 0],
                              [0, 1*s, 0, 0],
                              [0, 0, 1*s, 0],
                              [0, 0, 0, 1]])

            gt_affine = trans_inv.dot(scale.dot(rot.dot(trans)))
            gt_affine_inv = np.linalg.inv(gt_affine)
            dcopy = np.copy(d)

            #make sure the field remains invertible after the re-mapping
            vfu.reorient_vector_field_3d(dcopy, gt_affine)

            # Note: the spacings are used just to check convergence, so they don't need
            # to be very accurate. Here we are passing (0.5 * s) to force the algorithm
            # to make more iterations: in ANTS, there is a hard-coded bound on the maximum
            # residual, that's why we cannot force more iteration by changing the parameters.
            # We will investigate this issue with more detail in the future.

            inv_approx = vfu.invert_vector_field_fixed_point_3d(dcopy, gt_affine_inv,
                                                                np.array([s, s, s])*0.5,
                                                                40, 1e-7)

            mapping = imwarp.DiffeomorphicMap(3, (nr,nc), gt_affine)
            mapping.forward = dcopy
            mapping.backward = inv_approx
            residual, stats = mapping.compute_inversion_error()
            assert_almost_equal(stats[1], 0, decimal=3)
            assert_almost_equal(stats[2], 0, decimal=3)


def test_resample_vector_field_2d():
    r"""
    Expand a vector field by 2, then subsample by 2, the resulting
    field should be the original one
    """
    domain_shape = np.array((64, 64), dtype = np.int32)
    reduced_shape = np.array((32, 32), dtype = np.int32)
    factors = np.array([0.5, 0.5])
    d, dinv = vfu.create_harmonic_fields_2d(reduced_shape[0], reduced_shape[1], 0.3, 6)
    d = np.array(d, dtype = floating)

    expanded = vfu.resample_displacement_field_2d(d, factors,domain_shape)
    subsampled = expanded[::2, ::2, :]

    assert_array_almost_equal(d, subsampled)


def test_resample_vector_field_3d():
    r"""
    Expand a vector field by 2, then subsample by 2, the resulting
    field should be the original one
    """
    domain_shape = np.array((64, 64, 64), dtype = np.int32)
    reduced_shape = np.array((32, 32, 32), dtype = np.int32)
    factors = np.array([0.5, 0.5, 0.5])
    d, dinv = vfu.create_harmonic_fields_3d(reduced_shape[0], reduced_shape[1], reduced_shape[2], 0.3, 6)
    d = np.array(d, dtype = floating)

    expanded = vfu.resample_displacement_field_3d(d, factors,domain_shape)
    subsampled = expanded[::2, ::2, ::2, :]

    assert_array_almost_equal(d, subsampled)


def test_downsample_scalar_field_2d():
    np.random.seed(8315759)
    size = 32
    for reduce_r in [True, False]:
        nrows = size -1 if reduce_r else size
        for reduce_c in [True, False]:
            ncols = size -1 if reduce_c else size
            image = np.ndarray((size, size), dtype=floating)
            image[...] = np.random.randint(0, 10, np.size(image)).reshape((size, size))

            if reduce_r:
                image[-1, :] = 0
            if reduce_c:
                image[:, -1] = 0

            a = image[::2, ::2]
            b = image[1::2, ::2]
            c = image[::2, 1::2]
            d = image[1::2, 1::2]

            expected = 0.25*(a + b + c + d)

            if reduce_r:
                expected[-1,:]*=2
            if reduce_c:
                expected[:,-1]*=2

            actual = np.array(vfu.downsample_scalar_field_2d(image[:nrows, :ncols]))
            assert_array_almost_equal(expected, actual)


def test_downsample_displacement_field_2d():
    np.random.seed(2115556)
    size = 32
    for reduce_r in [True, False]:
        nrows = size -1 if reduce_r else size
        for reduce_c in [True, False]:
            ncols = size -1 if reduce_c else size
            field = np.ndarray((size, size, 2), dtype=floating)
            field[...] = np.random.randint(0, 10, np.size(field)).reshape((size, size, 2))

            if reduce_r:
                field[-1, :, :] = 0
            if reduce_c:
                field[:, -1, :] = 0

            a = field[::2, ::2, :]
            b = field[1::2, ::2, :]
            c = field[::2, 1::2, :]
            d = field[1::2, 1::2, :]

            expected = 0.25*(a + b + c + d)

            if reduce_r:
                expected[-1, :, :]*=2
            if reduce_c:
                expected[:, -1, :]*=2

            actual = np.array(vfu.downsample_displacement_field_2d(field[:nrows, :ncols, :]))
            assert_array_almost_equal(expected, actual)


def test_downsample_scalar_field_3d():
    np.random.seed(8315759)
    size = 32
    for reduce_s in [True, False]:
        nslices = size -1 if reduce_s else size
        for reduce_r in [True, False]:
            nrows = size -1 if reduce_r else size
            for reduce_c in [True, False]:
                ncols = size -1 if reduce_c else size
                image = np.ndarray((size, size, size), dtype=floating)
                image[...] = np.random.randint(0, 10, np.size(image)).reshape((size, size, size))

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

                expected = 0.125*(a + b + c + d + aa + bb + cc + dd)

                if reduce_s:
                    expected[-1, :, :] *= 2
                if reduce_r:
                    expected[:, -1, :] *= 2
                if reduce_c:
                    expected[:, :, -1] *= 2

                actual = np.array(vfu.downsample_scalar_field_3d(image[:nslices, :nrows, :ncols]))
                assert_array_almost_equal(expected, actual)


def test_downsample_displacement_field_3d():
    np.random.seed(8315759)
    size = 32
    for reduce_s in [True, False]:
        nslices = size -1 if reduce_s else size
        for reduce_r in [True, False]:
            nrows = size -1 if reduce_r else size
            for reduce_c in [True, False]:
                ncols = size -1 if reduce_c else size
                field = np.ndarray((size, size, size, 3), dtype=floating)
                field[...] = np.random.randint(0, 10, np.size(field)).reshape((size, size, size, 3))

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

                expected = 0.125*(a + b + c + d + aa + bb + cc + dd)

                if reduce_s:
                    expected[-1, :, :, :] *= 2
                if reduce_r:
                    expected[:, -1, :, :] *= 2
                if reduce_c:
                    expected[:, :, -1, :] *= 2

                actual = np.array(vfu.downsample_displacement_field_3d(field[:nslices, :nrows, :ncols]))
                assert_array_almost_equal(expected, actual)


def test_reorient_vector_field_2d():
    shape = (16,16)
    d, dinv = vfu.create_harmonic_fields_2d(shape[0], shape[1], 0.2, 4)
    d = np.array(d, dtype = floating)

    #the vector field rotated 90 degrees
    expected = np.ndarray(shape = shape+(2,), dtype = floating)
    expected[...,0] = -1 * d[...,1]
    expected[...,1] =  d[...,0]

    #rotate 45 degrees twice
    c = np.sqrt(0.5)
    affine = np.array([[c, -c],[c, c]])
    vfu.reorient_vector_field_2d(d, affine)
    vfu.reorient_vector_field_2d(d, affine)

    #verify almost equal
    assert_array_almost_equal(d, expected)


def test_reorient_vector_field_3d():
    shape = (16, 16, 16)
    d, dinv = vfu.create_harmonic_fields_3d(shape[0], shape[1], shape[2], 0.2, 4)
    d = np.array(d, dtype = floating)
    dinv = np.array(dinv, dtype = floating)

    #the vector field rotated 90 degrees around the last axis
    expected = np.ndarray(shape = shape+(3,), dtype = floating)
    expected[...,0] = -1 * d[...,1]
    expected[...,1] =  d[...,0]
    expected[...,2] =  d[...,2]

    #rotate 45 degrees twice around the last axis
    c = np.sqrt(0.5)
    affine = np.array([[c, -c, 0],[c, c, 0], [0, 0, 1]])
    vfu.reorient_vector_field_3d(d, affine)
    vfu.reorient_vector_field_3d(d, affine)

    #verify almost equal
    assert_array_almost_equal(d, expected)

    #the vector field rotated 90 degrees around the first axis
    expected[...,0] = dinv[...,0]
    expected[...,1] = -1 * dinv[...,2]
    expected[...,2] =  dinv[...,1]

    #rotate 45 degrees twice around the first axis
    affine = np.array([[1, 0, 0], [0, c, -c], [0, c, c]])
    vfu.reorient_vector_field_3d(dinv, affine)
    vfu.reorient_vector_field_3d(dinv, affine)

    #verify almost equal
    assert_array_almost_equal(dinv, expected)


def test_reorient_random_vector_fields():
    np.random.seed(1134781)
    # Test reorienting vector field
    for n_dims, func in ((2, vfu.reorient_vector_field_2d),
                        (3, vfu.reorient_vector_field_3d)):
        size = [20, 30, 40][:n_dims] + [n_dims]
        arr = np.random.normal(size = size)
        arr_32 = arr.astype(np.float32)
        affine = from_matvec(np.random.normal(size = (n_dims, n_dims)),
                            np.zeros(n_dims))
        func(arr_32, affine)
        assert_almost_equal(arr_32, apply_affine(affine, arr), 6)
        # Reorient reorients without translation
        trans = np.arange(n_dims) + 2
        affine[:-1, -1] = trans
        arr_32 = arr.astype(np.float32)
        func(arr_32, affine)
        assert_almost_equal(arr_32, apply_affine(affine, arr) - trans, 6)


if __name__=='__main__':
    test_random_displacement_field_2d()
    test_random_displacement_field_3d()
    test_harmonic_fields_2d()
    test_harmonic_fields_3d()
    test_circle()
    test_sphere()
    test_interpolate_scalar_2d()
    test_interpolate_scalar_nn_2d()
    test_interpolate_scalar_nn_3d()
    test_interpolate_scalar_3d()
    test_warping_2d()
    test_warping_3d()
    test_affine_warping_2d()
    test_affine_warping_3d()
    test_compose_vector_fields_2d()
    test_compose_vector_fields_3d()
    test_invert_vector_field_2d()
    test_invert_vector_field_3d()
    test_resample_vector_field_2d()
    test_resample_vector_field_3d()
    test_downsample_scalar_field_2d()
    test_downsample_scalar_field_3d()
    test_downsample_displacement_field_2d()
    test_downsample_displacement_field_3d()
    test_reorient_vector_field_2d()
    test_reorient_vector_field_3d()
    test_reorient_random_vector_fields()
