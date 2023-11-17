import numpy as np
import numpy.linalg as npl
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           assert_equal,
                           assert_raises,
                           assert_warns)
from dipy.core import geometry as geometry
from dipy.align import vector_fields as vf
from dipy.align import imaffine
from dipy.align.imaffine import AffineInversionError, AffineInvalidValuesError, \
    AffineMap, _number_dim_affine_matrix
from dipy.align.transforms import regtransforms
from dipy.align.tests.test_parzenhist import setup_random_transform
from dipy.testing.decorators import set_random_number_generator

# For each transform type, select a transform factor (indicating how large the
# true transform between static and moving images will be), a sampling scheme
# (either a positive integer less than or equal to 100, or None) indicating
# the percentage (if int) of voxels to be used for estimating the joint PDFs,
# or dense sampling (if None), and also specify a starting point (to avoid
# starting from the identity)
factors = {('TRANSLATION', 2): (2.0, 0.35, np.array([2.3, 4.5])),
           ('ROTATION', 2): (0.1, None, np.array([0.1])),
           ('RIGID', 2): (0.1, .50, np.array([0.12, 1.8, 2.7])),
           ('SCALING', 2): (0.01, None, np.array([1.05])),
           ('AFFINE', 2): (0.1, .50, np.array([0.99, -0.05, 1.3, 0.05, 0.99,
                                               2.5])),
           ('TRANSLATION', 3): (2.0, None, np.array([2.3, 4.5, 1.7])),
           ('ROTATION', 3): (0.1, 1.0, np.array([0.1, 0.15, -0.11])),
           ('RIGID', 3): (0.1, None, np.array([0.1, 0.15, -0.11, 2.3, 4.5,
                                               1.7])),
           ('SCALING', 3): (0.1, .35, np.array([0.95])),
           ('AFFINE', 3): (0.1, None, np.array([0.99, -0.05, 0.03, 1.3,
                                                0.05, 0.99, -0.10, 2.5,
                                                -0.07, 0.10, 0.99, -1.4]))}


def test_transform_centers_of_mass_3d():
    shape = (64, 64, 64)
    rm = 8
    sph = vf.create_sphere(shape[0] // 2, shape[1] // 2, shape[2] // 2, rm)
    moving = np.zeros(shape)
    # The center of mass will be (16, 16, 16), in image coordinates
    moving[:shape[0] // 2, :shape[1] // 2, :shape[2] // 2] = sph[...]

    rs = 16
    # The center of mass will be (32, 32, 32), in image coordinates
    static = vf.create_sphere(shape[0], shape[1], shape[2], rs)

    # Create arbitrary image-to-space transforms
    axis = np.array([.5, 2.0, 1.5])
    t = 0.15  # translation factor
    trans = np.array([[1, 0, 0, -t * shape[0]],
                      [0, 1, 0, -t * shape[1]],
                      [0, 0, 1, -t * shape[2]],
                      [0, 0, 0, 1]])
    trans_inv = npl.inv(trans)

    for rotation_angle in [-1 * np.pi / 6.0, 0.0, np.pi / 5.0]:
        for scale_factor in [0.83, 1.3, 2.07]:  # scale
            rot = np.zeros(shape=(4, 4))
            rot[:3, :3] = geometry.rodrigues_axis_rotation(axis,
                                                           rotation_angle)
            rot[3, 3] = 1.0
            scale = np.array([[1 * scale_factor, 0, 0, 0],
                              [0, 1 * scale_factor, 0, 0],
                              [0, 0, 1 * scale_factor, 0],
                              [0, 0, 0, 1]])

            static_grid2world = trans_inv.dot(scale.dot(rot.dot(trans)))
            moving_grid2world = npl.inv(static_grid2world)

            # Expected translation
            c_static = static_grid2world.dot((32, 32, 32, 1))[:3]
            c_moving = moving_grid2world.dot((16, 16, 16, 1))[:3]
            expected = np.eye(4)
            expected[:3, 3] = c_moving - c_static

            # Implementation under test
            actual = imaffine.transform_centers_of_mass(static,
                                                        static_grid2world,
                                                        moving,
                                                        moving_grid2world)
            assert_array_almost_equal(actual.affine, expected)


def test_transform_geometric_centers_3d():
    # Create arbitrary image-to-space transforms
    axis = np.array([.5, 2.0, 1.5])
    t = 0.15  # translation factor

    for theta in [-1 * np.pi / 6.0, 0.0, np.pi / 5.0]:  # rotation angle
        for s in [0.83, 1.3, 2.07]:  # scale
            m_shapes = [(256, 256, 128), (255, 255, 127), (64, 127, 142)]
            for shape_moving in m_shapes:
                s_shapes = [(256, 256, 128), (255, 255, 127), (64, 127, 142)]
                for shape_static in s_shapes:
                    moving = np.ndarray(shape=shape_moving)
                    static = np.ndarray(shape=shape_static)
                    trans = np.array([[1, 0, 0, -t * shape_static[0]],
                                      [0, 1, 0, -t * shape_static[1]],
                                      [0, 0, 1, -t * shape_static[2]],
                                      [0, 0, 0, 1]])
                    trans_inv = npl.inv(trans)
                    rot = np.zeros(shape=(4, 4))
                    rot[:3, :3] = geometry.rodrigues_axis_rotation(axis, theta)
                    rot[3, 3] = 1.0
                    scale = np.array([[1 * s, 0, 0, 0],
                                      [0, 1 * s, 0, 0],
                                      [0, 0, 1 * s, 0],
                                      [0, 0, 0, 1]])

                    static_grid2world = trans_inv.dot(
                        scale.dot(rot.dot(trans)))
                    moving_grid2world = npl.inv(static_grid2world)

                    # Expected translation
                    c_static = np.array(shape_static, dtype=np.float64) * 0.5
                    c_static = tuple(c_static)
                    c_static = static_grid2world.dot(c_static + (1,))[:3]
                    c_moving = np.array(shape_moving, dtype=np.float64) * 0.5
                    c_moving = tuple(c_moving)
                    c_moving = moving_grid2world.dot(c_moving + (1,))[:3]
                    expected = np.eye(4)
                    expected[:3, 3] = c_moving - c_static

                    # Implementation under test
                    actual = imaffine.transform_geometric_centers(
                        static, static_grid2world, moving, moving_grid2world)
                    assert_array_almost_equal(actual.affine, expected)


def test_transform_origins_3d():
    # Create arbitrary image-to-space transforms
    axis = np.array([.5, 2.0, 1.5])
    t = 0.15  # translation factor

    for theta in [-1 * np.pi / 6.0, 0.0, np.pi / 5.0]:  # rotation angle
        for s in [0.83, 1.3, 2.07]:  # scale
            m_shapes = [(256, 256, 128), (255, 255, 127), (64, 127, 142)]
            for shape_moving in m_shapes:
                s_shapes = [(256, 256, 128), (255, 255, 127), (64, 127, 142)]
                for shape_static in s_shapes:
                    moving = np.ndarray(shape=shape_moving)
                    static = np.ndarray(shape=shape_static)
                    trans = np.array([[1, 0, 0, -t * shape_static[0]],
                                      [0, 1, 0, -t * shape_static[1]],
                                      [0, 0, 1, -t * shape_static[2]],
                                      [0, 0, 0, 1]])
                    trans_inv = npl.inv(trans)
                    rot = np.zeros(shape=(4, 4))
                    rot[:3, :3] = geometry.rodrigues_axis_rotation(axis, theta)
                    rot[3, 3] = 1.0
                    scale = np.array([[1 * s, 0, 0, 0],
                                      [0, 1 * s, 0, 0],
                                      [0, 0, 1 * s, 0],
                                      [0, 0, 0, 1]])

                    static_grid2world = trans_inv.dot(
                        scale.dot(rot.dot(trans)))
                    moving_grid2world = npl.inv(static_grid2world)

                    # Expected translation
                    c_static = static_grid2world[:3, 3]
                    c_moving = moving_grid2world[:3, 3]
                    expected = np.eye(4)
                    expected[:3, 3] = c_moving - c_static

                    # Implementation under test
                    actual = imaffine.transform_origins(static,
                                                        static_grid2world,
                                                        moving,
                                                        moving_grid2world)
                    assert_array_almost_equal(actual.affine, expected)


@set_random_number_generator(202311)
def test_affreg_all_transforms(rng):
    # Test affine registration using all transforms with typical settings

    # Make sure dictionary entries are processed in the same order regardless
    # of the platform. Otherwise any random numbers drawn within the loop would
    # make the test non-deterministic even if we fix the seed before the loop.
    # Right now, this test does not draw any samples, but we still sort the
    # entries to prevent future related failures.
    for ttype in sorted(factors):
        dim = ttype[1]
        if dim == 2:
            nslices = 1
        else:
            nslices = 45
        factor = factors[ttype][0]
        sampling_pc = factors[ttype][1]
        trans = regtransforms[ttype]
        # Shorthand:
        srt = setup_random_transform
        static, moving, static_g2w, moving_g2w, smask, mmask, T = srt(
                                                                      trans,
                                                                      factor,
                                                                      nslices,
                                                                      1.0,
                                                                      rng=rng)
        # Sum of absolute differences
        start_sad = np.abs(static - moving).sum()
        metric = imaffine.MutualInformationMetric(32, sampling_pc)
        affreg = imaffine.AffineRegistration(metric,
                                             [1000, 100, 50],
                                             [3, 1, 0],
                                             [4, 2, 1],
                                             'L-BFGS-B',
                                             None,
                                             options=None)
        x0 = trans.get_identity_parameters()

        # test warning for using masks (even if all ones) with sparse sampling
        if sampling_pc not in [1.0, None]:
            affine_map = assert_warns(UserWarning, affreg.optimize,
                                      static, moving, trans, x0,
                                      static_g2w, moving_g2w,
                                      None, None,
                                      smask, mmask)
        else:
            affine_map = affreg.optimize(static, moving, trans, x0,
                                         static_g2w, moving_g2w,
                                         None, None,
                                         smask, mmask)

        transformed = affine_map.transform(moving)
        # Sum of absolute differences
        end_sad = np.abs(static - transformed).sum()
        reduction = 1 - end_sad / start_sad
        print("%s>>%f" % (ttype, reduction))
        assert(reduction > 0.9)

    # Verify that exception is raised if level_iters is empty
    metric = imaffine.MutualInformationMetric(32)
    assert_raises(ValueError, imaffine.AffineRegistration, metric, [])

    # Verify that exception is raised if masks are all zeros
    affine_map = assert_warns(UserWarning, affreg.optimize,
                              static, moving, trans, x0,
                              static_g2w, moving_g2w,
                              None, None,
                              np.zeros_like(smask), np.zeros_like(mmask))


@set_random_number_generator(202311)
def test_affreg_defaults(rng):
    # Test all default arguments with an arbitrary transform
    # Select an arbitrary transform (all of them are already tested
    # in test_affreg_all_transforms)
    transform_name = 'TRANSLATION'
    dim = 2
    ttype = (transform_name, dim)
    aff_options = ['mass', 'voxel-origin', 'centers', None, np.eye(dim + 1)]

    for starting_affine in aff_options:
        if dim == 2:
            nslices = 1
        else:
            nslices = 45
        factor = factors[ttype][0]
        transform = regtransforms[ttype]
        static, moving, static_grid2world, moving_grid2world, smask, mmask, T = \
            setup_random_transform(transform, factor, nslices, 1.0, rng=rng)
        # Sum of absolute differences
        start_sad = np.abs(static - moving).sum()

        metric = None
        x0 = None
        sigmas = None
        scale_factors = None
        level_iters = None
        static_grid2world = None
        moving_grid2world = None
        smask = None
        mmask = None
        for ss_sigma_factor in [1.0, None]:
            affreg = imaffine.AffineRegistration(metric,
                                                 level_iters,
                                                 sigmas,
                                                 scale_factors,
                                                 'L-BFGS-B',
                                                 ss_sigma_factor,
                                                 options=None)
            affine_map = affreg.optimize(static, moving, transform, x0,
                                         static_grid2world, moving_grid2world,
                                         starting_affine, None,
                                         smask, mmask)
            transformed = affine_map.transform(moving)
            # Sum of absolute differences
            end_sad = np.abs(static - transformed).sum()
            reduction = 1 - end_sad / start_sad
            print("%s>>%f" % (ttype, reduction))
            assert(reduction > 0.9)

            transformed_inv = affine_map.transform_inverse(static)
            # Sum of absolute differences
            end_sad = np.abs(moving - transformed_inv).sum()
            reduction = 1 - end_sad / start_sad
            print("%s>>%f" % (ttype, reduction))
            assert(reduction > 0.89)


@set_random_number_generator(2022966)
def test_mi_gradient(rng):
    # Test the gradient of mutual information
    h = 1e-5
    # Make sure dictionary entries are processed in the same order regardless
    # of the platform. Otherwise any random numbers drawn within the loop would
    # make the test non-deterministic even if we fix the seed before the loop:
    # in this case the samples are drawn with `np.random.randn` below

    for ttype in sorted(factors):
        transform = regtransforms[ttype]
        dim = ttype[1]
        if dim == 2:
            nslices = 1
        else:
            nslices = 45
        factor = factors[ttype][0]
        sampling_proportion = factors[ttype][1]
        theta = factors[ttype][2]
        # Start from a small rotation
        start = regtransforms[('ROTATION', dim)]
        nrot = start.get_number_of_parameters()
        starting_affine = \
            start.param_to_matrix(0.25 * rng.standard_normal(nrot))
        # Get data (pair of images related to each other by an known transform)
        static, moving, static_g2w, moving_g2w, smask, mmask, M = \
            setup_random_transform(transform, factor, nslices, 2.0, rng=rng)

        # Prepare a MutualInformationMetric instance
        mi_metric = imaffine.MutualInformationMetric(32, sampling_proportion)
        mi_metric.setup(
            transform,
            static,
            moving,
            starting_affine=starting_affine)
        # Compute the gradient with the implementation under test
        actual = mi_metric.gradient(theta)

        # Compute the gradient using finite-differences
        n = transform.get_number_of_parameters()
        expected = np.empty(n, dtype=np.float64)

        val0 = mi_metric.distance(theta)
        for i in range(n):
            dtheta = theta.copy()
            dtheta[i] += h
            val1 = mi_metric.distance(dtheta)
            expected[i] = (val1 - val0) / h

        dp = expected.dot(actual)
        enorm = npl.norm(expected)
        anorm = npl.norm(actual)
        nprod = dp / (enorm * anorm)
        assert(nprod >= 0.99)


def create_affine_transforms(
        dim, translations, rotations, scales, rot_axis=None):
    r""" Creates a list of affine transforms with all combinations of params

    This function is intended to be used for testing only. It generates
    affine transforms for all combinations of the input parameters in the
    following order: let T be a translation, R a rotation and S a scale. The
    generated affine will be:

    A = T.dot(S).dot(R).dot(T^{-1})

    Translation is handled this way because it is convenient to provide
    the translation parameters in terms of the center of rotation we wish
    to generate.

    Parameters
    ----------
    dim: int (either dim=2 or dim=3)
        dimension of the affine transforms
    translations: sequence of dim-tuples
        each dim-tuple represents a translation parameter
    rotations: sequence of floats
        each number represents a rotation angle in radians
    scales: sequence of floats
        each number represents a scale
    rot_axis: rotation axis (used for dim=3 only)

    Returns
    -------
    transforms: sequence of (dim + 1)x(dim + 1) matrices
        each matrix correspond to an affine transform with a combination
        of the input parameters
    """
    transforms = []
    for t in translations:
        trans_inv = np.eye(dim + 1)
        trans_inv[:dim, dim] = -t[:dim]
        trans = npl.inv(trans_inv)
        for theta in rotations:  # rotation angle
            if dim == 2:
                ct = np.cos(theta)
                st = np.sin(theta)
                rot = np.array([[ct, -st, 0],
                                [st, ct, 0],
                                [0, 0, 1]])
            else:
                rot = np.eye(dim + 1)
                rot[:3, :3] = geometry.rodrigues_axis_rotation(rot_axis, theta)

            for s in scales:  # scale
                scale = np.eye(dim + 1) * s
                scale[dim, dim] = 1

            affine = trans.dot(scale.dot(rot.dot(trans_inv)))
            transforms.append(affine)
    return transforms


@set_random_number_generator(2112927)
def test_affine_map(rng):
    dom_shape = np.array([64, 64, 64], dtype=np.int32)
    cod_shape = np.array([80, 80, 80], dtype=np.int32)
    # Radius of the circle/sphere (testing image)
    radius = 16
    # Rotation axis (used for 3D transforms only)
    rot_axis = np.array([.5, 2.0, 1.5])
    # Arbitrary transform parameters
    t = 0.15
    rotations = [-1 * np.pi / 10.0, 0.0, np.pi / 10.0]
    scales = [0.9, 1.0, 1.1]
    for dim1 in [2, 3]:
        # Setup current dimension
        if dim1 == 2:
            # Create image of a circle
            img = vf.create_circle(cod_shape[0], cod_shape[1], radius)
            oracle_linear = vf.transform_2d_affine
            oracle_nn = vf.transform_2d_affine_nn
        else:
            # Create image of a sphere
            img = vf.create_sphere(cod_shape[0], cod_shape[1], cod_shape[2],
                                   radius)
            oracle_linear = vf.transform_3d_affine
            oracle_nn = vf.transform_3d_affine_nn
        img = np.array(img)
        # Translation is the only parameter differing for 2D and 3D
        translations = [t * dom_shape[:dim1]]
        # Generate affine transforms
        gt_affines = create_affine_transforms(dim1, translations, rotations,
                                              scales, rot_axis)
        # Include the None case
        gt_affines.append(None)

        # testing str/format/repr
        for affine_mat in gt_affines:
            aff_map = AffineMap(affine_mat)
            assert_equal(str(aff_map), aff_map.__str__())
            assert_equal(repr(aff_map), aff_map.__repr__())
            for spec in ['f', 'r', 't', '']:
                assert_equal(format(aff_map, spec), aff_map.__format__(spec))

        for affine in gt_affines:

            # make both domain point to the same physical region
            # It's ok to use the same transform, we just want to test
            # that this information is actually being considered
            domain_grid2world = affine
            codomain_grid2world = affine
            grid2grid_transform = affine

            # Evaluate the transform with vector_fields module (already tested)
            expected_linear = oracle_linear(img, dom_shape[:dim1],
                                            grid2grid_transform)
            expected_nn = oracle_nn(img, dom_shape[:dim1], grid2grid_transform)

            # Evaluate the transform with the implementation under test
            affine_map = imaffine.AffineMap(affine,
                                            dom_shape[:dim1],
                                            domain_grid2world,
                                            cod_shape[:dim1],
                                            codomain_grid2world)
            actual_linear = affine_map.transform(img, interpolation='linear')
            actual_nn = affine_map.transform(img, interpolation='nearest')
            assert_array_almost_equal(actual_linear, expected_linear)
            assert_array_almost_equal(actual_nn, expected_nn)

            # Test set_affine with valid matrix
            affine_map.set_affine(affine)
            if affine is None:
                assert(affine_map.affine is None)
                assert(affine_map.affine_inv is None)
            else:
                # compatibility with previous versions
                assert_array_equal(affine, affine_map.affine)
                # new getter
                new_copy_affine = affine_map.affine
                # value must be the same
                assert_array_equal(affine, new_copy_affine)
                # but not its reference
                assert id(affine) != id(new_copy_affine)
                actual = affine_map.affine.dot(affine_map.affine_inv)
                assert_array_almost_equal(actual, np.eye(dim1 + 1))

            # Evaluate via the inverse transform

            # AffineMap will use the inverse of the input matrix when we call
            # `transform_inverse`. Since the inverse of the inverse of a matrix
            # is not exactly equal to the original matrix (numerical
            #  limitations) we need to invert the matrix twice to make sure
            # the oracle and the implementation under test apply the same
            # transform
            aff_inv = None if affine is None else npl.inv(affine)
            aff_inv_inv = None if aff_inv is None else npl.inv(aff_inv)
            expected_linear = oracle_linear(img, dom_shape[:dim1],
                                            aff_inv_inv)
            expected_nn = oracle_nn(img, dom_shape[:dim1], aff_inv_inv)

            affine_map = imaffine.AffineMap(aff_inv,
                                            cod_shape[:dim1],
                                            codomain_grid2world,
                                            dom_shape[:dim1],
                                            domain_grid2world)
            actual_linear = affine_map.transform_inverse(
                img, interpolation='linear')
            actual_nn = affine_map.transform_inverse(img,
                                                     interpolation='nearest')
            assert_array_almost_equal(actual_linear, expected_linear)
            assert_array_almost_equal(actual_nn, expected_nn)

        # Verify AffineMap can not be created with non-square matrix
        non_square_shapes = [np.zeros((dim1, dim1 + 1), dtype=np.float64),
                             np.zeros((dim1 + 1, dim1), dtype=np.float64)]
        for nsq in non_square_shapes:
            assert_raises(AffineInversionError, AffineMap, nsq)

        # Verify incorrect augmentations are caught
        for affine_mat in gt_affines:
            aff_map = AffineMap(affine_mat)
            if affine_mat is None:
                continue
            bad_aug = aff_map.affine
            # no zeros in the first n-1 columns on last row
            bad_aug[-1, :] = 1
            assert_raises(AffineInvalidValuesError, AffineMap, bad_aug)

            bad_aug = aff_map.affine
            bad_aug[-1, -1] = 0  # lower right not 1
            assert_raises(AffineInvalidValuesError, AffineMap, bad_aug)

        # Verify AffineMap cannot be created with a non-invertible matrix
        invalid_nan = np.zeros((dim1 + 1, dim1 + 1), dtype=np.float64)
        invalid_nan[1, 1] = np.nan
        invalid_zeros = np.zeros((dim1 + 1, dim1 + 1), dtype=np.float64)
        assert_raises(
            imaffine.AffineInvalidValuesError,
            imaffine.AffineMap,
            invalid_nan)
        assert_raises(
            AffineInvalidValuesError,
            imaffine.AffineMap,
            invalid_zeros)

        # Test exception is raised when the affine transform matrix is not
        # valid
        invalid_shape = np.eye(dim1)
        affmap_invalid_shape = imaffine.AffineMap(invalid_shape,
                                                  dom_shape[:dim1], None,
                                                  cod_shape[:dim1], None)
        assert_raises(ValueError, affmap_invalid_shape.transform, img)
        assert_raises(ValueError, affmap_invalid_shape.transform_inverse, img)

        # Verify exception is raised when sampling info is not provided
        valid = np.eye(3)
        affmap_invalid_shape = imaffine.AffineMap(valid)
        assert_raises(ValueError, affmap_invalid_shape.transform, img)
        assert_raises(ValueError, affmap_invalid_shape.transform_inverse, img)

        # Verify exception is raised when requesting an invalid interpolation
        assert_raises(ValueError, affine_map.transform, img, 'invalid')
        assert_raises(ValueError, affine_map.transform_inverse, img, 'invalid')

        # Verify exception is raised when attempting to warp an image of
        # invalid dimension
        for dim2 in [2, 3]:
            affine_map = imaffine.AffineMap(np.eye(dim2),
                                            cod_shape[:dim2], None,
                                            dom_shape[:dim2], None)
            for sh in [(2,), (2, 2, 2, 2)]:
                img = np.zeros(sh)
                assert_raises(ValueError, affine_map.transform, img)
                assert_raises(ValueError, affine_map.transform_inverse, img)
            aff_sing = np.zeros((dim2 + 1, dim2 + 1))
            aff_nan = np.zeros((dim2 + 1, dim2 + 1))
            aff_nan[...] = np.nan
            aff_inf = np.zeros((dim2 + 1, dim2 + 1))
            aff_inf[...] = np.inf

            assert_raises(
                AffineInvalidValuesError,
                affine_map.set_affine,
                aff_sing)
            assert_raises(AffineInvalidValuesError, affine_map.set_affine,
                          aff_nan)
            assert_raises(AffineInvalidValuesError, affine_map.set_affine,
                          aff_inf)

    # Verify AffineMap can not be created with non-2D matrices : len(shape) != 2
    for dim_not_2 in range(10):
        if dim_not_2 != _number_dim_affine_matrix:
            mat_large_dim = rng.random([2]*dim_not_2)
            assert_raises(AffineInversionError, AffineMap, mat_large_dim)


@set_random_number_generator()
def test_MIMetric_invalid_params(rng):
    transform = regtransforms[('AFFINE', 3)]
    static = rng.random((20, 20, 20))
    moving = rng.random((20, 20, 20))
    n = transform.get_number_of_parameters()
    sampling_proportion = 0.3
    theta_sing = np.zeros(n)
    theta_nan = np.zeros(n)
    theta_nan[...] = np.nan
    theta_inf = np.zeros(n)
    theta_nan[...] = np.inf

    mi_metric = imaffine.MutualInformationMetric(32, sampling_proportion)
    mi_metric.setup(transform, static, moving)
    for theta in [theta_sing, theta_nan, theta_inf]:
        # Test metric value at invalid params
        actual_val = mi_metric.distance(theta)
        assert(np.isinf(actual_val))

        # Test gradient at invalid params
        expected_grad = np.zeros(n)
        actual_grad = mi_metric.gradient(theta)
        assert_equal(actual_grad, expected_grad)

        # Test both
        actual_val, actual_grad = mi_metric.distance_and_gradient(theta)
        assert(np.isinf(actual_val))
        assert_equal(actual_grad, expected_grad)
