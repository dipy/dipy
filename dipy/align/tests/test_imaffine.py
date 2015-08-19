import numpy as np
import scipy as sp
import nibabel as nib
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           assert_almost_equal,
                           assert_equal)
from dipy.core import geometry as geometry
from dipy.data import get_data
from dipy.viz import regtools as rt
from dipy.align import floating
from dipy.align import vector_fields as vf
from dipy.align import imaffine
from dipy.align.transforms import (Transform,
                                   regtransforms)
from dipy.align.tests.test_parzenhist import (setup_random_transform,
                                              sample_domain_regular)

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
           ('AFFINE', 2): (0.1, .50, np.array([0.99, -0.05, 1.3, 0.05, 0.99, 2.5])),
           ('TRANSLATION', 3): (2.0, None, np.array([2.3, 4.5, 1.7])),
           ('ROTATION', 3): (0.1, 1.0, np.array([0.1, 0.15, -0.11])),
           ('RIGID', 3): (0.1, None, np.array([0.1, 0.15, -0.11, 2.3, 4.5, 1.7])),
           ('SCALING', 3): (0.1, .35, np.array([0.95])),
           ('AFFINE', 3): (0.1, None, np.array([0.99, -0.05,  0.03, 1.3,
                                                0.05,  0.99, -0.10, 2.5,
                                                -0.07, 0.10,  0.99, -1.4]))}


def test_align_centers_of_mass_3d():
    np.random.seed(1246592)
    shape = (64, 64, 64)
    rm = 8
    sp = vf.create_sphere(shape[0]//2, shape[1]//2, shape[2]//2, rm)
    moving = np.zeros(shape)
    # The center of mass will be (16, 16, 16), in image coordinates
    moving[:shape[0]//2, :shape[1]//2, :shape[2]//2] = sp[...]

    rs = 16
    # The center of mass will be (32, 32, 32), in image coordinates
    static = vf.create_sphere(shape[0], shape[1], shape[2], rs)

    # Create arbitrary image-to-space transforms
    axis = np.array([.5, 2.0, 1.5])
    t = 0.15 #translation factor
    trans = np.array([[1, 0, 0, -t*shape[0]],
                      [0, 1, 0, -t*shape[1]],
                      [0, 0, 1, -t*shape[2]],
                      [0, 0, 0, 1]])
    trans_inv = np.linalg.inv(trans)

    for rotation_angle in [-1 * np.pi/6.0, 0.0, np.pi/5.0]:
        for scale_factor in [0.83,  1.3, 2.07]: #scale
            rot = np.zeros(shape=(4,4))
            rot[:3, :3] = geometry.rodrigues_axis_rotation(axis,
                                                           rotation_angle)
            rot[3,3] = 1.0
            scale = np.array([[1 * scale_factor, 0, 0, 0],
                              [0, 1 * scale_factor, 0, 0],
                              [0, 0, 1 * scale_factor, 0],
                              [0, 0, 0, 1]])

            static_grid2world = trans_inv.dot(scale.dot(rot.dot(trans)))
            moving_grid2world = np.linalg.inv(static_grid2world)

            # Expected translation
            c_static = static_grid2world.dot((32, 32, 32, 1))[:3]
            c_moving = moving_grid2world.dot((16, 16, 16, 1))[:3]
            expected = np.eye(4);
            expected[:3, 3] = c_moving - c_static

            # Implementation under test
            actual = imaffine.align_centers_of_mass(static, static_grid2world,
                                                    moving, moving_grid2world)
            assert_array_almost_equal(actual.affine, expected)


def test_align_geometric_centers_3d():
    # Create arbitrary image-to-space transforms
    axis = np.array([.5, 2.0, 1.5])
    t = 0.15 #translation factor

    for theta in [-1 * np.pi/6.0, 0.0, np.pi/5.0]: #rotation angle
        for s in [0.83,  1.3, 2.07]: #scale
            m_shapes = [(256, 256, 128), (255, 255, 127), (64, 127, 142)]
            for shape_moving in m_shapes:
                s_shapes = [(256, 256, 128), (255, 255, 127), (64, 127, 142)]
                for shape_static in s_shapes:
                    moving = np.ndarray(shape=shape_moving)
                    static = np.ndarray(shape=shape_static)
                    trans = np.array([[1, 0, 0, -t*shape_static[0]],
                                      [0, 1, 0, -t*shape_static[1]],
                                      [0, 0, 1, -t*shape_static[2]],
                                      [0, 0, 0, 1]])
                    trans_inv = np.linalg.inv(trans)
                    rot = np.zeros(shape=(4,4))
                    rot[:3, :3] = geometry.rodrigues_axis_rotation(axis, theta)
                    rot[3,3] = 1.0
                    scale = np.array([[1 * s, 0, 0, 0],
                                      [0, 1 * s, 0, 0],
                                      [0, 0, 1 * s, 0],
                                      [0, 0, 0, 1]])

                    static_grid2world = trans_inv.dot(scale.dot(rot.dot(trans)))
                    moving_grid2world = np.linalg.inv(static_grid2world)

                    # Expected translation
                    c_static = np.array(shape_static, dtype = np.float64) * 0.5
                    c_static = tuple(c_static)
                    c_static = static_grid2world.dot(c_static+(1,))[:3]
                    c_moving = np.array(shape_moving, dtype = np.float64) * 0.5
                    c_moving = tuple(c_moving)
                    c_moving = moving_grid2world.dot(c_moving+(1,))[:3]
                    expected = np.eye(4);
                    expected[:3, 3] = c_moving - c_static

                    # Implementation under test
                    actual = imaffine.align_geometric_centers(static,
                        static_grid2world, moving, moving_grid2world)
                    assert_array_almost_equal(actual.affine, expected)


def test_align_origins_3d():
    # Create arbitrary image-to-space transforms
    axis = np.array([.5, 2.0, 1.5])
    t = 0.15 #translation factor

    for theta in [-1 * np.pi/6.0, 0.0, np.pi/5.0]: #rotation angle
        for s in [0.83,  1.3, 2.07]: #scale
            m_shapes = [(256, 256, 128), (255, 255, 127), (64, 127, 142)]
            for shape_moving in m_shapes:
                s_shapes = [(256, 256, 128), (255, 255, 127), (64, 127, 142)]
                for shape_static in s_shapes:
                    moving = np.ndarray(shape=shape_moving)
                    static = np.ndarray(shape=shape_static)
                    trans = np.array([[1, 0, 0, -t*shape_static[0]],
                                      [0, 1, 0, -t*shape_static[1]],
                                      [0, 0, 1, -t*shape_static[2]],
                                      [0, 0, 0, 1]])
                    trans_inv = np.linalg.inv(trans)
                    rot = np.zeros(shape=(4,4))
                    rot[:3, :3] = geometry.rodrigues_axis_rotation(axis, theta)
                    rot[3,3] = 1.0
                    scale = np.array([[1*s, 0, 0, 0],
                                      [0, 1*s, 0, 0],
                                      [0, 0, 1*s, 0],
                                      [0, 0, 0, 1]])

                    static_grid2world = trans_inv.dot(scale.dot(rot.dot(trans)))
                    moving_grid2world = np.linalg.inv(static_grid2world)

                    # Expected translation
                    c_static = static_grid2world[:3, 3]
                    c_moving = moving_grid2world[:3, 3]
                    expected = np.eye(4);
                    expected[:3, 3] = c_moving - c_static

                    # Implementation under test
                    actual = imaffine.align_origins(static, static_grid2world,
                                                    moving, moving_grid2world)
                    assert_array_almost_equal(actual.affine, expected)


def test_affreg_all_transforms():
    # Test affine registration using all transforms with typical settings

    # Make sure dictionary entries are processed in the same order regardless of
    # the platform. Otherwise any random numbers drawn within the loop would make
    # the test non-deterministic even if we fix the seed before the loop.
    # Right now, this test does not draw any samples, but we still sort the entries
    # to prevent future related failures.
    for ttype in sorted(factors):
        dim = ttype[1]
        if dim == 2:
            nslices = 1
        else:
            nslices = 45
        factor = factors[ttype][0]
        sampling_pc = factors[ttype][1]
        transform = regtransforms[ttype]

        static, moving, static_grid2world, moving_grid2world, smask, mmask, T = \
                        setup_random_transform(transform, factor, nslices, 1.0)
        # Sum of absolute differences
        start_sad = np.abs(static - moving).sum()
        metric = imaffine.MutualInformationMetric(32, sampling_pc)
        affreg = imaffine.AffineRegistration(metric,
                                             [10000, 1000, 100],
                                             [3, 1, 0],
                                             [4, 2, 1],
                                             'L-BFGS-B',
                                             None,
                                             options=None)
        x0 = transform.get_identity_parameters()
        affine_map = affreg.optimize(static, moving, transform, x0,
                                     static_grid2world, moving_grid2world)
        transformed = affine_map.transform(moving)
        # Sum of absolute differences
        end_sad = np.abs(static - transformed).sum()
        reduction = 1 - end_sad / start_sad
        print("%s>>%f"%(ttype, reduction))
        assert(reduction > 0.9)


def test_affreg_defaults():
    # Test all default arguments with an arbitrary transform
    # Select an arbitrary transform (all of them are already tested
    # in test_affreg_all_transforms)
    transform_name = 'TRANSLATION'
    dim = 2
    ttype = (transform_name, dim)

    for starting_affine in ['mass', 'voxel-origin', 'centers', None]:
        if dim == 2:
            nslices = 1
        else:
            nslices = 45
        factor = factors[ttype][0]
        sampling_pc = factors[ttype][1]
        transform = regtransforms[ttype]
        id_param = transform.get_identity_parameters()

        static, moving, static_grid2world, moving_grid2world, smask, mmask, T = \
                        setup_random_transform(transform, factor, nslices, 1.0)
        # Sum of absolute differences
        start_sad = np.abs(static - moving).sum()

        metric = None
        x0 = None
        sigmas = None
        scale_factors = None
        level_iters = None
        static_grid2world = None
        moving_grid2world = None
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
                                         starting_affine)
            transformed = affine_map.transform(moving)
            # Sum of absolute differences
            end_sad = np.abs(static - transformed).sum()
            reduction = 1 - end_sad / start_sad
            print("%s>>%f"%(ttype, reduction))
            assert(reduction > 0.9)

            transformed_inv = affine_map.transform_inverse(static)
            # Sum of absolute differences
            end_sad = np.abs(moving - transformed_inv).sum()
            reduction = 1 - end_sad / start_sad
            print("%s>>%f"%(ttype, reduction))
            assert(reduction > 0.9)


def test_mi_gradient():
    np.random.seed(2022966)
    # Test the gradient of mutual information
    h = 1e-5
    # Make sure dictionary entries are processed in the same order regardless of
    # the platform. Otherwise any random numbers drawn within the loop would make
    # the test non-deterministic even if we fix the seed before the loop:
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
        starting_affine = start.param_to_matrix(0.25 * np.random.randn(nrot))
        # Get data (pair of images related to each other by an known transform)
        static, moving, static_g2w, moving_g2w, smask, mmask, M = \
            setup_random_transform(transform, factor, nslices, 2.0)

        # Prepare a MutualInformationMetric instance
        mi_metric = imaffine.MutualInformationMetric(32, sampling_proportion)
        mi_metric.setup(transform, static, moving, starting_affine=starting_affine)
        # Compute the gradient with the implementation under test
        actual = mi_metric.gradient(theta)

        # Compute the gradient using finite-diferences
        n = transform.get_number_of_parameters()
        expected = np.empty(n, dtype=np.float64)

        val0 = mi_metric.distance(theta)
        for i in range(n):
            dtheta = theta.copy()
            dtheta[i] += h
            val1 = mi_metric.distance(dtheta)
            expected[i] = (val1 - val0) / h

        dp = expected.dot(actual)
        enorm = np.linalg.norm(expected)
        anorm = np.linalg.norm(actual)
        nprod = dp / (enorm * anorm)
        assert(nprod >= 0.99)
