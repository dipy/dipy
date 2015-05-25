import numpy as np
from dipy.align import floating
import dipy.align.vector_fields as vf
import dipy.align.imaffine as imaffine
import dipy.core.geometry as geometry
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           assert_almost_equal,
                           assert_equal)
import nibabel as nib
import dipy.align.mattes as mattes
import scipy as sp
from dipy.align.transforms import (Transform,
                                   regtransforms)
from dipy.align.imaffine import *
import dipy.viz.regtools as rt
import dipy.align.imaffine as imaffine
from dipy.data import get_data
from dipy.align.tests.test_mattes import setup_random_transform

# For each transform type, select a transform factor (indicating how large the
# true transform between static and moving images will be) and a sampling
# (either a positive integer less than or equal to 100, or None) indicating
# the percentage (if int) of voxels to be used for estimating the joint PDFs,
# or dense sampling (if None)
factors = {('TRANSLATION', 2): (2.0, 30),
           ('ROTATION', 2): (0.1, None),
           ('RIGID', 2): (0.1, 50),
           ('SCALING', 2): (0.01, None),
           ('AFFINE', 2): (0.1, 40),
           ('TRANSLATION', 3): (2.0, None),
           ('ROTATION', 3): (0.1, 35, 60),
           ('RIGID', 3): (0.1, None),
           ('SCALING', 3): (0.1, 30),
           ('AFFINE', 3): (0.1, None)}

def test_aff_centers_of_mass_3d():
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

    for theta in [-1 * np.pi/6.0, 0.0, np.pi/5.0]: #rotation angle
        for s in [0.83,  1.3, 2.07]: #scale
            rot = np.zeros(shape=(4,4))
            rot[:3, :3] = geometry.rodrigues_axis_rotation(axis, theta)
            rot[3,3] = 1.0
            scale = np.array([[1*s, 0, 0, 0],
                              [0, 1*s, 0, 0],
                              [0, 0, 1*s, 0],
                              [0, 0, 0, 1]])

            static_grid2space = trans_inv.dot(scale.dot(rot.dot(trans)))
            moving_grid2space = np.linalg.inv(static_grid2space)

            # Expected translation
            c_static = static_grid2space.dot((32, 32, 32, 1))[:3]
            c_moving = moving_grid2space.dot((16, 16, 16, 1))[:3]
            expected = np.eye(4);
            expected[:3, 3] = c_moving - c_static

            # Implementation under test
            actual = imaffine.aff_centers_of_mass(static, static_grid2space,
                                                  moving, moving_grid2space)
            assert_array_almost_equal(actual, expected)


def test_aff_geometric_centers_3d():
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

                    static_grid2space = trans_inv.dot(scale.dot(rot.dot(trans)))
                    moving_grid2space = np.linalg.inv(static_grid2space)

                    # Expected translation
                    c_static = np.array(shape_static, dtype = np.float64) * 0.5
                    c_static = tuple(c_static)
                    c_static = static_grid2space.dot(c_static+(1,))[:3]
                    c_moving = np.array(shape_moving, dtype = np.float64) * 0.5
                    c_moving = tuple(c_moving)
                    c_moving = moving_grid2space.dot(c_moving+(1,))[:3]
                    expected = np.eye(4);
                    expected[:3, 3] = c_moving - c_static

                    # Implementation under test
                    actual = imaffine.aff_geometric_centers(static,
                        static_grid2space, moving, moving_grid2space)
                    assert_array_almost_equal(actual, expected)


def test_aff_origins_3d():
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

                    static_grid2space = trans_inv.dot(scale.dot(rot.dot(trans)))
                    moving_grid2space = np.linalg.inv(static_grid2space)

                    # Expected translation
                    c_static = static_grid2space[:3, 3]
                    c_moving = moving_grid2space[:3, 3]
                    expected = np.eye(4);
                    expected[:3, 3] = c_moving - c_static

                    # Implementation under test
                    actual = imaffine.aff_origins(static, static_grid2space,
                                                  moving, moving_grid2space)
                    assert_array_almost_equal(actual, expected)


def test_affreg_all_transforms():
    # Test affine registration using all transforms with typical settings
    for ttype in factors.keys():
        dim = ttype[1]
        if dim == 2:
            nslices = 1
        else:
            nslices = 45
        factor = factors[ttype][0]
        sampling_pc = factors[ttype][1]
        transform = regtransforms[ttype]

        static, moving, static_grid2space, moving_grid2space, smask, mmask, T = \
                        setup_random_transform(transform, factor, nslices, 1.0)
        # Sum of absolute differences
        start_sad = np.abs(static - moving).sum()
        metric = imaffine.MattesMIMetric(32, sampling_pc)
        affreg = imaffine.AffineRegistration(metric,
                                             [10000, 1000, 100], 1e-5,
                                             [3, 1, 0],
                                             [4, 2, 1],
                                             'L-BFGS-B',
                                             None,
                                             options=None)
        x0 = transform.get_identity_parameters()
        sol = affreg.optimize(static, moving, transform, x0, static_grid2space,
                              moving_grid2space)
        warped = aff_warp(static, static_grid2space, moving,
                          moving_grid2space, sol)
        # Sum of absolute differences
        end_sad = np.abs(static - warped).sum()
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

    for prealign in ['mass', 'origins', 'centers', None]:
        if dim == 2:
            nslices = 1
        else:
            nslices = 45
        factor = factors[ttype][0]
        sampling_pc = factors[ttype][1]
        transform = regtransforms[ttype]
        id_param = transform.get_identity_parameters()

        static, moving, static_grid2space, moving_grid2space, smask, mmask, T = \
                        setup_random_transform(transform, factor, nslices, 1.0)
        # Sum of absolute differences
        start_sad = np.abs(static - moving).sum()

        metric = None
        x0 = None
        sigmas = None
        scale_factors = None
        level_iters = None
        static_grid2space = None
        moving_grid2space = None
        for ss_sigma_factor in [1.0, None]:
            affreg = imaffine.AffineRegistration(metric,
                                                 level_iters, 1e-5,
                                                 sigmas,
                                                 scale_factors,
                                                 'L-BFGS-B',
                                                 ss_sigma_factor,
                                                 options=None)
            sol = affreg.optimize(static, moving, transform, x0, static_grid2space,
                                  moving_grid2space, prealign)
            warped = aff_warp(static, static_grid2space, moving,
                              moving_grid2space, sol)
            # Sum of absolute differences
            end_sad = np.abs(static - warped).sum()
            reduction = 1 - end_sad / start_sad
            print("%s>>%f"%(ttype, reduction))
            assert(reduction > 0.9)