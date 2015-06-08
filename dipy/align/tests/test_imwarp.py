from __future__ import print_function
import numpy as np
import numpy.testing as npt
import nibabel.eulerangles as eulerangles
from numpy.testing import (assert_equal,
                           assert_array_equal,
                           assert_array_almost_equal,
                           assert_raises)
from ...__config__ import USING_VC_SSE2, USING_GCC_SSE2
from ...data import get_data
from .. import floating
from .. import imwarp as imwarp
from .. import metrics as metrics
from .. import vector_fields as vfu
from .. import VerbosityLevels
from ..imwarp import DiffeomorphicMap

NO_SSE2 = not (USING_VC_SSE2 or USING_GCC_SSE2)

def test_mult_aff():
    r"""mult_aff from imwarp returns the matrix product A.dot(B) considering
    None as the identity
    """
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[2.0, 0.0], [0.0, 2.0]])

    C = imwarp.mult_aff(A, B)
    expected_mult = np.array([[2.0, 4.0], [6.0, 8.0]])
    assert_array_almost_equal(C, expected_mult)

    C = imwarp.mult_aff(A, None)
    assert_array_almost_equal(C, A)

    C = imwarp.mult_aff(None, B)
    assert_array_almost_equal(C, B)

    C = imwarp.mult_aff(None, None)
    assert_equal(C, None)


def test_diffeomorphic_map_2d():
    r"""
    Creates a random displacement field that exactly maps pixels from an input
    image to an output image. First a discrete random assignment between the
    images is generated, then each pair of mapped points are transformed to
    the physical space by assigning a pair of arbitrary, fixed affine matrices
    to input and output images, and finaly the difference between their
    positions is taken as the displacement vector. The resulting displacement,
    although operating in physical space, maps the points exactly (up to
    numerical precision).
    """
    np.random.seed(2022966)
    domain_shape = (10, 10)
    codomain_shape = (10, 10)
    #create a simple affine transformation
    nr = domain_shape[0]
    nc = domain_shape[1]
    s = 1.1
    t = 0.25
    trans = np.array([[1, 0, -t*nr],
                      [0, 1, -t*nc],
                      [0, 0, 1]])
    trans_inv = np.linalg.inv(trans)
    scale = np.array([[1*s, 0, 0],
                      [0, 1*s, 0],
                      [0, 0, 1]])
    gt_affine = trans_inv.dot(scale.dot(trans))

    #create the random displacement field
    domain_grid2world = gt_affine
    codomain_grid2world = gt_affine
    disp, assign = vfu.create_random_displacement_2d(
                        np.array(domain_shape, dtype=np.int32),
                        domain_grid2world,np.array(codomain_shape, dtype=np.int32),
                        codomain_grid2world)
    disp = np.array(disp, dtype=floating)
    assign = np.array(assign)
    #create a random image (with decimal digits) to warp
    moving_image = np.ndarray(codomain_shape, dtype=floating)
    ns = np.size(moving_image)
    moving_image[...] = np.random.randint(0, 10, ns).reshape(codomain_shape)
    #set boundary values to zero so we don't test wrong interpolation due
    #to floating point precision
    moving_image[0,:] = 0
    moving_image[-1,:] = 0
    moving_image[:,0] = 0
    moving_image[:,-1] = 0

    #warp the moving image using the (exact) assignments
    expected = moving_image[(assign[...,0], assign[...,1])]

    #warp using a DiffeomorphicMap instance
    diff_map = imwarp.DiffeomorphicMap(2, domain_shape, domain_grid2world,
                                          domain_shape, domain_grid2world,
                                          codomain_shape, codomain_grid2world,
                                          None)
    diff_map.forward = disp

    #Verify that the transform method accepts different image types (note that
    #the actual image contained integer values, we don't want to test rounding)
    for type in [floating, np.float64, np.int64, np.int32]:
        moving_image = moving_image.astype(type)

        #warp using linear interpolation
        warped = diff_map.transform(moving_image, 'linear')
        #compare the images (the linear interpolation may introduce slight
        #precision errors)
        assert_array_almost_equal(warped, expected, decimal=5)

        #Now test the nearest neighbor interpolation
        warped = diff_map.transform(moving_image, 'nearest')
        #compare the images (now we dont have to worry about precision,
        #it is n.n.)
        assert_array_almost_equal(warped, expected)

        #verify the is_inverse flag
        inv = diff_map.inverse()
        warped = inv.transform_inverse(moving_image, 'linear')
        assert_array_almost_equal(warped, expected, decimal=5)

        warped = inv.transform_inverse(moving_image, 'nearest')
        assert_array_almost_equal(warped, expected)

    #Now test the inverse functionality
    diff_map = imwarp.DiffeomorphicMap(2, codomain_shape, codomain_grid2world,
                                          codomain_shape, codomain_grid2world,
                                          domain_shape, domain_grid2world, None)
    diff_map.backward = disp
    for type in [floating, np.float64, np.int64, np.int32]:
        moving_image = moving_image.astype(type)

        #warp using linear interpolation
        warped = diff_map.transform_inverse(moving_image, 'linear')
        #compare the images (the linear interpolation may introduce slight
        #precision errors)
        assert_array_almost_equal(warped, expected, decimal=5)

        #Now test the nearest neighbor interpolation
        warped = diff_map.transform_inverse(moving_image, 'nearest')
        #compare the images (now we don't have to worry about precision,
        #it is nearest neighbour)
        assert_array_almost_equal(warped, expected)

    #Verify that DiffeomorphicMap raises the appropriate exceptions when
    #the sampling information is undefined
    diff_map = imwarp.DiffeomorphicMap(2, domain_shape, domain_grid2world,
                                          domain_shape, domain_grid2world,
                                          codomain_shape, codomain_grid2world,
                                          None)
    diff_map.forward = disp
    diff_map.domain_shape = None
    #If we don't provide the sampling info, it should try to use the map's
    #info, but it's None...
    assert_raises(ValueError, diff_map.transform, moving_image, 'linear')

    #Same test for diff_map.transform_inverse
    diff_map = imwarp.DiffeomorphicMap(2, domain_shape, domain_grid2world,
                                          domain_shape, domain_grid2world,
                                          codomain_shape, codomain_grid2world,
                                          None)
    diff_map.forward = disp
    diff_map.codomain_shape = None
    #If we don't provide the sampling info, it should try to use the map's
    #info, but it's None...
    assert_raises(ValueError, diff_map.transform_inverse,
                  moving_image, 'linear')

    #We must provide, at least, the reference grid shape
    assert_raises(ValueError, imwarp.DiffeomorphicMap, 2, None)


def test_diffeomorphic_map_simplification_2d():
    r"""
    Create an invertible deformation field, and define a DiffeomorphicMap
    using different voxel-to-space transforms for domain, codomain, and
    reference discretizations, also use a non-identity pre-aligning matrix.
    Warp a circle using the diffeomorphic map to obtain the expected warped
    circle. Now simplify the DiffeomorphicMap and warp the same circle using
    this simplified map. Verify that the two warped circles are equal up to
    numerical precision.
    """
    #create a simple affine transformation
    dom_shape = (64, 64)
    cod_shape = (80, 80)
    nr = dom_shape[0]
    nc = dom_shape[1]
    s = 1.1
    t = 0.25
    trans = np.array([[1, 0, -t*nr],
                      [0, 1, -t*nc],
                      [0, 0, 1]])
    trans_inv = np.linalg.inv(trans)
    scale = np.array([[1*s, 0, 0],
                      [0, 1*s, 0],
                      [0, 0, 1]])
    gt_affine = trans_inv.dot(scale.dot(trans))
    # Create the invertible displacement fields and the circle
    radius = 16
    circle = vfu.create_circle(cod_shape[0], cod_shape[1], radius)
    d, dinv = vfu.create_harmonic_fields_2d(dom_shape[0], dom_shape[1], 0.3, 6)
    #Define different voxel-to-space transforms for domain, codomain and
    #reference grid, also, use a non-identity pre-align transform
    D = gt_affine
    C = imwarp.mult_aff(gt_affine, gt_affine)
    R = np.eye(3)
    P = gt_affine

    #Create the original diffeomorphic map
    diff_map = imwarp.DiffeomorphicMap(2, dom_shape, R,
                                          dom_shape, D,
                                          cod_shape, C,
                                          P)
    diff_map.forward = np.array(d, dtype = floating)
    diff_map.backward = np.array(dinv, dtype = floating)
    #Warp the circle to obtain the expected image
    expected = diff_map.transform(circle, 'linear')

    #Simplify
    simplified = diff_map.get_simplified_transform()
    #warp the circle
    warped = simplified.transform(circle, 'linear')
    #verify that the simplified map is equivalent to the
    #original one
    assert_array_almost_equal(warped, expected)
    #And of course, it must be simpler...
    assert_equal(simplified.domain_grid2world, None)
    assert_equal(simplified.codomain_grid2world, None)
    assert_equal(simplified.disp_grid2world, None)
    assert_equal(simplified.domain_world2grid, None)
    assert_equal(simplified.codomain_world2grid, None)
    assert_equal(simplified.disp_world2grid, None)


def test_diffeomorphic_map_simplification_3d():
    r"""
    Create an invertible deformation field, and define a DiffeomorphicMap
    using different voxel-to-space transforms for domain, codomain, and
    reference discretizations, also use a non-identity pre-aligning matrix.
    Warp a sphere using the diffeomorphic map to obtain the expected warped
    sphere. Now simplify the DiffeomorphicMap and warp the same sphere using
    this simplified map. Verify that the two warped spheres are equal up to
    numerical precision.
    """
    #create a simple affine transformation
    domain_shape = (64, 64, 64)
    codomain_shape = (80, 80, 80)
    nr = domain_shape[0]
    nc = domain_shape[1]
    ns = domain_shape[2]
    s = 1.1
    t = 0.25
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
    # Create the invertible displacement fields and the sphere
    radius = 16
    sphere = vfu.create_sphere(codomain_shape[0], codomain_shape[1],
                               codomain_shape[2], radius)
    d, dinv = vfu.create_harmonic_fields_3d(domain_shape[0], domain_shape[1],
                                            domain_shape[2], 0.3, 6)
    #Define different voxel-to-space transforms for domain, codomain and
    #reference grid, also, use a non-identity pre-align transform
    D = gt_affine
    C = imwarp.mult_aff(gt_affine, gt_affine)
    R = np.eye(4)
    P = gt_affine

    #Create the original diffeomorphic map
    diff_map = imwarp.DiffeomorphicMap(3, domain_shape, R,
                                          domain_shape, D,
                                          codomain_shape, C,
                                          P)
    diff_map.forward = np.array(d, dtype = floating)
    diff_map.backward = np.array(dinv, dtype = floating)
    #Warp the sphere to obtain the expected image
    expected = diff_map.transform(sphere, 'linear')

    #Simplify
    simplified = diff_map.get_simplified_transform()
    #warp the sphere
    warped = simplified.transform(sphere, 'linear')
    #verify that the simplified map is equivalent to the
    #original one
    assert_array_almost_equal(warped, expected)
    #And of course, it must be simpler...
    assert_equal(simplified.domain_grid2world, None)
    assert_equal(simplified.codomain_grid2world, None)
    assert_equal(simplified.disp_grid2world, None)
    assert_equal(simplified.domain_world2grid, None)
    assert_equal(simplified.codomain_world2grid, None)
    assert_equal(simplified.disp_world2grid, None)

def test_optimizer_exceptions():
    #An arbitrary valid metric
    metric = metrics.SSDMetric(2)
    # The metric must not be None
    assert_raises(ValueError, imwarp.SymmetricDiffeomorphicRegistration, None)
    # The iterations list must not be empty
    assert_raises(ValueError, imwarp.SymmetricDiffeomorphicRegistration,
                  metric, [])

    optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric, None)
    #Verify the default iterations list
    assert_array_equal(optimizer.level_iters, [100,100,25])

    #Verify exception thrown when attepting to fit the energy profile without
    #enough data
    assert_raises(ValueError, optimizer._get_energy_derivative)


def test_get_direction_and_spacings():
    xrot = 0.5
    yrot = 0.75
    zrot = 1.0
    direction_gt = eulerangles.euler2mat(zrot, yrot, xrot)
    spacings_gt = np.array([1.1, 1.2, 1.3])
    scaling_gt = np.diag(spacings_gt)
    translation_gt = np.array([1,2,3])

    affine = np.eye(4)
    affine[:3, :3] = direction_gt.dot(scaling_gt)
    affine[:3, 3] = translation_gt

    direction, spacings = imwarp.get_direction_and_spacings(affine, 3)
    assert_array_almost_equal(direction, direction_gt)
    assert_array_almost_equal(spacings, spacings_gt)

def simple_callback(sdr, status):
    if status == imwarp.RegistrationStages.INIT_START:
        sdr.INIT_START_CALLED = 1
    if status == imwarp.RegistrationStages.INIT_END:
        sdr.INIT_END_CALLED = 1
    if status == imwarp.RegistrationStages.OPT_START:
        sdr.OPT_START_CALLED = 1
    if status == imwarp.RegistrationStages.OPT_END:
        sdr.OPT_END_CALLED = 1
    if status == imwarp.RegistrationStages.SCALE_START:
        sdr.SCALE_START_CALLED = 1
    if status == imwarp.RegistrationStages.SCALE_END:
        sdr.SCALE_END_CALLED = 1
    if status == imwarp.RegistrationStages.ITER_START:
        sdr.ITER_START_CALLED = 1
    if status == imwarp.RegistrationStages.ITER_END:
        sdr.ITER_END_CALLED = 1


def subsample_profile(profile, nsamples):
    plen = len(profile)
    stride = np.max([1, (plen - 1) // (nsamples - 1)])
    subsampled = np.array(
        profile[:(1 + (nsamples - 1) * stride):stride])
    return subsampled

@npt.dec.skipif(NO_SSE2)
def test_ssd_2d_demons():
    r'''
    Classical Circle-To-C experiment for 2D Monomodal registration. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of SSD in
    2D using the Demons step, and this test checks that the current energy
    profile matches the saved one.
    '''
    fname_moving = get_data('reg_o')
    fname_static = get_data('reg_c')

    moving = np.load(fname_moving)
    static = np.load(fname_static)
    moving = np.array(moving, dtype=floating)
    static = np.array(static, dtype=floating)
    moving = (moving-moving.min())/(moving.max() - moving.min())
    static = (static-static.min())/(static.max() - static.min())
    #Create the SSD metric
    smooth = 4
    step_type = 'demons'
    similarity_metric = metrics.SSDMetric(2, smooth=smooth, step_type=step_type)

    #Configure and run the Optimizer
    level_iters = [200, 100, 50, 25]
    step_length = 0.25
    opt_tol = 1e-4
    inv_iter = 40
    inv_tol = 1e-3
    ss_sigma_factor = 0.2
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(similarity_metric,
        level_iters, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)

    #test callback being called
    optimizer.INIT_START_CALLED = 0
    optimizer.INIT_END_CALLED = 0
    optimizer.OPT_START_CALLED = 0
    optimizer.OPT_END_CALLED = 0
    optimizer.SCALE_START_CALLED = 0
    optimizer.SCALE_END_CALLED = 0
    optimizer.ITER_START_CALLED = 0
    optimizer.ITER_END_CALLED = 0

    optimizer.callback_counter_test = 0
    optimizer.callback = simple_callback

    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)
    subsampled_energy_profile = subsample_profile(
        optimizer.full_energy_profile, 10)
    print(subsampled_energy_profile)

    if USING_VC_SSE2:
        expected_profile = \
            np.array([312.6813333, 80.74625551, 49.43591374, 34.08871301,
                      25.18286981, 17.78955273, 25.91334939, 20.16932281,
                      43.86083145, 79.0966558 ])
    elif USING_GCC_SSE2:
        expected_profile = \
            np.array([312.6813333, 98.17321941, 60.98300837, 47.75387157,
                      34.11067498, 122.91901409, 19.75599298, 14.28763847,
                      36.33599718, 88.62426913])

    assert_array_almost_equal(subsampled_energy_profile,
                              expected_profile, decimal=5)
    assert_equal(optimizer.OPT_START_CALLED, 1)
    assert_equal(optimizer.OPT_END_CALLED, 1)
    assert_equal(optimizer.SCALE_START_CALLED, 1)
    assert_equal(optimizer.SCALE_END_CALLED, 1)
    assert_equal(optimizer.ITER_START_CALLED, 1)
    assert_equal(optimizer.ITER_END_CALLED, 1)


@npt.dec.skipif(NO_SSE2)
def test_ssd_2d_gauss_newton():
    r'''
    Classical Circle-To-C experiment for 2D Monomodal registration. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of SSD
    in 2D using the Gauss Newton step, and this test checks that the current
    energy profile matches the saved one.
    '''
    fname_moving = get_data('reg_o')
    fname_static = get_data('reg_c')

    moving = np.load(fname_moving)
    static = np.load(fname_static)
    moving = np.array(moving, dtype=floating)
    static = np.array(static, dtype=floating)
    moving = (moving-moving.min())/(moving.max() - moving.min())
    static = (static-static.min())/(static.max() - static.min())
    #Create the SSD metric
    smooth = 4
    inner_iter = 5
    step_type = 'gauss_newton'
    similarity_metric = metrics.SSDMetric(2, smooth, inner_iter, step_type)

    #Configure and run the Optimizer
    level_iters = [200, 100, 50, 25]
    step_length = 0.5
    opt_tol = 1e-4
    inv_iter = 40
    inv_tol = 1e-3
    ss_sigma_factor = 0.2
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(similarity_metric,
        level_iters, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)

    #test callback not being called
    optimizer.INIT_START_CALLED = 0
    optimizer.INIT_END_CALLED = 0
    optimizer.OPT_START_CALLED = 0
    optimizer.OPT_END_CALLED = 0
    optimizer.SCALE_START_CALLED = 0
    optimizer.SCALE_END_CALLED = 0
    optimizer.ITER_START_CALLED = 0
    optimizer.ITER_END_CALLED = 0

    optimizer.verbosity = VerbosityLevels.DEBUG
    id = np.eye(3)
    mapping = optimizer.optimize(static, moving, id, id, id)
    m = optimizer.get_map()
    assert_equal(mapping, m)
    subsampled_energy_profile = subsample_profile(
        optimizer.full_energy_profile, 10)
    print(subsampled_energy_profile)

    if USING_VC_SSE2:
        expected_profile = \
            np.array([312.68133316, 70.17782995, 21.38508088, 96.41054776,
                      49.990781, 43.11867579, 24.53952718, 51.0786643,
                      143.24848252, 150.48349573])
    elif USING_GCC_SSE2:
        expected_profile = \
            np.array([312.68133316, 70.17782938, 21.26798507, 96.51765054,
                      51.1495088, 37.86204803, 21.62425293, 49.44868302,
                      121.6643917, 137.91427228])

    assert_array_almost_equal(subsampled_energy_profile, expected_profile,
                              decimal = 5)
    assert_equal(optimizer.OPT_START_CALLED, 0)
    assert_equal(optimizer.OPT_END_CALLED, 0)
    assert_equal(optimizer.SCALE_START_CALLED, 0)
    assert_equal(optimizer.SCALE_END_CALLED, 0)
    assert_equal(optimizer.ITER_START_CALLED, 0)
    assert_equal(optimizer.ITER_END_CALLED, 0)


def get_synthetic_warped_circle(nslices):
    #get a subsampled circle
    fname_cicle = get_data('reg_o')
    circle = np.load(fname_cicle)[::4,::4].astype(floating)

    #create a synthetic invertible map and warp the circle
    d, dinv = vfu.create_harmonic_fields_2d(64, 64, 0.1, 4)
    d = np.asarray(d, dtype=floating)
    dinv = np.asarray(dinv, dtype=floating)
    mapping = DiffeomorphicMap(2, (64, 64))
    mapping.forward, mapping.backward = d, dinv
    wcircle = mapping.transform(circle)

    if(nslices == 1):
        return circle, wcircle

    #normalize and form the 3d by piling slices
    circle = (circle-circle.min())/(circle.max() - circle.min())
    circle_3d = np.ndarray(circle.shape + (nslices,), dtype=floating)
    circle_3d[...] = circle[...,None]
    circle_3d[...,0] = 0
    circle_3d[...,-1] = 0

    #do the same with the warped circle
    wcircle = (wcircle-wcircle.min())/(wcircle.max() - wcircle.min())
    wcircle_3d = np.ndarray(wcircle.shape + (nslices,), dtype=floating)
    wcircle_3d[...] = wcircle[...,None]
    wcircle_3d[...,0] = 0
    wcircle_3d[...,-1] = 0

    return circle_3d, wcircle_3d


@npt.dec.skipif(NO_SSE2)
def test_ssd_3d_demons():
    r'''
    Register a stack of circles ('cylinder') before and after warping them
    with a synthetic diffeomorphism. This test is intended to detect
    regressions only: we saved the energy profile (the sequence of energy
    values at each iteration) of a working version of SSD in 3D using the
    Demons step, and this test checks that the current energy profile matches
    the saved one. The validation of the "working version" was done by
    registering the 18 manually annotated T1 brain MRI database IBSR with each
    other and computing the jaccard index for all 31 common anatomical regions.
    '''
    moving, static = get_synthetic_warped_circle(30)
    moving[...,:8] = 0
    moving[...,-1:-9:-1] = 0
    static[...,:8] = 0
    static[...,-1:-9:-1] = 0

    #Create the SSD metric
    smooth = 4
    step_type = 'demons'
    similarity_metric = metrics.SSDMetric(3, smooth=smooth,
                                          step_type=step_type)

    #Create the optimizer
    level_iters = [10, 5]
    step_length = 0.25
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.5
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(similarity_metric,
        level_iters, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)
    energy_profile = subsample_profile(
        optimizer.full_energy_profile, 10)
    print(energy_profile)

    if USING_VC_SSE2:
        expected_profile = \
            np.array([312.22706987, 154.65556884, 53.88543188, 9.11484007,
                      36.46592407, 13.20522299, 48.65663399, 14.91579802,
                      49.82954704, 14.92646254])
    elif USING_GCC_SSE2:
        expected_profile = \
            np.array([312.22706987, 154.65556885, 53.88455398, 9.11770682,
                      36.48642824, 13.21706748, 48.67710635, 14.91782047,
                      49.84142899, 14.92531294])

    assert_array_almost_equal(energy_profile, expected_profile, decimal=4)


@npt.dec.skipif(NO_SSE2)
def test_ssd_3d_gauss_newton():
    r'''
    Register a stack of circles ('cylinder') before and after warping them with
    a synthetic diffeomorphism. This test is intended to detect regressions
    only: we saved the energy profile (the sequence of energy values at each
    iteration) of a working version of SSD in 3D using the Gauss-Newton step,
    and this test checks that the current energy profile matches the saved
    one. The validation of the "working version" was
    done by registering the 18 manually annotated T1 brain MRI database IBSR
    with each other and computing the jaccard index for all 31 common
    anatomical regions.
    '''
    moving, static = get_synthetic_warped_circle(35)
    moving[...,:10] = 0
    moving[...,-1:-11:-1] = 0
    static[...,:10] = 0
    static[...,-1:-11:-1] = 0

    #Create the SSD metric
    smooth = 4
    inner_iter = 5
    step_type = 'gauss_newton'
    similarity_metric = metrics.SSDMetric(3, smooth, inner_iter, step_type)

    #Create the optimizer
    level_iters = [10, 5]
    step_length = 0.25
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.5
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(similarity_metric,
        level_iters, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)
    energy_profile = subsample_profile(
        optimizer.full_energy_profile, 10)
    print(energy_profile)

    if USING_VC_SSE2:
        expected_profile = \
            np.array([348.3204721, 143.480757, 44.30003405, 8.73624842,
                      3.13227203, 14.70806563, 6.48360268, 23.52491883,
                      17.25669088, 48.99709064])
    elif USING_GCC_SSE2:
        expected_profile = \
            np.array([348.3204721, 143.48075646, 44.30003413, 8.73624841,
                      3.13227181, 14.70806845, 6.48360884, 23.52499421,
                      17.25667176, 48.997691])

    assert_array_almost_equal(energy_profile, expected_profile, decimal=4)


@npt.dec.skipif(NO_SSE2)
def test_cc_2d():
    r'''
    Register a circle to itself after warping it under a synthetic invertible
    map. This test is intended to detect regressions only: we saved the energy
    profile (the sequence of energy values at each iteration) of a working
    version of CC in 2D, and this test checks that the current energy profile
    matches the saved one.
    '''

    moving, static = get_synthetic_warped_circle(1)
    #Configure the metric
    sigma_diff = 3.0
    radius = 4
    metric = metrics.CCMetric(2, sigma_diff, radius)

    #Configure and run the Optimizer
    level_iters = [10, 5]
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric, level_iters)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)
    energy_profile = subsample_profile(
        optimizer.full_energy_profile, 10)
    print(energy_profile)

    if USING_VC_SSE2:
        expected_profile = \
            [-681.02276193, -910.57721051, -1012.76781394, -1021.24181308,
             -1016.97233745, -977.35458126, -1013.90114894, -989.04516449,
             -1021.72431465, -988.46698723]
    elif USING_GCC_SSE2:
        expected_profile = \
            [-681.02276236, -920.57714783, -1008.82241171, -1021.91021701,
             -994.86961164, -1026.52978164, -1015.83587405, -1020.02780802,
             -993.8576053, -1026.4369566 ]

    expected_profile = np.asarray(expected_profile)
    assert_array_almost_equal(energy_profile, expected_profile, decimal=5)


@npt.dec.skipif(NO_SSE2)
def test_cc_3d():
    r'''
    Register a stack of circles ('cylinder') before and after warping them with
    a synthetic diffeomorphism. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of CC in
    3D, and this test checks that the current energy profile matches the saved
    one. The validation of the "working version" was done by registering the
    18 manually annotated T1 brain MRI database IBSR with each other and
    computing the jaccard index for all 31 common anatomical regions. The
    "working version" of CC in 3D obtains very similar results as
    those reported for ANTS on the same database with the same number of
    iterations. Any modification that produces a change in the energy profile
    should be carefully validated to ensure no accuracy loss.
    '''
    moving, static = moving, static = get_synthetic_warped_circle(20)

    #Create the CC metric
    sigma_diff = 2.0
    radius = 4
    similarity_metric = metrics.CCMetric(3, sigma_diff, radius)

    #Create the optimizer
    level_iters = [20, 10]
    step_length = 0.25
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.5
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(similarity_metric,
        level_iters, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    optimizer.verbosity = VerbosityLevels.DEBUG

    mapping = optimizer.optimize(static, moving, None, None, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)
    energy_profile = subsample_profile(
        optimizer.full_energy_profile, 10)*1e-4
    print(energy_profile)

    if USING_VC_SSE2:
        expected_profile = \
            [-0.17336006, -0.20516197, -0.20448353, -0.20630727, -0.20652892,
             -0.2073403, -3.0046531, -3.43771429, -3.47262116, -3.51383381]
    elif USING_GCC_SSE2:
        expected_profile = \
            [-0.17136006, -0.20632291, -0.2038927, -0.20688352, -0.20821154,
             -0.20909298, -0.20872891, -0.20933514, -3.06861497, -3.07851062]

    expected_profile = np.asarray(expected_profile)
    assert_array_almost_equal(energy_profile, expected_profile, decimal=4)


@npt.dec.skipif(NO_SSE2)
def test_em_3d_gauss_newton():
    r'''
    Register a stack of circles ('cylinder') before and after warping them with
    a synthetic diffeomorphism. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of EM in
    3D, and this test checks that the current energy profile matches the saved
    one. The validation of the "working version" was
    done by registering the 18 manually annotated T1 brain MRI database IBSR
    with each other and computing the jaccard index for all 31 common
    anatomical regions. The "working version" of EM in 3D obtains very similar
    results as those reported for ANTS on the same database. Any modification
    that produces a change in the energy profile should be carefully validated
    to ensure no accuracy loss.
    '''
    moving, static = get_synthetic_warped_circle(30)
    moving[...,:8] = 0
    moving[...,-1:-9:-1] = 0
    static[...,:8] = 0
    static[...,-1:-9:-1] = 0

    #Create the EM metric
    smooth=25.0
    inner_iter=20
    step_length=0.25
    q_levels=256
    double_gradient=True
    iter_type='gauss_newton'
    similarity_metric = metrics.EMMetric(
        3, smooth, inner_iter, q_levels, double_gradient, iter_type)

    #Create the optimizer
    level_iters = [10, 5]
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.5
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(similarity_metric,
        level_iters, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)
    energy_profile = subsample_profile(
        optimizer.full_energy_profile, 10)
    print(energy_profile)

    if USING_VC_SSE2:
        expected_profile = \
            np.array([144.03694724, 63.06874155, 51.84694887, 39.6374044,
                      31.84981429, 44.3778833, 37.84961761, 38.00509734,
                      38.67423812, 38.47003306])
    elif USING_GCC_SSE2:
        expected_profile = \
            np.array([144.03694724, 63.06874148, 51.84694881, 39.63740417,
                      31.84981481, 44.37788414, 37.84961844, 38.00509881,
                      38.67423954, 38.47003339])

    assert_array_almost_equal(energy_profile, expected_profile, decimal=4)


@npt.dec.skipif(NO_SSE2)
def test_em_2d_gauss_newton():
    r'''
    Register a circle to itself after warping it under a synthetic invertible
    map. This test is intended to detect regressions only: we saved the energy
    profile (the sequence of energy values at each iteration) of a working
    version of EM in 2D, and this test checks that the current energy profile
    matches the saved one.
    '''

    moving, static = get_synthetic_warped_circle(1)

    #Configure the metric
    smooth=25.0
    inner_iter=20
    q_levels=256
    double_gradient=False
    iter_type='gauss_newton'
    metric = metrics.EMMetric(
        2, smooth, inner_iter, q_levels, double_gradient, iter_type)

    #Configure and run the Optimizer
    level_iters = [40, 20, 10]
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric, level_iters)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)
    energy_profile = subsample_profile(
        optimizer.full_energy_profile, 10)
    print(energy_profile)

    if USING_VC_SSE2:
        expected_profile = \
            [2.50773392, 0.41762978, 0.30900322, 0.14818498, 0.44620725,
             1.53134054, 1.42115728, 1.66358267, 1.184265, 46.13635772]
    elif USING_GCC_SSE2:
        expected_profile = \
            [2.50773392, 0.41763383, 0.30908578, 0.06241115, 0.11573476,
             2.48475885, 1.10053769, 0.9270271, 49.37186785, 44.72643467]

    assert_array_almost_equal(energy_profile, np.array(expected_profile),
                              decimal=5)


@npt.dec.skipif(NO_SSE2)
def test_em_3d_demons():
    r'''
    Register a stack of circles ('cylinder') before and after warping them with
    a synthetic diffeomorphism. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of EM in
    3D, and this test checks that the current energy profile matches the saved
    one. The validation of the "working version" was
    done by registering the 18 manually annotated T1 brain MRI database IBSR
    with each other and computing the jaccard index for all 31 common
    anatomical regions. The "working version" of EM in 3D obtains very similar
    results as those reported for ANTS on the same database. Any modification
    that produces a change in the energy profile should be carefully validated
    to ensure no accuracy loss.
    '''
    moving, static = get_synthetic_warped_circle(30)
    moving[...,:8] = 0
    moving[...,-1:-9:-1] = 0
    static[...,:8] = 0
    static[...,-1:-9:-1] = 0

    #Create the EM metric
    smooth=25.0
    inner_iter=20
    step_length=0.25
    q_levels=256
    double_gradient=True
    iter_type='demons'
    similarity_metric = metrics.EMMetric(
        3, smooth, inner_iter, q_levels, double_gradient, iter_type)

    #Create the optimizer
    level_iters = [10, 5]
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.5
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(similarity_metric,
        level_iters, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)
    energy_profile = subsample_profile(
        optimizer.full_energy_profile, 10)
    print(energy_profile)

    if USING_VC_SSE2:
        expected_profile = \
            np.array([144.03694708, 122.39512307, 111.31925381, 90.9100989,
                      93.93705232, 104.22993997, 110.57817867, 140.45262039,
                      133.87804571, 119.20794977])
    elif USING_GCC_SSE2:
        expected_profile = \
            np.array([144.03694708, 122.39512227, 111.31924572, 90.91010482,
                      93.93707059, 104.22996918, 110.57822649, 140.45298465,
                      133.87831302, 119.20826433])

    assert_array_almost_equal(energy_profile, expected_profile, decimal=4)


@npt.dec.skipif(NO_SSE2)
def test_em_2d_demons():
    r'''
    Register a circle to itself after warping it under a synthetic invertible
    map. This test is intended to detect regressions only: we saved the energy
    profile (the sequence of energy values at each iteration) of a working
    version of EM in 2D, and this test checks that the current energy profile
    matches the saved one.
    '''

    moving, static = get_synthetic_warped_circle(1)

    #Configure the metric
    smooth=25.0
    inner_iter=20
    q_levels=256
    double_gradient=False
    iter_type='demons'
    metric = metrics.EMMetric(
        2, smooth, inner_iter, q_levels, double_gradient, iter_type)

    #Configure and run the Optimizer
    level_iters = [40, 20, 10]
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric, level_iters)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)
    energy_profile = subsample_profile(
        optimizer.full_energy_profile, 10)
    print(energy_profile)

    if USING_VC_SSE2:
        expected_profile = \
            [2.50773393, 3.26942324, 1.81684393, 5.44878881, 40.0195918,
             31.87030788, 25.15710409, 29.82206485, 196.33114499, 213.86419995]
    elif USING_GCC_SSE2:
        expected_profile = \
            [2.50773393, 3.26942352, 1.8168445, 5.44879264, 40.01956373,
             31.65616398, 32.43115903, 35.24130742, 192.89072697, 195.456909]

    assert_array_almost_equal(energy_profile, np.array(expected_profile),
                              decimal=5)

if __name__=='__main__':
    test_scale_space_exceptions()
    test_optimizer_exceptions()
    test_mult_aff()
    test_diffeomorphic_map_2d()
    test_diffeomorphic_map_simplification_2d()
    test_diffeomorphic_map_simplification_3d()
    test_get_direction_and_spacings()
    test_ssd_2d_demons()
    test_ssd_2d_gauss_newton()
    test_ssd_3d_demons()
    test_ssd_3d_gauss_newton()
    test_cc_2d()
    test_cc_3d()
    test_em_2d_gauss_newton()
    test_em_3d_gauss_newton()
    test_em_3d_demons()
    test_em_2d_demons()
