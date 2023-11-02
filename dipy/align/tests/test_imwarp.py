import numpy as np
import nibabel.eulerangles as eulerangles
from numpy.testing import (assert_equal,
                           assert_array_equal,
                           assert_array_almost_equal,
                           assert_raises)
from dipy.core.interpolation import (interpolate_scalar_2d,
                                     interpolate_scalar_3d)
from dipy.data import get_fnames
from dipy.align import floating
from dipy.align import imwarp as imwarp
from dipy.align import metrics as metrics
from dipy.align import vector_fields as vfu
from dipy.align import VerbosityLevels
from dipy.align.imwarp import DiffeomorphicMap
from dipy.tracking.streamline import deform_streamlines
from dipy.testing.decorators import set_random_number_generator


def test_mult_aff():
    r""" Test matrix multiplication using None as identity
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


@set_random_number_generator(2022966)
def test_diffeomorphic_map_2d(rng):
    r""" Test 2D DiffeomorphicMap

    Creates a random displacement field that exactly maps pixels from an
    input image to an output image. First a discrete random assignment
    between the images is generated, then each pair of mapped points are
    transformed to the physical space by assigning a pair of arbitrary,
    fixed affine matrices to input and output images, and finally the
    difference between their positions is taken as the displacement vector.
    The resulting displacement, although operating in physical space,
    maps the points exactly (up to numerical precision).
    """
    domain_shape = (10, 10)
    codomain_shape = (10, 10)
    # create a simple affine transformation
    nr = domain_shape[0]
    nc = domain_shape[1]
    s = 1.1
    t = 0.25
    trans = np.array([[1, 0, -t * nr],
                      [0, 1, -t * nc],
                      [0, 0, 1]])
    trans_inv = np.linalg.inv(trans)
    scale = np.array([[1 * s, 0, 0],
                      [0, 1 * s, 0],
                      [0, 0, 1]])
    gt_affine = trans_inv.dot(scale.dot(trans))

    # create the random displacement field
    domain_grid2world = gt_affine
    codomain_grid2world = gt_affine
    disp, assign = vfu.create_random_displacement_2d(
        np.array(domain_shape, dtype=np.int32),
        domain_grid2world, np.array(codomain_shape, dtype=np.int32),
        codomain_grid2world)
    disp = np.array(disp, dtype=floating)
    assign = np.array(assign)
    # create a random image (with decimal digits) to warp
    moving_image = np.ndarray(codomain_shape, dtype=floating)
    ns = np.size(moving_image)
    moving_image[...] = rng.integers(0, 10, ns).reshape(codomain_shape)
    # set boundary values to zero so we don't test wrong interpolation due
    # to floating point precision
    moving_image[0, :] = 0
    moving_image[-1, :] = 0
    moving_image[:, 0] = 0
    moving_image[:, -1] = 0

    # warp the moving image using the (exact) assignments
    expected = moving_image[(assign[..., 0], assign[..., 1])]

    # warp using a DiffeomorphicMap instance
    diff_map = imwarp.DiffeomorphicMap(2, domain_shape, domain_grid2world,
                                       domain_shape, domain_grid2world,
                                       codomain_shape, codomain_grid2world,
                                       None)
    diff_map.forward = disp

    # Verify that the transform method accepts different image types (note that
    # the actual image contained integer values, we don't want to test
    # rounding)
    for _type in [floating, np.float64, np.int64, np.int32]:
        moving_image = moving_image.astype(_type)

        # warp using linear interpolation
        warped = diff_map.transform(moving_image, 'linear')
        # compare the images (the linear interpolation may introduce slight
        # precision errors)
        assert_array_almost_equal(warped, expected, decimal=5)

        # Now test the nearest neighbor interpolation
        warped = diff_map.transform(moving_image, 'nearest')
        # compare the images (now we don't have to worry about precision,
        # it is n.n.)
        assert_array_almost_equal(warped, expected)

        # verify the is_inverse flag
        inv = diff_map.inverse()
        warped = inv.transform_inverse(moving_image, 'linear')
        assert_array_almost_equal(warped, expected, decimal=5)

        warped = inv.transform_inverse(moving_image, 'nearest')
        assert_array_almost_equal(warped, expected)

    # Now test the inverse functionality
    diff_map = imwarp.DiffeomorphicMap(2, codomain_shape, codomain_grid2world,
                                       codomain_shape, codomain_grid2world,
                                       domain_shape, domain_grid2world, None)
    diff_map.backward = disp
    for _type in [floating, np.float64, np.int64, np.int32]:
        moving_image = moving_image.astype(_type)

        # warp using linear interpolation
        warped = diff_map.transform_inverse(moving_image, 'linear')
        # compare the images (the linear interpolation may introduce slight
        # precision errors)
        assert_array_almost_equal(warped, expected, decimal=5)

        # Now test the nearest neighbor interpolation
        warped = diff_map.transform_inverse(moving_image, 'nearest')
        # compare the images (now we don't have to worry about precision,
        # it is nearest neighbour)
        assert_array_almost_equal(warped, expected)

    # Verify that DiffeomorphicMap raises the appropriate exceptions when
    # the sampling information is undefined
    diff_map = imwarp.DiffeomorphicMap(2, domain_shape, domain_grid2world,
                                       domain_shape, domain_grid2world,
                                       codomain_shape, codomain_grid2world,
                                       None)
    diff_map.forward = disp
    diff_map.domain_shape = None
    # If we don't provide the sampling info, it should try to use the map's
    # info, but it's None...
    assert_raises(ValueError, diff_map.transform, moving_image, 'linear')

    # Same test for diff_map.transform_inverse
    diff_map = imwarp.DiffeomorphicMap(2, domain_shape, domain_grid2world,
                                       domain_shape, domain_grid2world,
                                       codomain_shape, codomain_grid2world,
                                       None)
    diff_map.forward = disp
    diff_map.codomain_shape = None
    # If we don't provide the sampling info, it should try to use the map's
    # info, but it's None...
    assert_raises(ValueError, diff_map.transform_inverse,
                  moving_image, 'linear')

    # We must provide, at least, the reference grid shape
    assert_raises(ValueError, imwarp.DiffeomorphicMap, 2, None)

    # Verify that matrices are correctly interpreted from string
    non_array_obj = diff_map
    array_obj = np.ones((3, 3))
    assert_raises(ValueError, diff_map.interpret_matrix, 'a different string')
    assert_raises(ValueError, diff_map.interpret_matrix, non_array_obj)
    assert(diff_map.interpret_matrix('identity') is None)
    assert(diff_map.interpret_matrix(None) is None)
    assert_array_equal(diff_map.interpret_matrix(array_obj), array_obj)


def test_diffeomorphic_map_simplification_2d():
    r""" Test simplification of 2D diffeomorphic maps

    Create an invertible deformation field, and define a DiffeomorphicMap
    using different voxel-to-space transforms for domain, codomain, and
    reference discretizations, also use a non-identity pre-aligning matrix.
    Warp a circle using the diffeomorphic map to obtain the expected warped
    circle. Now simplify the DiffeomorphicMap and warp the same circle
    using this simplified map. Verify that the two warped circles are equal
    up to numerical precision.
    """
    # create a simple affine transformation
    dom_shape = (64, 64)
    cod_shape = (80, 80)
    nr = dom_shape[0]
    nc = dom_shape[1]
    s = 1.1
    t = 0.25
    trans = np.array([[1, 0, -t * nr],
                      [0, 1, -t * nc],
                      [0, 0, 1]])
    trans_inv = np.linalg.inv(trans)
    scale = np.array([[1 * s, 0, 0],
                      [0, 1 * s, 0],
                      [0, 0, 1]])
    gt_affine = trans_inv.dot(scale.dot(trans))
    # Create the invertible displacement fields and the circle
    radius = 16
    circle = vfu.create_circle(cod_shape[0], cod_shape[1], radius)
    d, dinv = vfu.create_harmonic_fields_2d(dom_shape[0],
                                            dom_shape[1], 0.3, 6)
    # Define different voxel-to-space transforms for domain, codomain and
    # reference grid, also, use a non-identity pre-align transform
    D = gt_affine
    C = imwarp.mult_aff(gt_affine, gt_affine)
    R = np.eye(3)
    P = gt_affine

    # Create the original diffeomorphic map
    diff_map = imwarp.DiffeomorphicMap(2, dom_shape, R,
                                       dom_shape, D,
                                       cod_shape, C,
                                       P)
    diff_map.forward = np.array(d, dtype=floating)
    diff_map.backward = np.array(dinv, dtype=floating)
    # Warp the circle to obtain the expected image
    expected = diff_map.transform(circle, 'linear')

    # Simplify
    simplified = diff_map.get_simplified_transform()
    # warp the circle
    warped = simplified.transform(circle, 'linear')
    # verify that the simplified map is equivalent to the
    # original one
    assert_array_almost_equal(warped, expected)
    # And of course, it must be simpler...
    assert_equal(simplified.domain_grid2world, None)
    assert_equal(simplified.codomain_grid2world, None)
    assert_equal(simplified.disp_grid2world, None)
    assert_equal(simplified.domain_world2grid, None)
    assert_equal(simplified.codomain_world2grid, None)
    assert_equal(simplified.disp_world2grid, None)


def test_diffeomorphic_map_simplification_3d():
    r""" Test simplification of 3D diffeomorphic maps

    Create an invertible deformation field, and define a DiffeomorphicMap
    using different voxel-to-space transforms for domain, codomain, and
    reference discretizations, also use a non-identity pre-aligning matrix.
    Warp a sphere using the diffeomorphic map to obtain the expected warped
    sphere. Now simplify the DiffeomorphicMap and warp the same sphere
    using this simplified map. Verify that the two warped spheres are equal
    up to numerical precision.
    """
    # create a simple affine transformation
    domain_shape = (64, 64, 64)
    codomain_shape = (80, 80, 80)
    nr = domain_shape[0]
    nc = domain_shape[1]
    ns = domain_shape[2]
    s = 1.1
    t = 0.25
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
    # Create the invertible displacement fields and the sphere
    radius = 16
    sphere = vfu.create_sphere(codomain_shape[0], codomain_shape[1],
                               codomain_shape[2], radius)
    d, dinv = vfu.create_harmonic_fields_3d(domain_shape[0], domain_shape[1],
                                            domain_shape[2], 0.3, 6)
    # Define different voxel-to-space transforms for domain, codomain and
    # reference grid, also, use a non-identity pre-align transform
    D = gt_affine
    C = imwarp.mult_aff(gt_affine, gt_affine)
    R = np.eye(4)
    P = gt_affine

    # Create the original diffeomorphic map
    diff_map = imwarp.DiffeomorphicMap(3, domain_shape, R,
                                       domain_shape, D,
                                       codomain_shape, C,
                                       P)
    diff_map.forward = np.array(d, dtype=floating)
    diff_map.backward = np.array(dinv, dtype=floating)
    # Warp the sphere to obtain the expected image
    expected = diff_map.transform(sphere, 'linear')

    # Simplify
    simplified = diff_map.get_simplified_transform()
    # warp the sphere
    warped = simplified.transform(sphere, 'linear')
    # verify that the simplified map is equivalent to the
    # original one
    assert_array_almost_equal(warped, expected)
    # And of course, it must be simpler...
    assert_equal(simplified.domain_grid2world, None)
    assert_equal(simplified.codomain_grid2world, None)
    assert_equal(simplified.disp_grid2world, None)
    assert_equal(simplified.domain_world2grid, None)
    assert_equal(simplified.codomain_world2grid, None)
    assert_equal(simplified.disp_world2grid, None)


def test_optimizer_exceptions():
    r""" Test exceptions from SyN
    """
    # An arbitrary valid metric
    metric = metrics.SSDMetric(2)
    # The metric must not be None
    assert_raises(ValueError, imwarp.SymmetricDiffeomorphicRegistration,
                  None)
    # The iterations list must not be empty
    assert_raises(ValueError, imwarp.SymmetricDiffeomorphicRegistration,
                  metric, [])

    optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric, None)
    # Verify the default iterations list
    assert_array_equal(optimizer.level_iters, [100, 100, 25])

    # Verify exception thrown when attempting to fit the energy profile without
    # enough data
    assert_raises(ValueError, optimizer._get_energy_derivative)
    assert_raises(ValueError, optimizer.get_map)


def test_get_direction_and_spacings():
    r""" Test direction and spacings from affine transforms
    """
    xrot = 0.5
    yrot = 0.75
    zrot = 1.0
    direction_gt = eulerangles.euler2mat(zrot, yrot, xrot)
    spacings_gt = np.array([1.1, 1.2, 1.3])
    scaling_gt = np.diag(spacings_gt)
    translation_gt = np.array([1, 2, 3])

    affine = np.eye(4)
    affine[:3, :3] = direction_gt.dot(scaling_gt)
    affine[:3, 3] = translation_gt

    direction, spacings = imwarp.get_direction_and_spacings(affine, 3)
    assert_array_almost_equal(direction, direction_gt)
    assert_array_almost_equal(spacings, spacings_gt)


def simple_callback(sdr, status):
    r""" Verify callback function is called from SyN """
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


def test_ssd_2d_demons():
    r""" Test 2D SyN with SSD metric, demons-like optimizer

    Classical Circle-To-C experiment for 2D monomodal registration. We
    verify that the final registration is of good quality.
    """
    fname_moving = get_fnames('reg_o')
    fname_static = get_fnames('reg_c')

    moving = np.load(fname_moving)
    static = np.load(fname_static)
    moving = np.array(moving, dtype=floating)
    static = np.array(static, dtype=floating)
    moving = (moving - moving.min()) / (moving.max() - moving.min())
    static = (static - static.min()) / (static.max() - static.min())
    # Create the SSD metric
    smooth = 4
    step_type = 'demons'
    similarity_metric = metrics.SSDMetric(
        2, smooth=smooth, step_type=step_type)

    # Configure and run the Optimizer
    level_iters = [200, 100, 50, 25]
    step_length = 0.25
    opt_tol = 1e-4
    inv_iter = 40
    inv_tol = 1e-3
    ss_sigma_factor = 0.2
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric,
        level_iters,
        step_length,
        ss_sigma_factor,
        opt_tol,
        inv_iter,
        inv_tol)

    # test callback being called
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

    warped = mapping.transform(moving)
    starting_energy = np.sum((static - moving)**2)
    final_energy = np.sum((static - warped)**2)
    reduced = 1.0 - final_energy / starting_energy

    assert(reduced > 0.9)
    assert_equal(optimizer.OPT_START_CALLED, 1)
    assert_equal(optimizer.OPT_END_CALLED, 1)
    assert_equal(optimizer.SCALE_START_CALLED, 1)
    assert_equal(optimizer.SCALE_END_CALLED, 1)
    assert_equal(optimizer.ITER_START_CALLED, 1)
    assert_equal(optimizer.ITER_END_CALLED, 1)


def test_ssd_2d_gauss_newton():
    r""" Test 2D SyN with SSD metric, Gauss-Newton optimizer

    Classical Circle-To-C experiment for 2D monomodal registration. We
    verify that the final registration is of good quality.
    """
    fname_moving = get_fnames('reg_o')
    fname_static = get_fnames('reg_c')

    moving = np.load(fname_moving)
    static = np.load(fname_static)
    moving = np.array(moving, dtype=floating)
    static = np.array(static, dtype=floating)
    moving = (moving - moving.min()) / (moving.max() - moving.min())
    static = (static - static.min()) / (static.max() - static.min())
    # Create the SSD metric
    smooth = 4
    inner_iter = 5
    step_type = 'gauss_newton'
    similarity_metric = metrics.SSDMetric(2, smooth, inner_iter, step_type)

    # Configure and run the Optimizer
    level_iters = [200, 100, 50, 25]
    step_length = 0.5
    opt_tol = 1e-4
    inv_iter = 40
    inv_tol = 1e-3
    ss_sigma_factor = 0.2
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric,
        level_iters,
        step_length,
        ss_sigma_factor,
        opt_tol,
        inv_iter,
        inv_tol)

    # test callback not being called
    optimizer.INIT_START_CALLED = 0
    optimizer.INIT_END_CALLED = 0
    optimizer.OPT_START_CALLED = 0
    optimizer.OPT_END_CALLED = 0
    optimizer.SCALE_START_CALLED = 0
    optimizer.SCALE_END_CALLED = 0
    optimizer.ITER_START_CALLED = 0
    optimizer.ITER_END_CALLED = 0

    optimizer.verbosity = VerbosityLevels.DEBUG
    transformation = np.eye(3)
    mapping = optimizer.optimize(
        static, moving, transformation, transformation, transformation)
    m = optimizer.get_map()
    assert_equal(mapping, m)

    warped = mapping.transform(moving)
    starting_energy = np.sum((static - moving)**2)
    final_energy = np.sum((static - warped)**2)
    reduced = 1.0 - final_energy / starting_energy

    assert(reduced > 0.9)
    assert_equal(optimizer.OPT_START_CALLED, 0)
    assert_equal(optimizer.OPT_END_CALLED, 0)
    assert_equal(optimizer.SCALE_START_CALLED, 0)
    assert_equal(optimizer.SCALE_END_CALLED, 0)
    assert_equal(optimizer.ITER_START_CALLED, 0)
    assert_equal(optimizer.ITER_END_CALLED, 0)


def get_warped_stacked_image(image, nslices, b, m):
    r""" Creates a volume by stacking copies of a deformed image

    The image is deformed under an invertible field, and a 3D volume is
    generated as follows:
    the first and last `nslices`//3 slices are filled with zeros
    to simulate background. The remaining middle slices are filled with
    copies of the deformed `image` under the action of the invertible
    field.

    Parameters
    ----------
    image : 2d array shape(r, c)
        the image to be deformed
    nslices : int
        the number of slices in the final volume
    b, m : float
        parameters of the harmonic field (as in [1]).

    Returns
    -------
    vol : array shape(r, c) if `nslices`==1 else (r, c, `nslices`)
        the volumed generated using the undeformed image
    wvol : array shape(r, c) if `nslices`==1 else (r, c, `nslices`)
        the volumed generated using the warped image

    References
    ----------
    [1] Chen, M., Lu, W., Chen, Q., Ruchala, K. J., & Olivera, G. H. (2008).
        A simple fixed-point approach to invert a deformation field.
        Medical Physics, 35(1), 81. doi:10.1118/1.2816107
    """
    shape = image.shape
    # create a synthetic invertible map and warp the circle
    d, dinv = vfu.create_harmonic_fields_2d(shape[0], shape[1], b, m)
    d = np.asarray(d, dtype=floating)
    dinv = np.asarray(dinv, dtype=floating)
    mapping = DiffeomorphicMap(2, shape)
    mapping.forward, mapping.backward = d, dinv
    wimage = mapping.transform(image)

    if nslices == 1:
        return image, wimage

    # normalize and form the 3d by piling slices
    image = image.astype(floating)
    image = (image - image.min()) / (image.max() - image.min())
    zero_slices = nslices // 3
    vol = np.zeros(shape=image.shape + (nslices,))
    vol[..., zero_slices:(2 * zero_slices)] = image[..., None]
    wvol = np.zeros(shape=image.shape + (nslices,))
    wvol[..., zero_slices:(2 * zero_slices)] = wimage[..., None]

    return vol, wvol


def get_synthetic_warped_circle(nslices):
    # get a subsampled circle
    fname_cicle = get_fnames('reg_o')
    circle = np.load(fname_cicle)[::4, ::4].astype(floating)

    # create a synthetic invertible map and warp the circle
    d, dinv = vfu.create_harmonic_fields_2d(64, 64, 0.1, 4)
    d = np.asarray(d, dtype=floating)
    dinv = np.asarray(dinv, dtype=floating)
    mapping = DiffeomorphicMap(2, (64, 64))
    mapping.forward, mapping.backward = d, dinv
    wcircle = mapping.transform(circle)

    if nslices == 1:
        return circle, wcircle

    # normalize and form the 3d by piling slices
    circle = (circle - circle.min()) / (circle.max() - circle.min())
    circle_3d = np.ndarray(circle.shape + (nslices,), dtype=floating)
    circle_3d[...] = circle[..., None]
    circle_3d[..., 0] = 0
    circle_3d[..., -1] = 0

    # do the same with the warped circle
    wcircle = (wcircle - wcircle.min()) / (wcircle.max() - wcircle.min())
    wcircle_3d = np.ndarray(wcircle.shape + (nslices,), dtype=floating)
    wcircle_3d[...] = wcircle[..., None]
    wcircle_3d[..., 0] = 0
    wcircle_3d[..., -1] = 0

    return circle_3d, wcircle_3d


def test_ssd_3d_demons():
    r""" Test 3D SyN with SSD metric, demons-like optimizer

    Register a stack of circles ('cylinder') before and after warping them
    with a synthetic diffeomorphism. We verify that the final registration
    is of good quality.
    """
    moving, static = get_synthetic_warped_circle(30)
    moving[..., :8] = 0
    moving[..., -1:-9:-1] = 0
    static[..., :8] = 0
    static[..., -1:-9:-1] = 0

    # Create the SSD metric
    smooth = 4
    step_type = 'demons'
    similarity_metric = metrics.SSDMetric(3, smooth=smooth,
                                          step_type=step_type)

    # Create the optimizer
    level_iters = [10, 10]
    step_length = 0.1
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.5
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric,
        level_iters,
        step_length,
        ss_sigma_factor,
        opt_tol,
        inv_iter,
        inv_tol)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)

    warped = mapping.transform(moving)
    starting_energy = np.sum((static - moving)**2)
    final_energy = np.sum((static - warped)**2)
    reduced = 1.0 - final_energy / starting_energy

    assert(reduced > 0.9)


def test_ssd_3d_gauss_newton():
    r""" Test 3D SyN with SSD metric, Gauss-Newton optimizer

    Register a stack of circles ('cylinder') before and after warping them
    with a synthetic diffeomorphism. We verify that the final registration
    is of good quality.
    """
    moving, static = get_synthetic_warped_circle(35)
    moving[..., :10] = 0
    moving[..., -1:-11:-1] = 0
    static[..., :10] = 0
    static[..., -1:-11:-1] = 0

    # Create the SSD metric
    smooth = 4
    inner_iter = 5
    step_type = 'gauss_newton'
    similarity_metric = metrics.SSDMetric(3, smooth, inner_iter, step_type)

    # Create the optimizer
    level_iters = [10, 10]
    step_length = 0.1
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.5
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric,
        level_iters,
        step_length,
        ss_sigma_factor,
        opt_tol,
        inv_iter,
        inv_tol)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)

    warped = mapping.transform(moving)
    starting_energy = np.sum((static - moving)**2)
    final_energy = np.sum((static - warped)**2)
    reduced = 1.0 - final_energy / starting_energy

    assert(reduced > 0.9)


def test_cc_2d():
    r""" Test 2D SyN with CC metric

    Register a coronal slice from a T1w brain MRI before and after warping
    it under a synthetic invertible map. We verify that the final
    registration is of good quality.
    """
    fname = get_fnames('t1_coronal_slice')
    nslices = 1
    b = 0.1
    m = 4

    image = np.load(fname)
    moving, static = get_warped_stacked_image(image, nslices, b, m)

    # Configure the metric
    sigma_diff = 3.0
    radius = 4
    metric = metrics.CCMetric(2, sigma_diff, radius)

    # Configure and run the Optimizer
    level_iters = [15, 5]
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric, level_iters)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)

    warped = mapping.transform(moving)
    starting_energy = np.sum((static - moving)**2)
    final_energy = np.sum((static - warped)**2)
    reduced = 1.0 - final_energy / starting_energy

    assert(reduced > 0.9)


def test_cc_3d():
    r""" Test 3D SyN with CC metric

    Register a volume created by stacking copies of a coronal slice from
    a T1w brain MRI before and after warping it under a synthetic
    invertible map. We verify that the final registration is of good
    quality.
    """
    fname = get_fnames('t1_coronal_slice')
    nslices = 21
    b = 0.1
    m = 4

    image = np.load(fname)
    moving, static = get_warped_stacked_image(image, nslices, b, m)

    # Create the CC metric
    sigma_diff = 2.0
    radius = 2
    similarity_metric = metrics.CCMetric(3, sigma_diff, radius)

    # Create the optimizer
    level_iters = [20, 5]
    step_length = 0.25
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.2
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric,
        level_iters,
        step_length,
        ss_sigma_factor,
        opt_tol,
        inv_iter,
        inv_tol)
    optimizer.verbosity = VerbosityLevels.DEBUG

    mapping = optimizer.optimize(static, moving, None, None, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)

    warped = mapping.transform(moving)
    starting_energy = np.sum((static - moving)**2)
    final_energy = np.sum((static - warped)**2)
    reduced = 1.0 - final_energy / starting_energy

    assert(reduced > 0.9)


def test_em_3d_gauss_newton():
    r""" Test 3D SyN with EM metric, Gauss-Newton optimizer

    Register a volume created by stacking copies of a coronal slice from
    a T1w brain MRI before and after warping it under a synthetic
    invertible map. We verify that the final registration is of good
    quality.
    """
    fname = get_fnames('t1_coronal_slice')
    nslices = 21
    b = 0.1
    m = 4

    image = np.load(fname)
    moving, static = get_warped_stacked_image(image, nslices, b, m)

    # Create the EM metric
    smooth = 2.0
    inner_iter = 20
    step_length = 0.25
    q_levels = 256
    double_gradient = True
    iter_type = 'gauss_newton'
    similarity_metric = metrics.EMMetric(
        3, smooth, inner_iter, q_levels, double_gradient, iter_type)

    # Create the optimizer
    level_iters = [20, 5]
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 1.0
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric,
        level_iters,
        step_length,
        ss_sigma_factor,
        opt_tol,
        inv_iter,
        inv_tol)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)

    warped = mapping.transform(moving)
    starting_energy = np.sum((static - moving)**2)
    final_energy = np.sum((static - warped)**2)
    reduced = 1.0 - final_energy / starting_energy

    assert(reduced > 0.9)


def test_em_2d_gauss_newton():
    r""" Test 2D SyN with EM metric, Gauss-Newton optimizer

    Register a coronal slice from a T1w brain MRI before and after warping
    it under a synthetic invertible map. We verify that the final
    registration is of good quality.
    """

    fname = get_fnames('t1_coronal_slice')
    nslices = 1
    b = 0.1
    m = 4

    image = np.load(fname)
    moving, static = get_warped_stacked_image(image, nslices, b, m)

    # Configure the metric
    smooth = 5.0
    inner_iter = 20
    q_levels = 256
    double_gradient = False
    iter_type = 'gauss_newton'
    metric = metrics.EMMetric(
        2, smooth, inner_iter, q_levels, double_gradient, iter_type)

    # Configure and run the Optimizer
    level_iters = [40, 20, 10]
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric, level_iters)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)

    warped = mapping.transform(moving)
    starting_energy = np.sum((static - moving)**2)
    final_energy = np.sum((static - warped)**2)
    reduced = 1.0 - final_energy / starting_energy

    assert(reduced > 0.9)


def test_em_3d_demons():
    r""" Test 3D SyN with EM metric, demons-like optimizer

    Register a volume created by stacking copies of a coronal slice from
    a T1w brain MRI before and after warping it under a synthetic
    invertible map. We verify that the final registration is of good
    quality.
    """
    fname = get_fnames('t1_coronal_slice')
    nslices = 21
    b = 0.1
    m = 4

    image = np.load(fname)
    moving, static = get_warped_stacked_image(image, nslices, b, m)

    # Create the EM metric
    smooth = 2.0
    inner_iter = 20
    step_length = 0.25
    q_levels = 256
    double_gradient = True
    iter_type = 'demons'
    similarity_metric = metrics.EMMetric(
        3, smooth, inner_iter, q_levels, double_gradient, iter_type)

    # Create the optimizer
    level_iters = [20, 5]
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 1.0
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric,
        level_iters,
        step_length,
        ss_sigma_factor,
        opt_tol,
        inv_iter,
        inv_tol)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)

    warped = mapping.transform(moving)
    starting_energy = np.sum((static - moving)**2)
    final_energy = np.sum((static - warped)**2)
    reduced = 1.0 - final_energy / starting_energy

    assert(reduced > 0.9)


def test_em_2d_demons():
    r""" Test 2D SyN with EM metric, demons-like optimizer

    Register a coronal slice from a T1w brain MRI before and after warping
    it under a synthetic invertible map. We verify that the final
    registration is of good quality.
    """
    fname = get_fnames('t1_coronal_slice')
    nslices = 1
    b = 0.1
    m = 4

    image = np.load(fname)
    moving, static = get_warped_stacked_image(image, nslices, b, m)

    # Configure the metric
    smooth = 2.0
    inner_iter = 20
    q_levels = 256
    double_gradient = False
    iter_type = 'demons'
    metric = metrics.EMMetric(
        2, smooth, inner_iter, q_levels, double_gradient, iter_type)

    # Configure and run the Optimizer
    level_iters = [40, 20, 10]
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric, level_iters)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)

    warped = mapping.transform(moving)
    starting_energy = np.sum((static - moving)**2)
    final_energy = np.sum((static - warped)**2)
    reduced = 1.0 - final_energy / starting_energy

    assert(reduced > 0.9)


@set_random_number_generator(1741332)
def test_coordinate_mapping(rng):
    r"""Test coordinate mapping with DiffeomorphicMap

    1. Create a random displacement field and a small affine transform to map
       grid to world coordinates.
    2. Create a DiffeomorphicMap with the previously created field and affine
       transform.
    3. Create a random input image.
    4. Select a few non-boundary voxels from the domain grid.
    5. Warp the input image with the DiffeomorphicMap and interpolate the
       **warped image** at the selected locations. The result is the
       `expected` array.
    6. Map only the selected points using the DiffeomorphicMap and
       interpolate the **input image** at the warped points. The result is the
       `actual` array, which should be almost equal to the `expected` array.
    """
    for dim in range(2, 4):
        npoints = 100
        points = np.empty((npoints, dim), dtype=np.float64)
        if dim == 2:
            domain_shape = (10, 10)
            codomain_shape = (15, 15)
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
            n = codomain_shape[0] * codomain_shape[1]
            moving_image = rng.integers(0, 10, n).reshape(codomain_shape)
            moving_image = moving_image.astype(np.float64)
            # Select a few grid coordinates not at the boundary of the domain
            points[:, 0] = rng.integers(1, nr-1, npoints)
            points[:, 1] = rng.integers(1, nc-1, npoints)
            random_df = vfu.create_random_displacement_2d
            interpolate_f = interpolate_scalar_2d
        else:
            domain_shape = (10, 10, 10)
            codomain_shape = (15, 15, 15)
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
            n = codomain_shape[0] * codomain_shape[1] * codomain_shape[2]
            moving_image = rng.integers(0, 10, n).reshape(codomain_shape)
            moving_image = moving_image.astype(np.float64)
            # Select a few grid coordinates not at the boundary of the domain
            points[:, 0] = rng.integers(1, nr-1, npoints)
            points[:, 1] = rng.integers(1, nc-1, npoints)
            points[:, 2] = rng.integers(1, ns-1, npoints)
            random_df = vfu.create_random_displacement_3d
            interpolate_f = interpolate_scalar_3d

        # create the random displacement field
        domain_grid2world = gt_affine
        codomain_grid2world = gt_affine
        disp, assign = random_df(np.array(domain_shape, dtype=np.int32),
                                 domain_grid2world,
                                 np.array(codomain_shape, dtype=np.int32),
                                 codomain_grid2world)
        disp = disp.astype(floating)
        # Create a DiffeomorphicMap instance
        diff_map = imwarp.DiffeomorphicMap(dim, domain_shape,
                                           domain_grid2world, domain_shape,
                                           domain_grid2world, codomain_shape,
                                           codomain_grid2world, None)
        diff_map.forward = disp

        # Here, expected is obtained after two interpolation steps, therefore
        # we need to increase the tolerance when comparing against the result
        # using only one interpolation step (we set decimal=5 below)
        warped = diff_map.transform(moving_image, 'linear')
        expected, inside = interpolate_f(warped, points)

        # Now map the points with the implementation under test
        # Specify how to map the given array to world coordinates
        in2world = diff_map.domain_grid2world
        # Request mapping back from world to grid coordinates
        world2out = diff_map.domain_world2grid
        # Execute warping
        wpoints = diff_map.transform_points(points, in2world, world2out)
        # Interpolate at warped points and verify it's equal to direct warping
        actual, inside = interpolate_f(moving_image, wpoints)
        assert_array_almost_equal(actual, expected, decimal=5)

        if dim in [3, 4]:
            wpoints_2 = deform_streamlines([points, ], disp, np.eye(4),
                                           domain_grid2world, np.eye(4),
                                           codomain_grid2world)

            assert_array_almost_equal(wpoints, wpoints_2[0])
