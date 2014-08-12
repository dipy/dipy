from __future__ import print_function
import numpy as np
from numpy.testing import (assert_equal,
                           assert_almost_equal, 
                           assert_array_equal,
                           assert_array_almost_equal,
                           assert_raises)
import matplotlib.pyplot as plt
import dipy.align.imwarp as imwarp
import dipy.align.metrics as metrics 
import dipy.align.vector_fields as vfu
from dipy.data import get_data
from dipy.align import floating
import nibabel as nib
import nibabel.eulerangles as eulerangles
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align import VerbosityLevels

def test_mult_aff():
    r"""mult_aff from imwarp returns the matrix product A.dot(B) considering None 
    as the identity
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
    to input and output images, and finaly the difference between their positions
    is taken as the displacement vector. The resulting displacement, although 
    operating in physical space, maps the points exactly (up to numerical 
    precision).
    """
    input_shape = (10, 10)
    target_shape = (10, 10)
    #create a simple affine transformation
    nr = input_shape[0]
    nc = input_shape[1]
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
    gt_affine_inv = np.linalg.inv(gt_affine)

    #create the random displacement field
    input_affine = gt_affine
    target_affine = gt_affine
    disp, assign = vfu.create_random_displacement_2d(np.array(input_shape, dtype=np.int32),
                                                     input_affine, 
                                                     np.array(target_shape, dtype=np.int32),
                                                     target_affine)
    disp = np.array(disp, dtype=floating)
    assign = np.array(assign)
    #create a random image (with decimal digits) to warp
    moving_image = np.ndarray(target_shape, dtype=floating)
    moving_image[...] = np.random.randint(0, 10, np.size(moving_image)).reshape(tuple(target_shape))
    #set boundary values to zero so we don't test wrong interpolation due to floating point precision
    moving_image[0,:] = 0
    moving_image[-1,:] = 0
    moving_image[:,0] = 0
    moving_image[:,-1] = 0

    #warp the moving image using the synthetic displacement field
    target_affine_inv = np.linalg.inv(target_affine)
    input_affine_inv = np.linalg.inv(input_affine)
    affine_index = target_affine_inv.dot(input_affine)
    affine_disp = target_affine_inv

    #warp the moving image using the (exact) assignments
    expected = moving_image[(assign[...,0], assign[...,1])]

    #warp using a DiffeomorphicMap instance
    diff_map = imwarp.DiffeomorphicMap(2, input_shape, input_affine, 
                                          input_shape, input_affine, 
                                          target_shape, target_affine, 
                                          None)
    diff_map.forward = disp

    #Verify that the transform method accepts different image types (note that
    #the actual image contained integer values, we don't want to test rounding)
    for type in [floating, np.float64, np.int64, np.int32]:
        moving_image = moving_image.astype(type)

        #warp using linear interpolation
        warped = diff_map.transform(moving_image, 'linear')
        #compare the images (the linear interpolation may introduce slight precision errors)
        assert_array_almost_equal(warped, expected, decimal=5)

        #Now test the nearest neighbor interpolation
        warped = diff_map.transform(moving_image, 'nearest')
        #compare the images (now we dont have to worry about precision, it is n.n.)
        assert_array_almost_equal(warped, expected)

        #verify the is_inverse flag
        inv = diff_map.inverse()
        warped = inv.transform_inverse(moving_image, 'linear')
        assert_array_almost_equal(warped, expected, decimal=5)

        warped = inv.transform_inverse(moving_image, 'nearest')
        assert_array_almost_equal(warped, expected)

    #Now test the inverse functionality
    diff_map = imwarp.DiffeomorphicMap(2, target_shape, target_affine,
                                          target_shape, target_affine, 
                                          input_shape, input_affine, None)
    diff_map.backward = disp
    for type in [floating, np.float64, np.int64, np.int32]:
        moving_image = moving_image.astype(type)

        #warp using linear interpolation
        warped = diff_map.transform_inverse(moving_image, 'linear')
        #compare the images (the linear interpolation may introduce slight precision errors)
        assert_array_almost_equal(warped, expected, decimal=5)

        #Now test the nearest neighbor interpolation
        warped = diff_map.transform_inverse(moving_image, 'nearest')
        #compare the images (now we dont have to worry about precision, it is n.n.)
        assert_array_almost_equal(warped, expected)

    #Verify that DiffeomorphicMap raises the appropriate exceptions when
    #the sampling information is undefined
    diff_map = imwarp.DiffeomorphicMap(2, input_shape, input_affine,
                                          input_shape, input_affine, 
                                          target_shape, target_affine, 
                                          None)
    diff_map.forward = disp
    diff_map.domain_shape = None
    #If we don't provide the sampling info, it should try to use the map's info, but it's None...
    assert_raises(ValueError, diff_map.transform, moving_image, 'linear')

    #Same test for diff_map.transform_inverse
    diff_map = imwarp.DiffeomorphicMap(2, input_shape, input_affine,
                                          input_shape, input_affine, 
                                          target_shape, target_affine, 
                                          None)
    diff_map.forward = disp
    diff_map.codomain_shape = None
    #If we don't provide the sampling info, it should try to use the map's info, but it's None...
    assert_raises(ValueError, diff_map.transform_inverse, moving_image, 'linear')


def test_optimizer_exceptions():
    #An arbitrary valid metric
    metric = metrics.SSDMetric(2)
    # The metric must not be None
    assert_raises(ValueError, imwarp.SymmetricDiffeomorphicRegistration, None)
    # The iterations list must not be empty
    assert_raises(ValueError, imwarp.SymmetricDiffeomorphicRegistration, metric, [])

    optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric, None)
    #Verify the default iterations list
    assert_array_equal(optimizer.level_iters, [100,100,25])

    #Verify exception thrown when attepting to fit the energy profile without enough data
    assert_raises(ValueError, optimizer._get_energy_derivative)


def test_scale_space_exceptions():
    target_shape = (32, 32)

    #create a random image
    image = np.ndarray(target_shape, dtype=floating)
    image[...] = np.random.randint(0, 10, np.size(image)).reshape(tuple(target_shape))
    zeros = (image == 0).astype(np.int32)

    ss = imwarp.ScaleSpace(image,3)

    for invalid_level in [-1, 3, 4]:
        assert_raises(ValueError, ss.get_image, invalid_level)

    # Verify that the mask is correctly applied, when requested
    ss = imwarp.ScaleSpace(image,3, mask0=True)
    for level in range(3):
        img = ss.get_image(level)
        z = (img == 0).astype(np.int32)
        assert_array_equal(zeros, z)


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

    moving = plt.imread(fname_moving)
    static = plt.imread(fname_static)
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

    subsampled_energy_profile = np.array(optimizer.full_energy_profile[::10])
    if floating is np.float32:
        expected_profile = \
            np.array([ 312.6813333 ,  164.59050263,  103.73623002,   82.1164849,
                       63.31888794,   57.02372298,   48.88254136,   45.4015576 ,
                       42.45817589,  174.94422108,   92.43030985,   58.73123347,
                       43.70869018,   15.79207659,   20.30039959,   41.99069232,
                       37.1587315 ,   33.1963267 ,   32.89163671,   87.82289011,
                       78.28761195])
    else:
        expected_profile = \
            np.array([ 312.68133361,  164.59049075,  103.73635218,  82.11638224,
                       63.3188368 ,   57.02375694,   48.88245596,   45.4014475 ,
                       42.4579966 ,  174.94167955,   92.42725191,   58.72655199,
                       43.71955268,   15.78579491,   20.45497118,   41.92597862,
                       37.60531526,   33.25877969,   30.638574  ,   91.49825032,
                       80.524506  ])
    assert_array_almost_equal(subsampled_energy_profile, expected_profile)
    assert_equal(optimizer.OPT_START_CALLED, 1)
    assert_equal(optimizer.OPT_END_CALLED, 1)
    assert_equal(optimizer.SCALE_START_CALLED, 1)
    assert_equal(optimizer.SCALE_END_CALLED, 1)
    assert_equal(optimizer.ITER_START_CALLED, 1)
    assert_equal(optimizer.ITER_END_CALLED, 1)



def test_ssd_2d_gauss_newton():
    r'''
    Classical Circle-To-C experiment for 2D Monomodal registration. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of SSD in
    2D using the Gauss Newton step, and this test checks that the current energy 
    profile matches the saved one.
    '''
    fname_moving = get_data('reg_o')
    fname_static = get_data('reg_c')

    moving = plt.imread(fname_moving)
    static = plt.imread(fname_static)
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
    mapping = optimizer.optimize(static, moving, np.eye(3), np.eye(3), np.eye(3))
    m = optimizer.get_map()
    assert_equal(mapping, m)
    subsampled_energy_profile = np.array(optimizer.full_energy_profile[::10])
    if floating is np.float32:
        expected_profile = \
            np.array([ 312.68133316,   79.81322732,   28.37105316,   24.3985506,
                       13.92768078,   11.52267765,    9.11339687,   27.28819896,
                       42.9770759 ,  237.44444211,  153.43258717,  137.2169711])
    else:
        expected_profile = \
            np.array([ 312.68133361,   79.8132289 ,   27.28523819,  24.22883738,
                       56.71942103,   30.20320996,   19.4766414 ,   74.72561337,
                       108.0512537 ,  106.37445697])
    assert_array_almost_equal(subsampled_energy_profile, expected_profile)
    assert_equal(optimizer.OPT_START_CALLED, 0)
    assert_equal(optimizer.OPT_END_CALLED, 0)
    assert_equal(optimizer.SCALE_START_CALLED, 0)
    assert_equal(optimizer.SCALE_END_CALLED, 0)
    assert_equal(optimizer.ITER_START_CALLED, 0)
    assert_equal(optimizer.ITER_END_CALLED, 0)


def get_synthetic_warped_circle(nslices):
    #get a subsampled circle
    fname_cicle = get_data('reg_o')
    circle = plt.imread(fname_cicle)[::4,::4].astype(floating)
    
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


def test_ssd_3d_demons():
    r'''
    Register a stack of circles ('cylinder') before and after warping them with 
    a synthetic diffeomorphism. This test is intended to detect regressions
    only: we saved the energy profile (the sequence of energy values at each
    iteration) of a working version of SSD in 3D using the Demons step, and this
    test checks that the current energy profile matches the saved one. The
    validation of the "working version" was done by registering the 18 manually
    annotated T1 brain MRI database IBSR with each other and computing the
    jaccard index for all 31 common anatomical regions. 
    '''
    moving, static = get_synthetic_warped_circle(20)

    #Create the SSD metric
    smooth = 4
    step_type = 'demons'
    similarity_metric = metrics.SSDMetric(3, smooth=smooth, step_type=step_type) 

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
    energy_profile = np.array(optimizer.full_energy_profile)
    if floating is np.float32:
        expected_profile = \
            np.array([403.94219823, 203.36887564, 77.61427804, 56.98867467,
                  53.92329855, 58.89465317, 52.23391251, 50.99499898,
                  54.4620484, 56.38681749, 704.77163671, 545.9035756,
                  474.77734192, 408.73742241, 390.20879639])
    else:
        expected_profile = \
            np.array([403.9422303, 203.12101711, 77.52115348, 56.8696255, 54.50423221,
                      57.72650804, 52.62548123, 51.11063871, 54.1024288, 55.97184169,
                      702.32079572, 545.71667269, 474.51380596, 409.31399792,
                      390.01525582])
    assert_array_almost_equal(energy_profile, expected_profile, decimal=6)


def test_ssd_3d_gauss_newton():
    r'''
    Register a stack of circles ('cylinder') before and after warping them with 
    a synthetic diffeomorphism. This test is intended to detect regressions 
    only: we saved the energy profile (the sequence of energy values at each
    iteration) of a working version of SSD in 3D using the Gauss-Newton step,
    and this test checks that the current energy profile matches the saved
    one. The validation of the "working version" was
    done by registering the 18 manually annotated T1 brain MRI database IBSR 
    with each other and computing the jaccard index for all 31 common anatomical
    regions. 
    '''
    moving, static = get_synthetic_warped_circle(20)

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
    energy_profile = np.array(optimizer.full_energy_profile)
    if floating is np.float32:
        expected_profile = \
            np.array([403.94219818, 169.79534926, 63.69844891, 59.77780109,
                      64.64768902, 65.67809412, 72.57286657, 67.17178574,
                      75.60076973, 68.59294582, 789.49262601, 497.96409176,
                      316.83502919, 251.67253372, 229.54621757])
    else:
        expected_profile = \
            np.array([403.9422303, 169.79534488, 63.69845074, 59.77779426,
                      64.64769534, 65.67809432, 72.57287153, 67.1717859,
                      75.60077366, 68.59294658, 789.49268083, 497.96409147,
                      316.83505709, 251.67253329, 229.54622582])
    assert_array_almost_equal(energy_profile, expected_profile, decimal=6)


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
    level_iters = [40, 20, 10]
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric, level_iters)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)
    energy_profile = np.array(optimizer.full_energy_profile)
    if floating is np.float32:
        expected_profile = \
            [-444.8859258, -480.07839779, -476.52641875, -494.59864627,
             -468.50373273, -499.9001642, -466.9645747, -500.52084191,
             -479.60298025, -500.04943303, -480.67570611, -501.83191339,
             -483.24095212, -501.55135369, -479.04193767, -499.32692892,
             -480.18538286, -499.11852839, -1182.09237492, -1273.11955362,
             -1308.57535496, -1315.41962987, -1255.00880671, -1310.82949309,
             -1288.12459871, -1307.61656512, -1290.1064075, -1319.83693108,
             -1301.69989167, -1309.06049878, -1290.52963668, -1319.06339573,
             -1291.83740764, -1313.95351987, -1293.66093869, -1324.17976231,
             -2648.89452177, -2763.11657576, -2779.09991839, -2781.24031821,
             -2788.13001868, -2781.79589833, -2801.98841105, -2804.14700532,
             -2791.12485719, -2802.90185029]
    else:
        expected_profile = \
            [-444.88592666, -482.07839337, -476.52642889, -494.59862985, 
             -470.50374775, -499.90014736, -468.96455418, -502.52084289,
             -481.60297606, -502.04943271, -482.67570438, -501.83191949,
             -483.24095619, -501.55135648, -483.04193618, -501.32692871,
             -482.1853888, -501.11852656, -1186.09212048, -1299.71514399,
             -1320.31249747, -1290.89361373, -1326.22103835, -1289.98002706,
             -1328.45888701, -1274.5623379, -1331.11998209, -1282.70494694,
             -1324.07591508, -1289.88003787, -1328.38444015, -1302.70671235,
             -1325.28389325, -1301.67699475, -1324.16741283, -1300.47921775,
             -1326.33796673, -1294.12591978, -2740.27517211, -2781.36940826,
             -2759.02192129, -2784.64067034, -2768.97542969, -2797.68969526,
             -2771.82843971, -2796.64238586, -2772.80475357, -2807.61178147]
    expected_profile = np.asarray(expected_profile)
    assert_array_almost_equal(energy_profile, expected_profile)


def test_cc_3d():
    r'''
    Register a stack of circles ('cylinder') before and after warping them with 
    a synthetic diffeomorphism. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of CC in
    3D, and this test checks that the current energy profile matches the saved
    one. The validation of the "working version" was
    done by registering the 18 manually annotated T1 brain MRI database IBSR 
    with each other and computing the jaccard index for all 31 common anatomical
    regions. The "working version" of CC in 3D obtains very similar results as
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
    level_iters = [20, 10, 5]
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
    energy_profile = np.array(optimizer.full_energy_profile)*1e-4
    if floating is np.float32:
        expected_profile = \
            [-0.238303, -0.23880914, -0.23366084, -0.24587241, -0.25002449,
             -0.24883956, -0.24821317, -0.24855679, -0.24767354, -0.24855821,
             -0.24755259, -0.24854851, -0.24752999, -0.24863215, -0.24756772,
             -0.24871609, -1.33053237, -1.40720239, -1.45474453, -1.48024919,
             -1.46278896, -1.48172018, -1.48258362, -1.47398067, -1.47501424,
             -1.47884117, -6.3944489, -6.43456439, -6.4186091, -6.39643279,
             -6.37781429]
    else:
        expected_profile = \
            [-0.238303, -0.23880914, -0.23366084, -0.24587241, -0.2500245,
             -0.24883956, -0.24821317, -0.24855679, -0.24767354, -0.24855821,
             -0.24755259, -0.24854851, -0.24752999, -0.24863215, -0.24756772,
             -0.24871609, -1.32333246, -1.40720244, -1.4547445, -1.48024954,
             -1.46278776, -1.48171975, -1.48259097, -1.47398431, -1.47502077,
             -1.47885007, -6.39237887, -6.6268766, -6.81787186, -6.86739777,
             -6.87949771]
    expected_profile = np.asarray(expected_profile)
    assert_array_almost_equal(energy_profile, expected_profile, decimal=6)


def test_em_3d():
    r'''
    Register a stack of circles ('cylinder') before and after warping them with 
    a synthetic diffeomorphism. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of EM in
    3D, and this test checks that the current energy profile matches the saved
    one. The validation of the "working version" was
    done by registering the 18 manually annotated T1 brain MRI database IBSR 
    with each other and computing the jaccard index for all 31 common anatomical
    regions. The "working version" of EM in 3D obtains very similar results as
    those reported for ANTS on the same database. Any modification that produces
    a change in the energy profile should be carefully validated to ensure no 
    accuracy loss.
    '''
    moving, static = moving, static = get_synthetic_warped_circle(20)

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
    level_iters = [10, 5, 2]
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
    energy_profile = np.array(optimizer.full_energy_profile)*1e-3
    if floating is np.float32:
        expected_profile = \
            np.array([0.00435425, 0.0112303, 0.01024076, 0.02397494, 0.02247797,
                      0.01114549, 0.01023147, 0.0023124, 0.00180482, 0.0024268,
                      0.0934704, 0.0690733, 0.05560828, 0.0467435, 0.04082496,
                      1.33835227, 1.15010779])
    else:
        expected_profile = \
            np.array([0.00435425, 0.01123031, 0.01024076, 0.02397494, 0.02243585,
                      0.01109642, 0.01240944, 0.00363153, 0.0058803, 0.00469722,
                      0.08594618, 0.0692719, 0.05217823, 0.0446487, 0.04080833,
                      1.35778748, 1.20501598])
    assert_array_almost_equal(energy_profile, expected_profile, decimal=6)


def test_em_2d():
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
    step_length=0.25
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
    energy_profile = np.array(optimizer.full_energy_profile)
    if floating is np.float32:
        expected_profile = \
            [2.50773392e+00, 9.34571704e-01, 4.15081141e-01, 1.09400870e+00,
             1.19416716e+00, 8.02488104e-01, 6.92726938e-01, 1.67985164e-01,
             1.72585846e-01, 4.13782192e-01, 2.24417883e-01, 3.52777077e-01,
             2.41496389e-01, 1.58323708e-01, 2.23282229e-01, 3.52568734e-01,
             1.85037698e-02, 2.74443686e-01, 1.75231926e-01, 1.97625301e-01,
             9.96772866e-02, 3.24320404e-01, 9.25193599e-01, 3.62592183e-01,
             5.38317166e+00, 3.57994310e+00, 2.97754570e+00, 2.75208386e+00,
             2.27063864e+00, 1.88742856e+00, 2.16282526e+00, 1.74927644e+00,
             1.87649209e+00, 2.46089150e+00, 2.80669605e+00, 2.14988167e+00,
             2.44685809e+00, 3.02630703e+00, 2.77681275e+00, 5.89815872e+01,
             5.62737716e+01, 5.28780704e+01, 5.29825036e+01, 5.45278575e+01,
             5.35388263e+01, 5.10646548e+01, 4.57296130e+01, 4.63424135e+01,
             4.87388128e+01]
    else:
        expected_profile = \
            [2.50773436e+00, 9.34571717e-01, 4.15080394e-01, 1.09400690e+00,
             1.19416637e+00, 8.02473107e-01, 6.92686655e-01, 1.67976175e-01,
             1.72560749e-01, 4.13719504e-01, 2.05889592e-01, 2.39722355e-01,
             5.05173958e-02, 1.15651311e-01, 5.63989487e-01, 3.56484478e-01,
             3.93268881e-01, 5.59973587e-02, 4.66046091e-02, 1.57818367e-01,
             1.59689844e-01, 1.45536408e-01, 3.26460998e-01, 1.03794413e-01,
             1.59460925e-01, 3.18144982e-01, 1.09154920e-01, 1.43548858e-01,
             2.42523489e-01, 3.95769878e-01, 4.57346798e+00, 4.36374184e+00,
             1.64710981e+00, 1.63966712e+00, 1.78468834e+00, 1.67119942e+00,
             1.79750893e+00, 1.66704373e+00, 2.75698785e+00, 1.94695121e+00,
             1.73763389e+00, 2.05770309e+00, 3.36601243e+00, 2.34135378e+00,
             6.28258598e+01, 6.06442843e+01, 5.61887383e+01, 5.11311233e+01,
             5.01569515e+01, 4.94887470e+01, 4.50796361e+01, 4.40937727e+01,
             4.46509266e+01, 4.63176665e+01]
    assert_array_almost_equal(energy_profile, np.array(expected_profile))


def test_em_3d_demons():
    r'''
    Register a stack of circles ('cylinder') before and after warping them with 
    a synthetic diffeomorphism. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of EM in
    3D, and this test checks that the current energy profile matches the saved
    one. The validation of the "working version" was
    done by registering the 18 manually annotated T1 brain MRI database IBSR 
    with each other and computing the jaccard index for all 31 common anatomical
    regions. The "working version" of EM in 3D obtains very similar results as
    those reported for ANTS on the same database. Any modification that produces
    a change in the energy profile should be carefully validated to ensure no 
    accuracy loss.
    '''
    moving, static = moving, static = get_synthetic_warped_circle(20)

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
    level_iters = [10, 5, 2]
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
    energy_profile = np.array(optimizer.full_energy_profile)*1e-3
    if floating is np.float32:
        expected_profile = \
            np.array([4.35425306e-03, 3.41080923e-03, 5.68638232e-03,
                      8.49876602e-03, 3.44713359e-02, 8.04287771e-03,
                      6.63319925e-02, 3.18628773e-02, 5.59754920e-03,
                      2.51800195e-02, 2.38294092e-01, 2.12284541e-01,
                      1.89159004e-01, 1.85445524e-01, 2.10568869e-01,
                      4.58700183e+00, 4.65788895e+00])
    else:
        expected_profile = \
            np.array([4.35425305e-03, 3.41081111e-03, 5.68638229e-03,
                      8.49873360e-03, 3.44715416e-02, 8.04286633e-03,
                      6.63320214e-02, 3.18629431e-02, 5.59754886e-03,
                      2.51800141e-02, 2.38268293e-01, 2.15299085e-01,
                      1.98889632e-01, 2.04714053e-01, 2.01402902e-01,
                      4.45895513e+00, 4.59735285e+00])
    assert_array_almost_equal(energy_profile, expected_profile, decimal=6)


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
    step_length=0.25
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
    energy_profile = np.array(optimizer.full_energy_profile)
    if floating is np.float32:
        expected_profile = \
            [2.50773393, 3.73145783, 5.40504231, 2.40535998, 3.55940621,
             4.4547173, 5.49395654, 3.81368295, 2.13862014, 2.82826264,
             3.38060606, 3.78130601, 1.67064668, 3.38399353, 5.36355581,
             0.90105923, 38.18281603, 32.05473537, 25.13158733, 25.87968985,
             32.33134566, 29.48797557, 34.31334813, 32.71613773, 22.27334974,
             28.88625101, 26.34424393, 28.36712306, 30.93762419, 34.60418468,
             30.14407462, 179.69094699, 200.19956179, 201.27681693, 188.1983909,
             205.2271774, 197.86160568, 194.02926529, 201.00884584, 199.63156892,
             188.87939733]
    else:
        expected_profile = \
            [2.50773436, 3.73145924, 5.40505096, 2.40536245, 3.55940816,
             4.45472925, 5.4939637, 3.81367048, 2.13862585, 2.8282605,
             3.38060737, 3.78130285, 1.67064774, 3.3839928, 5.3635572,
             0.90106199, 38.18281287, 32.05476963, 25.13158948, 25.87968899,
             32.33134182, 28.46797943, 28.27341656, 26.19940397, 25.11322278,
             22.76430642, 27.08316401, 24.03250357, 27.41708423, 24.08874854,
             27.1707052, 30.44080513, 29.84499709, 184.33256832, 201.98870559,
             209.61305411, 226.29310316, 209.52645064, 192.75257157, 192.68933774,
             196.74400394, 183.32843833, 189.89789045]
    assert_array_almost_equal(energy_profile, np.array(expected_profile))

if __name__=='__main__':
    test_scale_space_exceptions()
    test_optimizer_exceptions()
    test_mult_aff()
    test_diffeomorphic_map_2d()
    test_get_direction_and_spacings()
    test_ssd_2d_demons()
    test_ssd_2d_gauss_newton()
    test_ssd_3d_demons()
    test_ssd_3d_gauss_newton()
    test_cc_2d()
    test_cc_3d()
    test_em_2d()
    test_em_3d()
    test_em_3d_demons()
    test_em_2d_demons()
