import numpy as np
from numpy.testing import (assert_equal,
                           assert_almost_equal, 
                           assert_array_equal,
                           assert_array_almost_equal)
import matplotlib.pyplot as plt
import dipy.align.imwarp as imwarp
import dipy.align.metrics as metrics 
import dipy.align.vector_fields as vfu
from dipy.data import get_data
from dipy.align import floating
import nibabel as nib


def getRotationMatrix(angles):
    ca=np.cos(angles[0])
    cb=np.cos(angles[1])
    cg=np.cos(angles[2])
    sa=np.sin(angles[0])
    sb=np.sin(angles[1])
    sg=np.sin(angles[2])
    return np.array([[cb*cg,-ca*sg+sa*sb*cg,sa*sg+ca*sb*cg],
                     [cb*sg,ca*cg+sa*sb*sg,-sa*cg+ca*sb*sg],
                     [-sb,sa*cb,ca*cb]])


def test_cc_factors_2d():
    r"""
    Compares the output of the optimized function to compute the cross-
    correlation factors against a direct (not optimized, but less error prone)
    implementation. 
    """
    import dipy.align.cc as cc
    a = np.array(range(20*20), dtype = floating).reshape(20,20)
    b = np.array(range(20*20)[::-1], dtype = floating).reshape(20,20)
    factors = np.asarray(cc.precompute_cc_factors_2d(a,b,3))
    expected = np.asarray(cc.precompute_cc_factors_2d_test(a,b,3))
    assert_array_almost_equal(factors, expected)


def test_warping_2d():
    r"""
    Creates a random displacement field that exactly maps pixels from an input
    image to an output image. First a discrete random assignment between the 
    images is generated, then each pair of mapped points are transformed to
    the physical space by assigning a pair of arbitrary, fixed affine matrices
    to input and output images, and finaly the diference between their positions
    is taken as the displacement vector. The resulting displacement, although 
    operating in physical space, maps the points exactly (up to numerical 
    precision).
    """
    from dipy.align import floating
    import dipy.align.imwarp as imwarp
    import dipy.align.vector_fields as vfu
    from numpy.testing import (assert_array_almost_equal)
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
    #set boundary values to zero so we don't test wrong interpolation due to
    #floating point precision
    moving_image[0,:] = 0
    moving_image[-1,:] = 0
    moving_image[:,0] = 0
    moving_image[:,-1] = 0

    #warp the moving image using the synthetic displacement field
    target_affine_inv = np.linalg.inv(target_affine)
    input_affine_inv = np.linalg.inv(input_affine)
    affine_index = target_affine_inv.dot(input_affine)
    affine_disp = target_affine_inv

    #apply the implementation under test
    warped = np.array(vfu.warp_image(moving_image, disp, None, affine_index, affine_disp))

    #warp the moving image using the (exact) assignments
    expected = moving_image[(assign[...,0], assign[...,1])]

    #compare the images
    assert_array_almost_equal(warped, expected, decimal = 5)
    
    #Now test the nearest neighbor interpolation
    warped = np.array(vfu.warp_image_nn(moving_image, disp, None, affine_index, affine_disp))
    #compare the images (now we dont have to worry about precision, it is n.n.)
    assert_array_almost_equal(warped, expected)

    #test consolidation
    consolidated = vfu.consolidate_2d(disp, affine_index, affine_disp)
    warped = np.array(vfu.warp_image(moving_image, consolidated, None, None, None))
    assert_array_almost_equal(warped, expected, decimal = 5)
    warped = np.array(vfu.warp_image_nn(moving_image, consolidated, None, None, None))
    assert_array_almost_equal(warped, expected)


def test_warping_3d():
    r"""
    Creates a random displacement field that exactly maps pixels from an input
    image to an output image. First a discrete random assignment between the 
    images is generated, then each pair of mapped points are transformed to
    the physical space by assigning a pair of arbitrary, fixed affine matrices
    to input and output images, and finaly the diference between their positions
    is taken as the displacement vector. The resulting displacement, although 
    operating in physical space, maps the points exactly (up to numerical 
    precision).
    """
    from dipy.align import floating
    import dipy.align.imwarp as imwarp
    import dipy.align.vector_fields as vfu
    from numpy.testing import (assert_array_almost_equal)
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
    gt_affine_inv = np.linalg.inv(gt_affine)

    #create the random displacement field
    input_affine = gt_affine
    target_affine = gt_affine
    disp, assign = vfu.create_random_displacement_3d(np.array(input_shape, dtype=np.int32),
                                                     input_affine, 
                                                     np.array(target_shape, dtype=np.int32),
                                                     target_affine)
    disp = np.array(disp, dtype = floating)
    assign = np.array(assign)
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

    #warp the moving image using the synthetic displacement field
    target_affine_inv = np.linalg.inv(target_affine)
    input_affine_inv = np.linalg.inv(input_affine)
    affine_index = target_affine_inv.dot(input_affine)
    affine_disp = target_affine_inv

    #apply the implementation under test
    warped = np.array(vfu.warp_volume(moving_image, disp, None, affine_index, affine_disp))

    #warp the moving image using the (exact) assignments
    expected = moving_image[(assign[...,0], assign[...,1], assign[...,2])]

    #compare the images
    assert_array_almost_equal(warped, expected, decimal = 5)
    
    #Now test the nearest neighbor interpolation
    warped = np.array(vfu.warp_volume_nn(moving_image, disp, None, affine_index, affine_disp))
    #compare the images (now we dont have to worry about precision, it is n.n.)
    assert_array_almost_equal(warped, expected)

    #test consolidation
    consolidated = vfu.consolidate_3d(disp, affine_index, affine_disp)
    warped = np.array(vfu.warp_volume(moving_image, consolidated, None, None, None))
    assert_array_almost_equal(warped, expected, decimal = 5)
    warped = np.array(vfu.warp_volume_nn(moving_image, consolidated, None, None, None))
    assert_array_almost_equal(warped, expected)


def test_compose_vector_fields_2d():
    r"""
    Creates two random displacement field that exactly map pixels from an input
    image to an output image. The resulting displacements and their composition,
    although operating in physical space, map the points exactly (up to numerical 
    precision).
    """
    from dipy.align import floating
    import dipy.align.imwarp as imwarp
    import dipy.align.vector_fields as vfu
    from numpy.testing import (assert_array_almost_equal)
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
    gt_affine_inv = np.linalg.inv(gt_affine)

    #create two random displacement fields
    input_affine = gt_affine
    target_affine = gt_affine

    disp1, assign1 = vfu.create_random_displacement_2d(np.array(input_shape, dtype=np.int32),
                                                       input_affine, 
                                                       np.array(target_shape, dtype=np.int32),
                                                       target_affine)
    disp1 = np.array(disp1, dtype=floating)
    assign1 = np.array(assign1)

    disp2, assign2 = vfu.create_random_displacement_2d(np.array(input_shape, dtype=np.int32),
                                                       input_affine, 
                                                       np.array(target_shape, dtype=np.int32),
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
    input_affine_inv = np.linalg.inv(input_affine)

    target_affine_inv = np.linalg.inv(target_affine)
    input_affine_inv = np.linalg.inv(input_affine)
    premult_index = target_affine_inv.dot(input_affine)
    premult_disp = target_affine_inv
    
    composition, stats = vfu.compose_vector_fields_2d(disp1, disp2,
                                                      premult_index,
                                                      premult_disp,
                                                      1.0)
    #apply the implementation under test
    warped = np.array(vfu.warp_image(moving_image, composition, None, premult_index, premult_disp))
    assert_array_almost_equal(warped, expected)

    #test also using nearest neighbor interpolation
    warped = np.array(vfu.warp_image_nn(moving_image, composition, None, premult_index, premult_disp))
    assert_array_almost_equal(warped, expected)


def test_compose_vector_fields_3d():
    r"""
    Creates two random displacement field that exactly map pixels from an input
    image to an output image. The resulting displacements and their composition,
    although operating in physical space, map the points exactly (up to numerical 
    precision).
    """
    from dipy.align import floating
    import dipy.align.imwarp as imwarp
    import dipy.align.vector_fields as vfu
    from numpy.testing import (assert_array_almost_equal)
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
    gt_affine_inv = np.linalg.inv(gt_affine)

    #create two random displacement fields
    input_affine = gt_affine
    target_affine = gt_affine

    disp1, assign1 = vfu.create_random_displacement_3d(np.array(input_shape, dtype=np.int32),
                                                       input_affine, 
                                                       np.array(target_shape, dtype=np.int32),
                                                       target_affine)
    disp1 = np.array(disp1, dtype=floating)
    assign1 = np.array(assign1)

    disp2, assign2 = vfu.create_random_displacement_3d(np.array(input_shape, dtype=np.int32),
                                                       input_affine, 
                                                       np.array(target_shape, dtype=np.int32),
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
    input_affine_inv = np.linalg.inv(input_affine)

    target_affine_inv = np.linalg.inv(target_affine)
    input_affine_inv = np.linalg.inv(input_affine)
    premult_index = target_affine_inv.dot(input_affine)
    premult_disp = target_affine_inv
    
    composition, stats = vfu.compose_vector_fields_3d(disp1, disp2,
                                               premult_index,
                                               premult_disp,
                                               1.0)
    #apply the implementation under test
    warped = np.array(vfu.warp_volume(moving_image, composition, None, premult_index, premult_disp))
    assert_array_almost_equal(warped, expected)

    #test also using nearest neighbor interpolation
    warped = np.array(vfu.warp_volume_nn(moving_image, composition, None, premult_index, premult_disp))
    assert_array_almost_equal(warped, expected)

def test_ssd_2d():
    r'''
    Classical Circle-To-C experiment for 2D Monomodal registration. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of SSD in
    2D, and this test checks that the current energy profile matches the saved
    one.
    '''
    fname_moving = get_data('reg_o')
    fname_static = get_data('reg_c')

    moving = plt.imread(fname_moving)
    static = plt.imread(fname_static)
    moving = moving[:, :, 0].astype(floating)
    static = static[:, :, 0].astype(floating)
    moving = np.array(moving, dtype = floating)
    static = np.array(static, dtype = floating)
    moving = (moving-moving.min())/(moving.max() - moving.min())
    static = (static-static.min())/(static.max() - static.min())
    #Create the SSD metric
    smooth = 4
    inner_iter = 20
    step_type = 'demons'
    similarity_metric = metrics.SSDMetric(2, smooth, inner_iter, step_type) 

    #Configure and run the Optimizer
    opt_iter = [25, 50, 100, 200]
    step_length = 0.25
    opt_tol = 1e-4
    inv_iter = 40
    inv_tol = 1e-3
    ss_sigma_factor = 0.2
    registration_optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric, opt_iter, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    registration_optimizer.optimize(static, moving, None)
    subsampled_energy_profile = registration_optimizer.full_energy_profile[::10]
    if floating is np.float32:
        expected_profile = [312.68133330313657, 164.59050263180393, 103.73623002323137, 
                            82.11648490159492, 63.318887939731574, 57.02372298083006,
                            48.88254135529228, 45.4015576044856, 42.458175894395374,
                            174.94422108434813, 92.43030985225451, 58.73123346962446,
                            43.70869017524925, 15.792076588788733, 20.300399590591233,
                            41.990692318305285, 37.158731502418554, 33.196326703572694,
                            32.89163671093136]
    else:
        expected_profile = [312.68133361375715, 164.59049074753798, 103.736352184813,
                            82.11638224407756, 63.318836798898616, 57.023756943619546,
                            48.88245595553537, 45.40144749845953, 42.457996601384224,
                            174.94167955106752, 92.42725190908986, 58.72655198707222,
                            43.7195526751881, 15.785794912626038, 20.454971177705318,
                            41.925978619294945, 37.6053152641555, 33.258779694884204,
                            30.638574002639203]
    assert_array_almost_equal(np.array(subsampled_energy_profile), np.array(expected_profile))

def test_ssd_3d():
    r'''
    Register a B0 image against itself after a linear transformation. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of SSD in
    3D, and this test checks that the current energy profile matches the saved
    one. The validation of the "working version" was
    done by registering the 18 manually annotated T1 brain MRI database IBSR 
    with each other and computing the jaccard index for all 31 common anatomical
    regions. 
    '''
    from dipy.data import read_sherbrooke_3shell

    img, gtab = read_sherbrooke_3shell()

    moving = np.array(img.get_data()[..., 0], dtype = floating)

    #Warp the S0 with a synthetic rotation
    degrees = np.array([2.0, 3.0, 4.0])
    angles = degrees * (np.pi/180.0)
    rotation = getRotationMatrix(angles)
    new_shape = np.array(moving.shape, dtype = np.int32)
    static = np.asarray(vfu.warp_volume_affine(moving, new_shape, rotation))

    #Create the SSD metric
    smooth = 4
    inner_iter = 20
    step_type = 'demons'
    similarity_metric = metrics.SSDMetric(3, smooth, inner_iter, step_type) 

    #Create the optimizer
    opt_iter = [5, 10, 10]
    step_length = 0.25
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.5
    registration_optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric, opt_iter, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    registration_optimizer.optimize(static, moving, None)
    energy_profile = np.array(registration_optimizer.full_energy_profile)
    if floating is np.float32:
        expected_profile = np.array([89.90622614, 59.17855102, 42.74823811,
                                     32.03280596, 25.57576981, 20.53462398,
                                     17.31927042, 15.81617853, 15.58714397,
                                     15.79245671, 110.6181378, 86.68461346,
                                     73.85097571, 66.12964541, 62.44040318, 
                                     59.95437311, 58.41602419, 57.58576527,
                                     57.18839396, 57.22796409])
    else:
        expected_profile = np.array([89.90622638, 59.17855087, 42.74823914,
                                     32.03280675, 25.57577089, 20.53462478,
                                     17.31927101, 15.81617887, 15.58714426,
                                     15.79245702, 110.61813147, 86.68461034,
                                     73.85097358, 66.12964501, 62.44040252,
                                     59.9543727, 58.41602413, 57.58576517,
                                     57.18839377, 57.22796444])
    assert_array_almost_equal(np.array(energy_profile), np.array(expected_profile), decimal=6)


def test_cc_2d():
    r'''
    Register two slices from the Sherbrooke database. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of CC in
    2D, and this test checks that the current energy profile matches the saved
    one.
    '''
    from dipy.data import read_sherbrooke_3shell

    img, gtab = read_sherbrooke_3shell()

    data = np.array(img.get_data()[..., 0], dtype = floating)

    static = data[:,:,30]
    moving = data[:,:,33]

    moving = (moving-moving.min())/(moving.max() - moving.min())
    static = (static-static.min())/(static.max() - static.min())

    #Configure the metric
    sigma_diff = 3.0
    radius = 4
    metric = metrics.CCMetric(2, sigma_diff, radius)

    #Configure and run the Optimizer
    opt_iter = [25, 50, 100]
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric, opt_iter)
    mapping = optimizer.optimize(static, moving, None)
    subsampled_energy_profile = optimizer.full_energy_profile[::5]
    
    if floating is np.float32:
        expected_profile = [-980.5516704079644, -1072.261161206346, -1103.8685455764867,
                            -1117.6115363057233, -1113.3258340702514, -1119.0483736679764,
                            -1119.355168844521, -2568.897038646955, -2808.4292225881754,
                            -2913.4903113092396, -2953.2785987089596, -2989.5050840994772,
                            -3028.706810600274, -3075.9169639536844, -3084.7444879075288,
                            -3121.0133127512822, -3139.0963683496484]
    else:
        expected_profile = [-980.551647721365, -1072.2611770607534, -1103.8685374990507, 
                            -1117.6115406679749, -1113.3258036795055, -1119.0483485626748, 
                            -1119.3551281328932, -2568.897086240164, -2808.42923678202,
                            -2913.4903027936844, -2953.278656195647, -2989.505113986426, 
                            -3028.7068130016064, -3075.9169915865714, -3084.7445328739173, 
                            -3121.0133491212946, -3139.09637501727]
    assert_array_almost_equal(np.array(subsampled_energy_profile), np.array(expected_profile))


def test_cc_factors_3d():
    r"""
    Compares the output of the optimized function to compute the cross-
    correlation factors against a direct (not optimized, but less error prone)
    implementation. 
    """
    import dipy.align.cc as cc
    a = np.array(range(20*20*20), dtype = floating).reshape(20,20,20)
    b = np.array(range(20*20*20)[::-1], dtype = floating).reshape(20,20,20)
    factors = np.asarray(cc.precompute_cc_factors_3d(a,b,3))
    expected = np.asarray(cc.precompute_cc_factors_3d_test(a,b,3))
    assert_array_almost_equal(factors, expected)


def test_cc_3d():
    r'''
    Register a B0 image against itself after a linear transformation. This test
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
    from dipy.data import read_sherbrooke_3shell

    img, gtab = read_sherbrooke_3shell()

    moving = np.array(img.get_data()[..., 0], dtype = floating)

    #Warp the S0 with a synthetic rotation
    degrees = np.array([2.0, 3.0, 4.0])
    angles = degrees * (np.pi/180.0)
    rotation = getRotationMatrix(angles)
    new_shape = np.array(moving.shape, dtype = np.int32)
    static = np.asarray(vfu.warp_volume_affine(moving, new_shape, rotation))

    #Create the CC metric
    sigma_diff = 2.0
    radius = 4
    similarity_metric = metrics.CCMetric(3, sigma_diff, radius)

    #Create the optimizer
    opt_iter = [5, 10, 10]
    step_length = 0.25
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.5
    registration_optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric, opt_iter, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    registration_optimizer.optimize(static, moving, None)
    energy_profile = np.array(registration_optimizer.full_energy_profile)*1e-6
    if floating is np.float32:
        expected_profile = np.array([-0.01433875, -0.01651356, -0.01688335,
                                     -0.01792773, -0.01791432, -0.01861407,
                                     -0.01898663, -0.01958575, -0.01967341, 
                                     -0.02049394, -0.0640203,  -0.06774939,
                                     -0.07179282, -0.07436513, -0.07631286, 
                                     -0.07866266, -0.07978836, -0.08150028,
                                     -0.08202129, -0.08336826])
    else:
        expected_profile = np.array([-0.01433875, -0.01651356, -0.01688335,
                                     -0.01792773, -0.01791432, -0.01861407,
                                     -0.01898663, -0.01958575, -0.01967341,
                                     -0.02049394, -0.06402031, -0.0677494,
                                     -0.07179282, -0.07436513, -0.07631286,
                                     -0.07866266, -0.07978836, -0.08150029,
                                     -0.08202129, -0.08336826])
    assert_array_almost_equal(np.array(energy_profile), np.array(expected_profile), decimal=6)


def test_em_3d():
    r'''
    Register a B0 image against itself after a linear transformation. This test
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
    from dipy.data import read_sherbrooke_3shell

    img, gtab = read_sherbrooke_3shell()

    moving = np.array(img.get_data()[..., 0], dtype = floating)

    #Warp the S0 with a synthetic rotation
    degrees = np.array([2.0, 3.0, 4.0])
    angles = degrees * (np.pi/180.0)
    rotation = getRotationMatrix(angles)
    new_shape = np.array(moving.shape, dtype = np.int32)
    static = np.asarray(vfu.warp_volume_affine(moving, new_shape, rotation))
    moving = (moving - moving.min())/(moving.max() - moving.min())
    static = (static -static.min())/ (static.max() - static.min())
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
    opt_iter = [1, 5, 10]
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.5
    registration_optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric, opt_iter, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    registration_optimizer.optimize(static, moving, None)
    energy_profile = registration_optimizer.full_energy_profile
    if floating is np.float32:
        expected_profile = [66.8847885131836, 54.29118347167969, 40.54933738708496, 
                            27.369705200195312, 18.024410247802734, 10.493958950042725,
                            6.5725555419921875, 3.5759841203689575, 3.182631254196167,
                            2.946118712425232, 28.103188514709473, 22.814631462097168,
                            18.84000587463379, 15.05379867553711, 12.480103492736816]
    else:
        expected_profile = [66.8861055278503, 54.291890244496564, 40.55012047483241,
                            27.37006218812411, 18.02331933450858, 10.49541670206046,
                            6.596505536350229, 3.567458641841514, 3.198956972821441,
                            2.920064943954684, 29.525329153333253, 24.402373025155605, 
                            20.027218533112965, 15.817854037365548, 13.503856639081103]
    assert_array_almost_equal(np.array(energy_profile), np.array(expected_profile), decimal=6)


def test_em_2d():
    r'''
    Register two slices from the Sherbrooke database. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of CC in
    2D, and this test checks that the current energy profile matches the saved
    one.
    '''
    from dipy.data import read_sherbrooke_3shell

    img, gtab = read_sherbrooke_3shell()

    data = np.array(img.get_data()[..., 0], dtype = floating)

    static = data[:,:,30]
    moving = data[:,:,33]

    moving = (moving-moving.min())/(moving.max() - moving.min())
    static = (static-static.min())/(static.max() - static.min())

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
    opt_iter = [25, 50, 100]
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric, opt_iter)
    mapping = optimizer.optimize(static, moving, None)
    subsampled_energy_profile = optimizer.full_energy_profile[::2]
    
    if floating is np.float32:
        expected_profile = [9.29666519165039, 8.840325832366943, 8.403895616531372,
                            9.217527627944946, 8.205434799194336, 8.004725933074951,
                            7.5680084228515625, 7.995570421218872, 8.856264591217041,
                            45.529253005981445, 44.46803855895996, 40.56655502319336,
                            39.50901794433594, 36.85078048706055, 35.39692401885986,
                            34.85632801055908, 35.41213321685791, 32.592201232910156,
                            32.75309371948242, 32.96829891204834, 33.156304359436035,
                            32.72635364532471, 31.77465057373047, 31.539039611816406,
                            32.992366790771484, 31.77255153656006, 32.799750328063965]
    else:
        expected_profile = [9.29666548784764, 8.840335993722759, 8.40389920311196,
                            9.217560274293554, 8.217516709578959, 7.9898629536621595,
                            7.818460722068478, 7.478561704964592, 7.698365075963448,
                            8.070323670985568, 41.743818149321484, 40.63616644077951,
                            37.31836508598798, 35.03585845856205, 34.657958343192,
                            34.40429714707601, 33.58123879840137, 33.753932257877516,
                            32.67009554981988, 34.2049633831725, 34.57927805161205,
                            33.954154589484304]
    assert_array_almost_equal(np.array(subsampled_energy_profile), np.array(expected_profile))


def test_invert_vector_field_2d():
    r"""
    Inverts a synthetic, analytically invertible, displacement field
    """
    d, dinv = vfu.create_harmonic_fields_2d(100, 100, 0.2, 8)
    d = np.asarray(d).astype(floating)
    dinv = np.asarray(dinv).astype(floating)
    
    inv_approx = vfu.invert_vector_field_fixed_point_2d(d, np.eye(3),
                                                        np.ones(2),
                                                        40, 1e-6)
    mapping = imwarp.DiffeomorphicMap(2, (100,100))
    mapping.forward = d
    mapping.backward = inv_approx
    residual, stats = mapping.compute_inversion_error()
    assert_almost_equal(stats[1], 0, decimal=4)
    assert_almost_equal(stats[2], 0, decimal=4)


def test_invert_vector_field_3d():
    r"""
    Inverts a synthetic, analytically invertible, displacement field
    """
    d, dinv = vfu.create_harmonic_fields_3d(100, 100, 100, 0.2, 8)
    d = np.asarray(d).astype(floating)
    dinv = np.asarray(dinv).astype(floating)
    
    inv_approx = vfu.invert_vector_field_fixed_point_3d(d, np.eye(4),
                                                        np.ones(3),
                                                        20, 1e-4)
    mapping = imwarp.DiffeomorphicMap(3, (100,100,100))
    mapping.forward = d
    mapping.backward = inv_approx
    residual, stats = mapping.compute_inversion_error()
    assert_almost_equal(stats[1], 0, decimal=3)
    assert_almost_equal(stats[2], 0, decimal=3)


if __name__=='__main__':
    #Unit tests
    test_warping_2d()
    test_warping_3d()
    test_compose_vector_fields_2d()
    test_compose_vector_fields_3d()
    test_invert_vector_field_2d()
    test_invert_vector_field_3d()
    test_cc_factors_2d()
    test_cc_factors_3d()
    #Integration tests
    test_ssd_2d()
    test_ssd_3d()
    test_cc_2d()
    test_cc_3d()
    test_em_2d()
    test_em_3d()
