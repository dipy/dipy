from __future__ import print_function
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
from dipy.align.imwarp import DiffeomorphicMap


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
    mapping = registration_optimizer.optimize(static, moving, None)
    subsampled_energy_profile = np.array(registration_optimizer.full_energy_profile[::10])
    if floating is np.float32:
        expected_profile = np.array([ 312.6813333 ,  164.59050263,  103.73623002,   82.1164849 ,
         63.31888794,   57.02372298,   48.88254136,   45.4015576 ,
         42.45817589,  174.94422108,   92.43030985,   58.73123347,
         43.70869018,   15.79207659,   20.30039959,   41.99069232,
         37.1587315 ,   33.1963267 ,   32.89163671,   87.82289011,
         78.28761195])
    else:
        expected_profile = np.array([ 312.68133361,  164.59049075,  103.73635218,   82.11638224,
         63.3188368 ,   57.02375694,   48.88245596,   45.4014475 ,
         42.4579966 ,  174.94167955,   92.42725191,   58.72655199,
         43.71955268,   15.78579491,   20.45497118,   41.92597862,
         37.60531526,   33.25877969,   30.638574  ,   91.49825032,
         80.524506  ])
    assert_array_almost_equal(np.array(subsampled_energy_profile), np.array(expected_profile))


def get_synthetic_warped_circle(nslices):
    #get a subsampled circle
    fname_cicle = get_data('reg_o')
    circle = plt.imread(fname_cicle)[::4,::4,0].astype(floating)
    
    #create a synthetic invertible map and warp the circle
    d, dinv = vfu.create_harmonic_fields_2d(64, 64, 0.1, 4)
    d = np.asarray(d, dtype = floating)
    dinv = np.asarray(dinv, dtype = floating)
    mapping = DiffeomorphicMap(2, (64, 64))
    mapping.forward, mapping.backward = d, dinv
    wcircle = mapping.transform(circle)

    if(nslices == 1):
        return circle, wcircle

    #normalize and form the 3d by piling slices
    circle = (circle-circle.min())/(circle.max() - circle.min())
    circle_3d = np.ndarray(circle.shape + (nslices,), dtype = floating)
    circle_3d[...] = circle[...,None]
    circle_3d[...,0] = 0
    circle_3d[...,-1] = 0

    #do the same with the warped circle
    wcircle = (wcircle-wcircle.min())/(wcircle.max() - wcircle.min())
    wcircle_3d = np.ndarray(wcircle.shape + (nslices,), dtype = floating)
    wcircle_3d[...] = wcircle[...,None]
    wcircle_3d[...,0] = 0
    wcircle_3d[...,-1] = 0

    return circle_3d, wcircle_3d


def test_ssd_3d():
    r'''
    Register a stack of circle and c images. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of SSD in
    3D, and this test checks that the current energy profile matches the saved
    one. The validation of the "working version" was
    done by registering the 18 manually annotated T1 brain MRI database IBSR 
    with each other and computing the jaccard index for all 31 common anatomical
    regions. 
    '''
    moving, static = get_synthetic_warped_circle(20)

    #Create the SSD metric
    smooth = 4
    inner_iter = 20
    step_type = 'demons'
    similarity_metric = metrics.SSDMetric(3, smooth, inner_iter, step_type) 

    #Create the optimizer
    opt_iter = [5, 10]
    step_length = 0.25
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.5
    registration_optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric, opt_iter, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    registration_optimizer.verbosity = 10
    mapping = registration_optimizer.optimize(static, moving, None)
    energy_profile = np.array(registration_optimizer.full_energy_profile)
    if floating is np.float32:
        expected_profile = np.array([  601.17342436,   468.94537817,   418.67267847,   393.0580613 ,
         367.8863422 ,   319.61865314,   272.3558511 ,   269.57838565,
         254.63664301,   266.9605625 ,  2541.47438277,  2033.988534  ,
        1779.69793906,  1693.11368711,  1653.95419258])
    else:
        expected_profile = np.array([  601.17344986,   468.97523898,   418.73047322,   393.0534384 ,
         367.80005903,   319.44987629,   272.62769902,   268.10394736,
         254.30487935,   267.7249719 ,  2547.05251526,  2035.19403818,
        1780.21839845,  1692.64443559,  1653.6224987 ])
    assert_array_almost_equal(np.array(energy_profile), np.array(expected_profile), decimal=6)


def test_cc_2d():
    r'''
    Register two slices from the Sherbrooke database. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of CC in
    2D, and this test checks that the current energy profile matches the saved
    one.
    '''

    moving, static = get_synthetic_warped_circle(1)

    #Configure the metric
    sigma_diff = 3.0
    radius = 4
    metric = metrics.CCMetric(2, sigma_diff, radius)

    #Configure and run the Optimizer
    opt_iter = [10, 20, 40]
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric, opt_iter)
    mapping = optimizer.optimize(static, moving, None)
    energy_profile = np.array(optimizer.full_energy_profile)
    
    if floating is np.float32:
        expected_profile = np.array([ -435.79559516,  -460.80739355,  -469.88508346,  -486.87396486,
        -486.04298263,  -484.30780055,  -489.19779192,  -484.44738633,
        -489.17020371,  -485.6637196 ,  -488.70801039,  -487.46399496,
        -489.71671264,  -488.09117139,  -490.42271222,  -488.27909614,
        -490.28857064,  -487.60445667,  -490.03035784,  -485.72591888,
        -490.60729319, -1260.19301574, -1327.14719131, -1309.49160837,
       -1342.19150863, -1356.90061164, -1275.25601701, -1317.07887913,
       -1343.0784944 , -1301.45605487, -1336.04013439, -1366.93546512,
       -1328.10275902, -1317.85372622, -1317.62486769, -1274.53697105,
       -1337.79152122, -2801.90904108, -2857.68596628, -2849.56767541,
       -2867.77931765, -2846.8404648 , -2875.67021308, -2851.85228212,
       -2879.43368375, -2861.36274169, -2889.69112071])
    else:
        expected_profile = np.array([ -435.7955967 ,  -460.80739935,  -469.88508352,  -486.87396919,
        -486.0429746 ,  -484.30780608,  -489.19779364,  -484.44739074,
        -489.17020447,  -485.66372153,  -488.7080131 ,  -487.46399372,
        -489.71671982,  -488.09117245,  -490.42271431,  -488.27909883,
        -490.28856556,  -487.60445041,  -490.03035556,  -485.72592274,
        -490.60729406, -1258.19305758, -1358.34000624, -1348.08308818,
       -1376.5332102 , -1361.61634539, -1371.62866869, -1354.9690168 ,
       -1356.56553571, -1365.8866856 , -1308.45095778, -1366.49097861,
       -1330.98891026, -1353.73575477, -2765.92375447, -2871.07572026,
       -2885.22181863, -2873.25158879, -2883.36175689, -2882.74507256,
       -2892.91338306, -2891.84375023, -2894.12822118, -2890.7756098 ])
    assert_array_almost_equal(np.array(energy_profile), np.array(expected_profile))


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
    moving, static = moving, static = get_synthetic_warped_circle(20)

    #Create the CC metric
    sigma_diff = 2.0
    radius = 4
    similarity_metric = metrics.CCMetric(3, sigma_diff, radius)

    #Create the optimizer
    opt_iter = [5, 10, 20]
    step_length = 0.25
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.5
    registration_optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric, opt_iter, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    mapping = registration_optimizer.optimize(static, moving, None)
    energy_profile = np.array(registration_optimizer.full_energy_profile)*1e-4
    if floating is np.float32:
        expected_profile = np.array([-0.23135541, -0.22171793, -0.23767394, -0.24236032, -0.22654608,
       -0.19675488, -0.24164528, -0.24076027, -0.22999321, -0.22685398,
       -0.20686259, -0.23939138, -0.24139779, -1.32298218, -1.37421899,
       -1.37280958, -1.38166606, -1.37794505, -1.38500984, -1.38071534,
       -1.37929357, -1.37501299, -1.38839658, -6.12090669, -6.19221629,
       -6.19314241, -6.13668367, -6.11476345])
    else:
        expected_profile = np.array([-0.23135541, -0.22171793, -0.23767394, -0.24236032, -0.22654608,
       -0.19675488, -0.24164527, -0.24076027, -0.22999321, -0.22685398,
       -0.20686259, -0.23939137, -0.24139779, -1.32178231, -1.37421862,
       -1.37280946, -1.38166568, -1.37794478, -1.38500996, -1.38071547,
       -1.37928428, -1.37501037, -1.38838905, -6.11733785, -6.4959287 ,
       -6.63564872, -6.6980932 , -6.74961869])
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
    opt_iter = [2, 5, 10]
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.5
    registration_optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric, opt_iter, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    mapping = registration_optimizer.optimize(static, moving, None)
    energy_profile = np.array(registration_optimizer.full_energy_profile)*1e-3
    if floating is np.float32:
        expected_profile = np.array([  1.43656316e-02,   1.60135407e-02,   1.65427418e-02,
         2.03429403e-02,   2.30559626e-02,   2.11856403e-02,
         1.42458258e-02,   8.64275515e-03,   8.33457708e-03,
         1.02794735e-02,   2.04303673e-01,   1.61742622e-01,
         1.63019783e-01,   1.48859322e-01,   1.48795532e-01,
         2.62520447e+01,   2.18337886e+01])
    else:
        expected_profile = np.array([  1.43656285e-02,   1.59238234e-02,   1.65779725e-02,
         1.98081392e-02,   1.94982952e-02,   2.15197208e-02,
         1.38253150e-02,   1.04429025e-02,   1.35234400e-02,
         9.21913082e-03,   2.11519503e-01,   1.86257761e-01,
         1.86538648e-01,   1.53222573e-01,   1.59835528e-01,
         2.67523821e+01,   1.94538666e+01])
    assert_array_almost_equal(np.array(energy_profile), np.array(expected_profile), decimal=6)


def test_em_2d():
    r'''
    Register two slices from the Sherbrooke database. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of CC in
    2D, and this test checks that the current energy profile matches the saved
    one.
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
    opt_iter = [10, 20, 40]
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric, opt_iter)
    mapping = optimizer.optimize(static, moving, None)
    energy_profile = np.array(optimizer.full_energy_profile)
    
    if floating is np.float32:
        expected_profile = np.array([    5.07521038,     3.85801816,     3.62394243,     3.34811198,
           3.00624034,     2.72018113,     2.63478592,     2.7936472 ,
           2.92318225,     2.6157892 ,     2.51011077,     2.62440899,
           2.68817791,     2.89652365,     2.50406319,     3.78818488,
           3.45390534,    99.18659401,    74.6229744 ,    64.09078789,
          69.01992416,    58.42120171,    69.20737267,    58.37458324,
          53.61078644,    66.57945728,    59.71329784,    54.3897047 ,
          45.40669155,    38.95014477,    40.46437645,    38.89031315,
          45.31602287,    49.25625038,    23.68274498,    27.68768597,
          20.4745121 ,  1406.37990189,  1262.33969498,  1139.53244781,
        1010.52179337,  1075.52775955,  1040.64184952,   982.98088455,
         896.89818954,   723.24193192,   634.29199982])
    else:
        expected_profile = np.array([   5.07521878,    3.85800199,    3.6239301 ,    3.3481309 ,
          3.01210333,    2.74506291,    2.53923509,    2.70308067,
          2.56536905,    2.43899028,    2.22859222,    2.20692294,
          2.44995661,    2.43933628,    3.71727011,    2.72983738,
        103.95299745,   82.85215148,   48.57418885,   42.20932002,
         43.36162648,   21.53416567,   22.7587274 ,   15.52892072,
         15.0085294 ,   13.50509862,   11.98182061,   15.26312404,
         11.44581538,   10.87602659,   11.69346951,   10.66371745,
         11.01152779,   11.11890406,   10.62128561,   11.92621734,
        722.5427506 ,  696.7656346 ,  944.74943684,  610.29530423,
        565.06123932,  503.72437706,  551.15546174,  530.49991948,
         56.54027481,   55.97779424])
    assert_array_almost_equal(np.array(energy_profile), np.array(expected_profile))


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
