import numpy as np
from numpy.testing import (assert_equal,
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
    import dipy.align.cc as cc
    a = np.array(range(20*20), dtype = floating).reshape(20,20)
    b = np.array(range(20*20)[::-1], dtype = floating).reshape(20,20)
    factors = np.asarray(cc.precompute_cc_factors_2d(a,b,3))
    expected = np.asarray(cc.precompute_cc_factors_2d_test(a,b,3))
    assert_array_almost_equal(factors, expected)


def test_warping_2d():
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
                                                     np.array(input_affine).astype(floating), 
                                                     np.array(target_shape, dtype=np.int32),
                                                     np.array(target_affine).astype(floating))
    disp = np.array(disp)
    assign = np.array(assign)
    #create a random image (with decimal digits) to warp
    moving_image = np.ndarray(target_shape, dtype=floating)
    moving_image[...] = np.random.randint(0, 10, size(moving_image)).reshape(tuple(target_shape))
    #set boundary values to zero so we don't test wrong interpolation due to
    #floating point precision
    moving_image[0,:] = 0
    moving_image[-1,:] = 0
    moving_image[:,0] = 0
    moving_image[:,-1] = 0

    #warp the moving image using the synthetic displacement field
    target_affine_inv = np.linalg.inv(target_affine)
    input_affine_inv = np.linalg.inv(input_affine)
    affines = imwarp.compute_warping_affines(T_inv = target_affine_inv, 
                                             R = input_affine,
                                             R_inv = input_affine_inv, 
                                             A = None, 
                                             B = None)

    #apply the implementation under test
    warped = np.array(vfu.warp_image(moving_image, disp, affines[0], affines[1], affines[2]))

    #warp the moving image using the (exact) assignments
    expected = moving_image[(assign[...,0], assign[...,1])]

    #compare the images
    assert_array_almost_equal(warped, expected, decimal = 5)
    
    #Now test the nearest neighbor interpolation
    warped = np.array(vfu.warp_image_nn(moving_image, disp, affines[0], affines[1], affines[2]))
    #compare the images (now we dont have to worry about precision, it is n.n.)
    assert_array_almost_equal(warped, expected)


def test_compose_vector_fields():
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
                                                       np.array(input_affine).astype(floating), 
                                                       np.array(target_shape, dtype=np.int32),
                                                       np.array(target_affine).astype(floating))
    disp1 = np.array(disp1)
    assign1 = np.array(assign1)

    disp2, assign2 = vfu.create_random_displacement_2d(np.array(input_shape, dtype=np.int32),
                                                       np.array(input_affine).astype(floating), 
                                                       np.array(target_shape, dtype=np.int32),
                                                       np.array(target_affine).astype(floating))
    disp2 = np.array(disp2)
    assign2 = np.array(assign2)

    #create a random image (with decimal digits) to warp
    moving_image = np.ndarray(target_shape, dtype=floating)
    moving_image[...] = np.random.randint(0, 10, size(moving_image)).reshape(tuple(target_shape))
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

    premult_disp = input_affine_inv
    premult_index = input_affine_inv.dot(target_affine)
    
    composition, stats = vfu.compose_vector_fields_2d(disp1, disp2,
                                               np.array(premult_index, dtype=floating),
                                               np.array(premult_disp, dtype = floating),
                                               1.0)
    #apply the implementation under test
    affines = imwarp.compute_warping_affines(T_inv = target_affine_inv, 
                                             R = input_affine,
                                             R_inv = input_affine_inv, 
                                             A = None, 
                                             B = None)
    warped = np.array(vfu.warp_image(moving_image, composition, affines[0], affines[1], affines[2]))

    assert_array_almost_equal(warped, expected)
    

def test_compose_vector_fields_2d_easy():
    r"""
    Compose a constant vector field equal to (1,1) with itself. The result
    must be (2,2) everywhere except at the [n-1, :], [:, n-1] boundary. 
    The maximum and average norm must be 2*\sqrt(2) with standard deviation 0.
    """
    import dipy.align.vector_fields as vf
    d1 = np.ones(shape = (10,10,2), dtype = floating)
    #test using the identity
    identity = np.eye(3, dtype = floating)
    c, s = vf.compose_vector_fields_2d(d1, d1, identity, identity, 1.0)
    expected_c = np.ones_like(d1)
    expected_c *= 2
    expected_c[9,...] = 0
    expected_c[:,9,...] = 0
    assert_array_almost_equal(c, expected_c)
    expected_s = np.array([2*np.sqrt(2), 2*np.sqrt(2), 0])
    assert_array_almost_equal(expected_s, s)
    #now flip the axes
    flip = np.zeros((3,3), dtype = floating)
    flip_inv = np.zeros((3,3), dtype = floating)
    flip[0, 1] = 2
    flip[1, 0] = 2
    flip[2, 2] = 1
    flip_inv[0, 1] = 0.5
    flip_inv[1, 0] = 0.5
    flip_inv[2, 2] = 1
    #if the two affine matrices (for d1 and d2) are the same,
    #then the point pre-multiplication  is the identity
    c, s = vf.compose_vector_fields_2d(d1, d1, identity, flip_inv, 1.0)
    assert_array_almost_equal(c, expected_c)
    expected_s = np.array([2*np.sqrt(2), 2*np.sqrt(2), 0])
    assert_array_almost_equal(expected_s, s)
    #Now use flip as the second displacement matrix (provide the inverse
    #as second argument), and the identity as the first displacement matrix,
    #this means that the SECOND displacement is defined at a larger region, 
    #so we can evaluate the result of adding the displacements along all the d1
    #domain (i.e. with no zeros at the boundary)
    expected_c = np.ones_like(d1)
    expected_c *= 2
    c, s = vf.compose_vector_fields_2d(d1, d1, flip_inv, flip_inv, 1.0)
    assert_array_almost_equal(c, expected_c)
    expected_s = np.array([2*np.sqrt(2), 2*np.sqrt(2), 0])
    assert_array_almost_equal(expected_s, s)
    #Now, if we use flip_inv as the second displacement matrix (provide 
    #flip as the second argument), and the identity as the first 
    #displacement matrix, this means that the SECOND displacement is defined 
    #at a smaller region, so we'll have many more zeros at the boundary
    #(everywhere but at the upper-left quarer)
    expected_c = np.ones_like(d1)
    expected_c *= 2
    expected_c[4:,...] = 0
    expected_c[:,4:,...] = 0
    c, s = vf.compose_vector_fields_2d(d1, d1, flip, flip, 1.0)
    assert_array_almost_equal(c, expected_c)
    expected_s = np.array([2*np.sqrt(2), 2*np.sqrt(2), 0])
    assert_array_almost_equal(expected_s, s)


def test_prepend_affine_to_displacement_field_2d():
    r"""
    Apply a 90 degrees rotation to the displacement field 
    """
    import dipy.align.vector_fields as vf
    from dipy.align import floating
    d1 = np.ones(shape = (11, 11, 2), dtype = floating)
    aff = np.eye(3, dtype = floating)
    aff[0, 0] = 0
    aff[1, 1] = 0
    aff[0, 1] = -1
    aff[1, 0] = 1
    vf.prepend_affine_to_displacement_field_2d(d1, aff)


def test_ssd_2d():
    r'''
    Classical Circle-To-C experiment for 2D Monomodal registration
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
    #Configure and run the Optimizer
    smooth = 4
    inner_iter =5
    step_length = 0.25
    step_type = 0
    similarity_metric = metrics.SSDMetric(2, smooth, inner_iter, step_length, step_type) 
    opt_iter = [20, 100, 100, 100]
    opt_tol = 1e-4
    inv_iter = 40
    inv_tol = 1e-3
    registration_optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric, opt_iter, opt_tol, inv_iter, inv_tol)
    registration_optimizer.optimize(static, moving, None)
    subsampled_energy_profile = registration_optimizer.full_energy_profile[::10]
    if floating is np.float32:
        expected_profile = [302.6112060546875, 274.73687744140625, 258.37188720703125, 
                            242.0118865966797, 225.96595764160156, 212.8796844482422, 
                            200.59584045410156, 188.8486328125, 178.32041931152344, 
                            165.65579223632812, 702.3003540039062, 606.8388061523438, 
                            511.5794372558594, 417.9437255859375, 329.8865661621094, 
                            242.92117309570312, 165.19544982910156, 99.67949676513672, 
                            56.240074157714844, 39.08817672729492, 179.33363342285156, 
                            71.15731811523438, 51.66040802001953, 43.185237884521484, 
                            37.47501754760742, 34.42680358886719, 32.244903564453125, 
                            29.302459716796875, 28.516944885253906, 26.80443000793457]
    else:
        expected_profile = [302.61125317089767, 274.7369256624365, 258.3718071091768, 
                            242.01193676614497, 225.96598999638158, 212.87967285363396, 
                            200.5959806064401, 188.84863550148992, 178.3204633084462, 
                            165.6558070298394, 702.300837028873, 606.8390237201834, 
                            511.5795215789606, 417.944226511893, 329.88682685347106, 
                            242.92150013784828, 165.1957684235344, 99.67985374850804, 
                            56.24016825599313, 39.088227648263825, 179.33400779171248, 
                            71.15748591895019, 51.66042879906375, 43.18517211651795, 
                            37.47503071707744, 34.426881654216494, 32.24493906419912, 
                            29.302506040713634, 28.516894783752793, 26.804434032428883]
    print subsampled_energy_profile
    return
    assert_array_almost_equal(np.array(subsampled_energy_profile), np.array(expected_profile), decimal=6)


def test_cc_2d():
    r'''
    Classical Circle-To-C experiment for 2D Monomodal registration
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
    #Configure and run the Optimizer
    step_length = 0.25
    sigma_diff = 3.0
    radius = 4
    similarity_metric = metrics.CCMetric(2, step_length, sigma_diff, radius)

    opt_iter = [20, 100, 100, 100]
    opt_tol = 1e-4
    inv_iter = 40
    inv_tol = 1e-3
    registration_optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric, opt_iter, opt_tol, inv_iter, inv_tol)
    registration_optimizer.optimize(static, moving, None)
    subsampled_energy_profile = registration_optimizer.full_energy_profile[::10]
    if floating is np.float32:
        expected_profile = [-426.1901222185769, -944.5248469140995, -1002.3064082339106, 
                            -1040.1056075808694, -1062.2259196511095, -1082.8851555725555, 
                            -2509.5668030864754, -2540.3621632485692, -5528.721353559879]
    else:
        expected_profile = [-552.1893949392513, -944.6304790721674, -1002.6083646596371, 
                            -1040.583964107436, -1062.8579790173615, -1084.154634572068, 
                            -1110.642770914628, -2542.092851995484, -2666.7628581632466, 
                            -5667.78957088863]
    print subsampled_energy_profile
    return
    assert_array_almost_equal(np.array(subsampled_energy_profile), np.array(expected_profile), decimal=6)


def test_cc_factors_3d():
    import dipy.align.cc as cc
    a = np.array(range(20*20*20), dtype = floating).reshape(20,20,20)
    b = np.array(range(20*20*20)[::-1], dtype = floating).reshape(20,20,20)
    factors = np.asarray(cc.precompute_cc_factors_3d(a,b,3))
    expected = np.asarray(cc.precompute_cc_factors_3d_test(a,b,3))
    assert_array_almost_equal(factors, expected)


def test_compose_vector_fields_3d():
    r"""
    Compose a constant vector field equal to (1,1,1) with itself. The result
    must be (2,2,2) everywhere except at the [n-1, :, :], [:, n-1, :],
    [:, :, n-1] boundary. 
    The maximum and average norm must be 2*\sqrt(3) with standard deviation 0.
    """
    import dipy.align.vector_fields as vf
    d1 = np.ones(shape = (10, 10, 10, 3), dtype = floating)
    c, s = vf.compose_vector_fields_3d(d1, d1)
    expected_c = np.ones_like(d1)
    expected_c *= 2
    expected_c[9, ...] = 0
    expected_c[:, 9, ...] = 0
    expected_c[:, :, 9, ...] = 0
    assert_array_almost_equal(c, expected_c)
    expected_s = np.array([2*np.sqrt(3), 2*np.sqrt(3), 0])
    assert_array_almost_equal(expected_s, s)


def test_cc_3d():
    from dipy.data import read_sherbrooke_3shell

    img, gtab = read_sherbrooke_3shell()

    moving = np.array(img.get_data()[..., 0], dtype = floating)

    #Warp the S0 with a synthetic rotation
    degrees = np.array([2.0, 3.0, 4.0])
    angles = degrees * (np.pi/180.0)
    rotation = getRotationMatrix(angles).astype(floating)
    new_shape = np.array(moving.shape, dtype = np.int32)
    static = np.asarray(vfu.warp_volume_affine(moving, new_shape, rotation))

    #Create the CC metric
    step_length = 0.25
    sigma_diff = 3.0
    radius = 4
    similarity_metric = metrics.CCMetric(3, step_length, sigma_diff, radius)

    #Create the optimizer
    opt_iter = [5, 10, 10]
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    registration_optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric, opt_iter, opt_tol, inv_iter, inv_tol)
    registration_optimizer.optimize(static, moving, None)
    energy_profile = np.array(registration_optimizer.full_energy_profile)*1e-6
    if floating is np.float32:
        expected_profile = np.array([-0.01488042, -0.01790866, -0.01886851, 
                                     -0.01998733, -0.0205144,  -0.02071616,
                                     -0.02195101, -0.02269356, -0.02263611, 
                                     -0.02340408, -0.11016121, -0.13088377,
                                     -0.13368836, -0.13891167, -0.14116072, 
                                     -0.14432491, -0.14479642, -0.1473532,
                                     -0.14748695, -0.14979191])*1e-6
    else:
        expected_profile = np.array([-0.01488042, -0.01790866, -0.01886851, 
                                     -0.01998733, -0.0205144,  -0.02071616,
                                     -0.02195101, -0.02269356, -0.02263611, 
                                     -0.02340408, -0.11016121, -0.13089485,
                                     -0.13368353, -0.138887,   -0.14112689, 
                                     -0.14430802, -0.14483744, -0.14737834,
                                     -0.14744341, -0.14978013])*1e-6
    print energy_profile
    return
    assert_array_almost_equal(np.array(energy_profile), np.array(expected_profile), decimal=6)


def test_em_3d():
    from dipy.data import read_sherbrooke_3shell

    img, gtab = read_sherbrooke_3shell()

    moving = np.array(img.get_data()[..., 0], dtype = floating)

    #Warp the S0 with a synthetic rotation
    degrees = np.array([2.0, 3.0, 4.0])
    angles = degrees * (np.pi/180.0)
    rotation = getRotationMatrix(angles).astype(floating)
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
    iter_type='v_cycle'
    similarity_metric = metrics.EMMetric(
        3, smooth, inner_iter, step_length, q_levels, double_gradient, iter_type)

    #Create the optimizer
    opt_iter = [1, 5, 10]
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    registration_optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric, opt_iter, opt_tol, inv_iter, inv_tol)
    registration_optimizer.optimize(static, moving, None)
    energy_profile = registration_optimizer.full_energy_profile
    if floating is np.float32:
        expected_profile =[11.12615966796875, 8.084357261657715, 6.636898040771484, 
                            4.629724383354187, 4.004666566848755, 3.1289035081863403, 
                            2.2731465697288513, 1.8173362612724304, 2.061128258705139, 
                            1.6410276293754578, 31.634721755981445, 24.582207679748535, 
                            19.60957908630371, 15.937037467956543, 13.944169521331787]

    else:
        expected_profile =[11.126297989876795, 8.084506642727089, 6.636979472116404, 
                            4.62543551294909, 3.9926128517335844, 3.0231896806152454, 
                            1.929883720362989, 1.562734306076318, 2.069354258402535, 
                            2.044004912659469, 28.434427672995895, 22.07834272698154, 
                            17.817407211769005, 15.205636938768833, 13.310639093692913]
    print energy_profile
    return
    assert_array_almost_equal(np.array(energy_profile), np.array(expected_profile), decimal=6)

if __name__=='__main__':
    test_warping()
    # test_cc_factors_2d()
    # test_compose_vector_fields_2d()
    # test_prepend_affine_to_displacement_field_2d()
    # test_ssd_2d()
    # test_cc_2d()
    # test_cc_factors_3d()
    # test_compose_vector_fields_3d()
    # test_cc_3d()
    # test_em_3d()