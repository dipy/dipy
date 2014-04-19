import numpy as np
from dipy.align import floating
import dipy.align.vector_fields as vfu
from numpy.testing import (assert_array_almost_equal,
                           assert_almost_equal)


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


def test_invert_vector_field_2d():
    r"""
    Inverts a synthetic, analytically invertible, displacement field
    """
    import dipy.align.imwarp as imwarp

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
    import dipy.align.imwarp as imwarp

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
    test_warping_2d()
    test_warping_3d()
    test_compose_vector_fields_2d()
    test_compose_vector_fields_3d()
    test_invert_vector_field_2d()
    test_invert_vector_field_3d()
