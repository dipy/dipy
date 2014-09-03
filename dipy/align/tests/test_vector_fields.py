import numpy as np
from dipy.align import floating
import dipy.align.vector_fields as vfu
from numpy.testing import (assert_array_almost_equal,
                           assert_almost_equal)
import dipy.align.imwarp as imwarp


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
    np.random.seed(8315759)
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
    disp, assign = vfu.create_random_displacement_2d(np.array(input_shape,
                                                     dtype=np.int32),
                                                     input_affine, 
                                                     np.array(target_shape,
                                                     dtype=np.int32),
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
    warped = np.array(vfu.warp_2d(moving_image, disp, None, affine_index,
                                     affine_disp))

    #warp the moving image using the (exact) assignments
    expected = moving_image[(assign[...,0], assign[...,1])]

    #compare the images
    assert_array_almost_equal(warped, expected, decimal=5)
    
    #Now test the nearest neighbor interpolation
    warped = np.array(vfu.warp_2d_nn(moving_image, disp, None, affine_index,
                      affine_disp))
    #compare the images (now we dont have to worry about precision, it is n.n.)
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
    np.random.seed(8315759)
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
    disp, assign = vfu.create_random_displacement_3d(np.array(input_shape,
                                                     dtype=np.int32),
                                                     input_affine, 
                                                     np.array(target_shape,
                                                     dtype=np.int32),
                                                     target_affine)
    disp = np.array(disp, dtype=floating)
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
    warped = np.array(vfu.warp_3d(moving_image, disp, None, affine_index,
                                      affine_disp))

    #warp the moving image using the (exact) assignments
    expected = moving_image[(assign[...,0], assign[...,1], assign[...,2])]

    #compare the images
    assert_array_almost_equal(warped, expected, decimal=5)
    
    #Now test the nearest neighbor interpolation
    warped = np.array(vfu.warp_3d_nn(moving_image, disp, None, affine_index,
                                         affine_disp))
    #compare the images (now we dont have to worry about precision, it is n.n.)
    assert_array_almost_equal(warped, expected)


def test_affine_warping_2d():
    r"""
    Affine (and invertible) warping can be performed using a Diffeomorphic Map
    with zero displacement fields. We test affine warping by verifying that
    the result of the warping under the DiffeomorphicMap is equivalent to
    call the warping affine cython routine with the product of the
    DiffeomorphicMap's transforms. Note that if this case fails, either
    DiffeomorphicMap is wrong or the affine warping is wrong (or both),
    but the intention of this test is to detect regressions only.
    """
    # Create a simple invertible affine transform
    domain_shape = (64, 64)
    codomain_shape = (80, 80)
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
    # Create an image of a circle
    radius = 16
    circle = vfu.create_circle(codomain_shape[0], codomain_shape[1], radius)
    circle = np.array(circle, dtype = floating)
    #Define different voxel-to-space transforms for domain, codomain and reference grid,
    #also, use a non-identity pre-align transform
    D = gt_affine
    C = imwarp.mult_aff(gt_affine, gt_affine)
    R = np.eye(3)
    P = gt_affine
    
    #Create the diffeomorphic map
    diff_map = imwarp.DiffeomorphicMap(2, domain_shape, R,
                                          domain_shape, D, 
                                          codomain_shape, C, 
                                          P)
    #Assign zero displacements
    diff_map.forward = np.zeros(shape=domain_shape+(2,), dtype = floating)
    diff_map.backward = np.zeros(shape=domain_shape+(2,), dtype = floating)

    # Test affine warping with bilinear interpolation
    expected = diff_map.transform(circle, 'linear')
    composition = imwarp.mult_aff(diff_map.codomain_affine_inv, 
                                  imwarp.mult_aff(P, D))
    warped = vfu.warp_2d_affine(circle, np.array(domain_shape, dtype = np.int32), composition)
    assert_array_almost_equal(warped, expected)

    # Test affine warping with nearest-neighbor interpolation
    expected = diff_map.transform(circle, 'nearest')
    composition = imwarp.mult_aff(diff_map.codomain_affine_inv, 
                                  imwarp.mult_aff(P, D))
    warped = vfu.warp_2d_affine_nn(circle, np.array(domain_shape, dtype = np.int32), composition)
    assert_array_almost_equal(warped, expected)


def test_affine_warping_3d():
    r"""
    Affine (and invertible) warping can be performed using a Diffeomorphic Map
    with zero displacement fields. We test affine warping by verifying that
    the result of the warping under the DiffeomorphicMap is equivalent to
    call the warping affine cython routine with the product of the
    DiffeomorphicMap's transforms. Note that if this case fails, either
    DiffeomorphicMap is wrong or the affine warping is wrong (or both),
    but the intention of this test is to detect regressions only.
    """
    # Create a simple invertible affine transform
    domain_shape = (64, 64, 64)
    codomain_shape = (80, 80, 80)
    ns = domain_shape[0]
    nr = domain_shape[1]
    nc = domain_shape[2]
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
    # Create an image of a circle
    radius = 16
    sphere = vfu.create_sphere(codomain_shape[0], codomain_shape[1], codomain_shape[2], radius)
    sphere = np.array(sphere, dtype = floating)
    #Define different voxel-to-space transforms for domain, codomain and reference grid,
    #also, use a non-identity pre-align transform
    D = gt_affine
    C = imwarp.mult_aff(gt_affine, gt_affine)
    R = np.eye(4)
    P = gt_affine
    
    #Create the diffeomorphic map
    diff_map = imwarp.DiffeomorphicMap(3, domain_shape, R,
                                          domain_shape, D, 
                                          codomain_shape, C, 
                                          P)
    #Assign zero displacements
    diff_map.forward = np.zeros(shape=domain_shape+(3,), dtype = floating)
    diff_map.backward = np.zeros(shape=domain_shape+(3,), dtype = floating)

    # Test affine warping with trilinear interpolation
    expected = diff_map.transform(sphere, 'linear')
    composition = imwarp.mult_aff(diff_map.codomain_affine_inv, 
                                  imwarp.mult_aff(P, D))
    warped = vfu.warp_3d_affine(sphere, np.array(domain_shape, dtype = np.int32), composition)
    assert_array_almost_equal(warped, expected)

    # Test affine warping with nearest-neighbor interpolation
    expected = diff_map.transform(sphere, 'nearest')
    composition = imwarp.mult_aff(diff_map.codomain_affine_inv, 
                                  imwarp.mult_aff(P, D))
    warped = vfu.warp_3d_affine_nn(sphere, np.array(domain_shape, dtype = np.int32), composition)
    assert_array_almost_equal(warped, expected)


def test_compose_vector_fields_2d():
    r"""
    Creates two random displacement field that exactly map pixels from an input
    image to an output image. The resulting displacements and their composition,
    although operating in physical space, map the points exactly (up to
    numerical precision).
    """
    np.random.seed(8315759)
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

    disp1, assign1 = vfu.create_random_displacement_2d(np.array(input_shape,
                                                       dtype=np.int32),
                                                       input_affine, 
                                                       np.array(target_shape,
                                                       dtype=np.int32),
                                                       target_affine)
    disp1 = np.array(disp1, dtype=floating)
    assign1 = np.array(assign1)

    disp2, assign2 = vfu.create_random_displacement_2d(np.array(input_shape,
                                                       dtype=np.int32),
                                                       input_affine, 
                                                       np.array(target_shape,
                                                       dtype=np.int32),
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
                                                      1.0, None)
    #apply the implementation under test
    warped = np.array(vfu.warp_2d(moving_image, composition, None,
                                     premult_index, premult_disp))
    assert_array_almost_equal(warped, expected)

    #test also using nearest neighbor interpolation
    warped = np.array(vfu.warp_2d_nn(moving_image, composition, None,
                                        premult_index, premult_disp))
    assert_array_almost_equal(warped, expected)


def test_compose_vector_fields_3d():
    r"""
    Creates two random displacement field that exactly map pixels from an input
    image to an output image. The resulting displacements and their composition,
    although operating in physical space, map the points exactly (up to
    numerical precision).
    """
    np.random.seed(8315759)
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

    disp1, assign1 = vfu.create_random_displacement_3d(np.array(input_shape,
                                                       dtype=np.int32),
                                                       input_affine, 
                                                       np.array(target_shape,
                                                       dtype=np.int32),
                                                       target_affine)
    disp1 = np.array(disp1, dtype=floating)
    assign1 = np.array(assign1)

    disp2, assign2 = vfu.create_random_displacement_3d(np.array(input_shape,
                                                       dtype=np.int32),
                                                       input_affine,
                                                       np.array(target_shape,
                                                       dtype=np.int32),
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
                                               1.0, None)
    #apply the implementation under test
    warped = np.array(vfu.warp_3d(moving_image, composition, None,
                                      premult_index, premult_disp))
    assert_array_almost_equal(warped, expected)

    #test also using nearest neighbor interpolation
    warped = np.array(vfu.warp_3d_nn(moving_image, composition, None,
                                         premult_index, premult_disp))
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


def test_resample_vector_field_2d():
    r"""
    Expand a vector field by 2, then subsample by 2, the resulting
    field should be the original one
    """
    domain_shape = np.array((64, 64), dtype = np.int32)
    reduced_shape = np.array((32, 32), dtype = np.int32)
    factors = np.array([0.5, 0.5])
    d, dinv = vfu.create_harmonic_fields_2d(reduced_shape[0], reduced_shape[1], 0.3, 6)
    d = np.array(d, dtype = floating)
    
    expanded = vfu.resample_displacement_field_2d(d, factors,domain_shape)
    subsampled = expanded[::2, ::2, :]

    assert_array_almost_equal(d, subsampled)


def test_resample_vector_field_3d():
    r"""
    Expand a vector field by 2, then subsample by 2, the resulting
    field should be the original one
    """
    domain_shape = np.array((64, 64, 64), dtype = np.int32)
    reduced_shape = np.array((32, 32, 32), dtype = np.int32)
    factors = np.array([0.5, 0.5, 0.5])
    d, dinv = vfu.create_harmonic_fields_3d(reduced_shape[0], reduced_shape[1], reduced_shape[2], 0.3, 6)
    d = np.array(d, dtype = floating)
    
    expanded = vfu.resample_displacement_field_3d(d, factors,domain_shape)
    subsampled = expanded[::2, ::2, ::2, :]

    assert_array_almost_equal(d, subsampled)


def test_downsample_scalar_field_2d():
    np.random.seed(8315759)
    shape = (32, 32)
    image = np.ndarray(shape, dtype=floating)
    image[...] = np.random.randint(0, 10, np.size(image)).reshape(shape)
    a = image[::2, ::2]
    b = image[1::2, ::2]
    c = image[::2, 1::2]
    d = image[1::2, 1::2]
    expected = 0.25*(a + b + c + d)
    actual = np.array(vfu.downsample_scalar_field_2d(image))
    assert_array_almost_equal(expected, actual)
    

def test_downsample_displacement_field_2d():
    np.random.seed(8315759)
    shape = (32, 32, 2)
    field = np.ndarray(shape, dtype=floating)
    field[...] = np.random.randint(0, 10, np.size(field)).reshape(shape)
    a = field[::2, ::2, :]
    b = field[1::2, ::2, :]
    c = field[::2, 1::2, :]
    d = field[1::2, 1::2, :]
    expected = 0.25*(a + b + c + d)
    actual = np.array(vfu.downsample_displacement_field_2d(field))
    assert_array_almost_equal(expected, actual)

def test_downsample_scalar_field_3d():
    np.random.seed(8315759)
    shape = (32, 32, 32)
    volume = np.ndarray(shape, dtype=floating)
    volume[...] = np.random.randint(0, 10, np.size(volume)).reshape(shape)
    a = volume[::2, ::2, ::2]
    b = volume[1::2, ::2, ::2]
    c = volume[::2, 1::2, ::2]
    d = volume[1::2, 1::2, ::2]
    aa = volume[::2, ::2, 1::2]
    bb = volume[1::2, ::2, 1::2]
    cc = volume[::2, 1::2, 1::2]
    dd = volume[1::2, 1::2, 1::2]
    expected = 0.125*(a + b + c + d + aa + bb + cc + dd)
    actual = np.array(vfu.downsample_scalar_field_3d(volume))
    assert_array_almost_equal(expected, actual)


def test_downsample_displacement_field_3d():
    np.random.seed(8315759)
    shape = (32, 32, 32, 3)
    field = np.ndarray(shape, dtype=floating)
    field[...] = np.random.randint(0, 10, np.size(field)).reshape(shape)
    a = field[::2, ::2, ::2, :]
    b = field[1::2, ::2, ::2, :]
    c = field[::2, 1::2, ::2, :]
    d = field[1::2, 1::2, ::2, :]
    aa = field[::2, ::2, 1::2, :]
    bb = field[1::2, ::2, 1::2, :]
    cc = field[::2, 1::2, 1::2, :]
    dd = field[1::2, 1::2, 1::2, :]
    expected = 0.125*(a + b + c + d + aa + bb + cc + dd)
    actual = np.array(vfu.downsample_displacement_field_3d(field))
    assert_array_almost_equal(expected, actual)


def test_reorient_vector_field_2d():
    shape = (16,16)
    d, dinv = vfu.create_harmonic_fields_2d(shape[0], shape[1], 0.2, 4)
    d = np.array(d, dtype = floating)

    #the vector field rotated 90 degrees
    expected = np.ndarray(shape = shape+(2,), dtype = floating)
    expected[...,0] = -1 * d[...,1]
    expected[...,1] =  d[...,0]
    
    #rotate 45 degrees twice
    c = np.sqrt(0.5)
    affine = np.array([[c, -c],[c, c]])
    vfu.reorient_vector_field_2d(d, affine)
    vfu.reorient_vector_field_2d(d, affine)

    #verify almost equal
    assert_array_almost_equal(d, expected)


def test_reorient_vector_field_3d():
    shape = (16, 16, 16)
    d, dinv = vfu.create_harmonic_fields_3d(shape[0], shape[1], shape[2], 0.2, 4)
    d = np.array(d, dtype = floating)
    dinv = np.array(dinv, dtype = floating)

    #the vector field rotated 90 degrees around the last axis
    expected = np.ndarray(shape = shape+(3,), dtype = floating)
    expected[...,0] = -1 * d[...,1]
    expected[...,1] =  d[...,0]
    expected[...,2] =  d[...,2]

    #rotate 45 degrees twice around the last axis
    c = np.sqrt(0.5)
    affine = np.array([[c, -c, 0],[c, c, 0], [0, 0, 1]])
    vfu.reorient_vector_field_3d(d, affine)
    vfu.reorient_vector_field_3d(d, affine)

    #verify almost equal
    assert_array_almost_equal(d, expected)

    #the vector field rotated 90 degrees around the first axis
    expected[...,0] = dinv[...,0]
    expected[...,1] = -1 * dinv[...,2]
    expected[...,2] =  dinv[...,1]

    #rotate 45 degrees twice around the first axis
    affine = np.array([[1, 0, 0], [0, c, -c], [0, c, c]])
    vfu.reorient_vector_field_3d(dinv, affine)
    vfu.reorient_vector_field_3d(dinv, affine)

    #verify almost equal
    assert_array_almost_equal(dinv, expected)


if __name__=='__main__':
    test_warping_2d()
    test_warping_3d()
    test_affine_warping_2d()
    test_affine_warping_3d()
    test_compose_vector_fields_2d()
    test_compose_vector_fields_3d()
    test_invert_vector_field_2d()
    test_invert_vector_field_3d()
    test_resample_vector_field_2d()
    test_resample_vector_field_3d()
    test_downsample_scalar_field_2d()
    test_downsample_scalar_field_3d()
    test_downsample_displacement_field_2d()
    test_downsample_displacement_field_3d()
    test_reorient_vector_field_2d()
    test_reorient_vector_field_3d()
