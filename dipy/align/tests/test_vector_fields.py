import numpy as np
from dipy.align import floating
import dipy.align.vector_fields as vfu
from numpy.testing import (assert_array_almost_equal,
                           assert_almost_equal,
                           assert_equal)
import dipy.align.imwarp as imwarp


def test_interpolate_scalar_2d():
    from scipy.ndimage.interpolation import map_coordinates 
    np.random.seed(5324989)
    sz = 64
    target_shape = (sz, sz)
    image = np.ndarray(target_shape, dtype=floating)
    image[...] = np.random.randint(0, 10, np.size(image)).reshape(target_shape)
    #Select some coordinates to interpolate at
    nsamples = 200
    locations = np.random.ranf(2 * nsamples).reshape((nsamples,2)) * (sz+2) - 1.0

    #Call the implementation under test
    interp, inside = vfu.interpolate_scalar_2d(image, locations)

    #Call the reference implementation
    expected = map_coordinates(image, locations.transpose(), order=1)

    assert_array_almost_equal(expected, interp)

    #Test the 'inside' flag
    for i in range(nsamples):
        if (locations[i, 0]<0 or locations[i, 0]>(sz-1)) or (locations[i, 1]<0 or locations[i, 1]>(sz-1)):
            assert_equal(inside[i], 0)
        else:
            assert_equal(inside[i], 1)


def test_interpolate_scalar_nn_2d():
    from scipy.ndimage.interpolation import map_coordinates 
    np.random.seed(1924781)
    sz = 64
    target_shape = (sz, sz)
    image = np.ndarray(target_shape, dtype=floating)
    image[...] = np.random.randint(0, 10, np.size(image)).reshape(target_shape)
    #Select some coordinates to interpolate at
    nsamples = 200
    locations = np.random.ranf(2 * nsamples).reshape((nsamples,2)) * (sz+2) - 1.0

    #Call the implementation under test
    interp, inside = vfu.interpolate_scalar_nn_2d(image, locations)

    #Call the reference implementation
    expected = map_coordinates(image, locations.transpose(), order=0)

    assert_array_almost_equal(expected, interp)

    #Test the 'inside' flag
    for i in range(nsamples):
        if (locations[i, 0]<0 or locations[i, 0]>(sz-1)) or (locations[i, 1]<0 or locations[i, 1]>(sz-1)):
            assert_equal(inside[i], 0)
        else:
            assert_equal(inside[i], 1)


def test_interpolate_scalar_nn_3d():
    from scipy.ndimage.interpolation import map_coordinates 
    np.random.seed(3121121)
    sz = 64
    target_shape = (sz, sz, sz)
    image = np.ndarray(target_shape, dtype=floating)
    image[...] = np.random.randint(0, 10, np.size(image)).reshape(target_shape)
    #Select some coordinates to interpolate at
    nsamples = 200
    locations = np.random.ranf(3 * nsamples).reshape((nsamples,3)) * (sz+2) - 1.0

    #Call the implementation under test
    interp, inside = vfu.interpolate_scalar_nn_3d(image, locations)

    #Call the reference implementation
    expected = map_coordinates(image, locations.transpose(), order=0)

    assert_array_almost_equal(expected, interp)

    #Test the 'inside' flag
    for i in range(nsamples):
        expected_inside = 1
        for axis in range(3):
            if (locations[i, axis]<0 or locations[i, axis]>(sz-1)):
                expected_inside = 0
                break
        assert_equal(inside[i], expected_inside)


def test_interpolate_scalar_3d():
    from scipy.ndimage.interpolation import map_coordinates 
    np.random.seed(9216326)
    sz = 64
    target_shape = (sz, sz, sz)
    image = np.ndarray(target_shape, dtype=floating)
    image[...] = np.random.randint(0, 10, np.size(image)).reshape(target_shape)
    #Select some coordinates to interpolate at
    nsamples = 200
    locations = np.random.ranf(3 * nsamples).reshape((nsamples,3)) * (sz+2) - 1.0

    #Call the implementation under test
    interp, inside = vfu.interpolate_scalar_3d(image, locations)

    #Call the reference implementation
    expected = map_coordinates(image, locations.transpose(), order=1)

    assert_array_almost_equal(expected, interp)

    #Test the 'inside' flag
    for i in range(nsamples):
        expected_inside = 1
        for axis in range(3):
            if (locations[i, axis]<0 or locations[i, axis]>(sz-1)):
                expected_inside = 0
                break
        assert_equal(inside[i], expected_inside)


def test_interpolate_vector_3d():
    from scipy.ndimage.interpolation import map_coordinates 
    np.random.seed(7711219)
    sz = 64
    target_shape = (sz, sz, sz)
    field = np.ndarray(target_shape+(3,), dtype=floating)
    field[...] = np.random.randint(0, 10, np.size(field)).reshape(target_shape+(3,))
    #Select some coordinates to interpolate at
    nsamples = 200
    locations = np.random.ranf(3 * nsamples).reshape((nsamples,3)) * (sz+2) - 1.0

    #Call the implementation under test
    interp, inside = vfu.interpolate_vector_3d(field, locations)

    #Call the reference implementation
    expected = np.zeros_like(interp)
    for i in range(3):
        expected[...,i] = map_coordinates(field[...,i], locations.transpose(), order=1)

    assert_array_almost_equal(expected, interp)

    #Test the 'inside' flag
    for i in range(nsamples):
        expected_inside = 1
        for axis in range(3):
            if (locations[i, axis]<0 or locations[i, axis]>(sz-1)):
                expected_inside = 0
                break
        assert_equal(inside[i], expected_inside)


def test_interpolate_vector_2d():
    from scipy.ndimage.interpolation import map_coordinates 
    np.random.seed(1271244)
    sz = 64
    target_shape = (sz, sz)
    field = np.ndarray(target_shape+(2,), dtype=floating)
    field[...] = np.random.randint(0, 10, np.size(field)).reshape(target_shape+(2,))
    #Select some coordinates to interpolate at
    nsamples = 200
    locations = np.random.ranf(2 * nsamples).reshape((nsamples,2)) * (sz+2) - 1.0

    #Call the implementation under test
    interp, inside = vfu.interpolate_vector_2d(field, locations)

    #Call the reference implementation
    expected = np.zeros_like(interp)
    for i in range(2):
        expected[...,i] = map_coordinates(field[...,i], locations.transpose(), order=1)

    assert_array_almost_equal(expected, interp)

    #Test the 'inside' flag
    for i in range(nsamples):
        expected_inside = 1
        for axis in range(2):
            if (locations[i, axis]<0 or locations[i, axis]>(sz-1)):
                expected_inside = 0
                break
        assert_equal(inside[i], expected_inside)


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
    
    for time_scaling in [0.25, 0.5, 1.0, 2.0, 4.0]:
        composition, stats = vfu.compose_vector_fields_2d(disp1,
                                                          disp2/time_scaling,
                                                          premult_index,
                                                          premult_disp,
                                                          time_scaling, None)
        #apply the implementation under test
        warped = np.array(vfu.warp_2d(moving_image, composition, None,
                                         premult_index, premult_disp))
        assert_array_almost_equal(warped, expected)

        #test also using nearest neighbor interpolation
        warped = np.array(vfu.warp_2d_nn(moving_image, composition, None,
                                            premult_index, premult_disp))
        assert_array_almost_equal(warped, expected)

    # Test non-overlapping case
    x_0 = np.asarray(range(input_shape[0]))
    x_1 = np.asarray(range(input_shape[1]))
    X = np.ndarray(input_shape + (2,), dtype = np.float64)
    O = np.ones(input_shape)
    X[...,0]= x_0[:, None] * O
    X[...,1]= x_1[None, :] * O
    random_labels = np.random.randint(0, 2, input_shape[0]*input_shape[1]*2)
    random_labels = random_labels.reshape(input_shape+(2,))
    values = np.array([-1, target_shape[0]])
    disp1 = (values[random_labels] - X).astype(floating)
    composition, stats = vfu.compose_vector_fields_2d(disp1,
                                                      disp2,
                                                      None,
                                                      None,
                                                      1.0, None)
    assert_array_almost_equal(composition, np.zeros_like(composition))


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
    
    for time_scaling in [0.25, 0.5, 1.0, 2.0, 4.0]:
        composition, stats = vfu.compose_vector_fields_3d(disp1,
                                                          disp2/time_scaling,
                                                          premult_index,
                                                          premult_disp,
                                                          time_scaling, None)
        #apply the implementation under test
        warped = np.array(vfu.warp_3d(moving_image, composition, None,
                                          premult_index, premult_disp))
        assert_array_almost_equal(warped, expected)

        #test also using nearest neighbor interpolation
        warped = np.array(vfu.warp_3d_nn(moving_image, composition, None,
                                             premult_index, premult_disp))
        assert_array_almost_equal(warped, expected)

    # Test non-overlapping case
    x_0 = np.asarray(range(input_shape[0]))
    x_1 = np.asarray(range(input_shape[1]))
    x_2 = np.asarray(range(input_shape[2]))
    X = np.ndarray(input_shape + (3,), dtype = np.float64)
    O = np.ones(input_shape)
    X[...,0]= x_0[:, None, None] * O
    X[...,1]= x_1[None, :, None] * O
    X[...,2]= x_1[None, None, :] * O
    random_labels = np.random.randint(0, 2, input_shape[0]*input_shape[1]*input_shape[2]*3)
    random_labels = random_labels.reshape(input_shape+(3,))
    values = np.array([-1, target_shape[0]])
    disp1 = (values[random_labels] - X).astype(floating)
    composition, stats = vfu.compose_vector_fields_3d(disp1,
                                                      disp2,
                                                      None,
                                                      None,
                                                      1.0, None)
    assert_array_almost_equal(composition, np.zeros_like(composition))



def test_invert_vector_field_2d():
    r"""
    Inverts a synthetic, analytically invertible, displacement field
    """
    import dipy.align.imwarp as imwarp
    shape = (64, 64)
    nr = shape[0]
    nc = shape[1]
    # Create an arbitrary image-to-space transform
    t = 2.5 #translation factor

    trans = np.array([[1, 0, -t*nr],
                      [0, 1, -t*nc],
                      [0, 0, 1]])
    trans_inv = np.linalg.inv(trans)

    d, dinv = vfu.create_harmonic_fields_2d(nr, nc, 0.2, 8)
    d = np.asarray(d).astype(floating)
    dinv = np.asarray(dinv).astype(floating)

    for theta in [-1 * np.pi/5.0, 0.0, np.pi/5.0]: #rotation angle
        for s in [0.5,  1.0, 2.0]: #scale
            ct = np.cos(theta)
            st = np.sin(theta)

            rot = np.array([[ct, -st, 0],
                            [st, ct, 0],
                            [0, 0, 1]])

            scale = np.array([[1*s, 0, 0],
                              [0, 1*s, 0],
                              [0, 0, 1]])

            gt_affine = trans_inv.dot(scale.dot(rot.dot(trans)))
            gt_affine_inv = np.linalg.inv(gt_affine)
            dcopy = np.copy(d)

            #make sure the field remains invertible after the re-mapping
            vfu.reorient_vector_field_2d(dcopy, gt_affine) 

            inv_approx = vfu.invert_vector_field_fixed_point_2d(dcopy, gt_affine_inv,
                                                                np.array([s, s]),
                                                                40, 1e-7)

            mapping = imwarp.DiffeomorphicMap(2, (nr,nc), gt_affine)
            mapping.forward = dcopy
            mapping.backward = inv_approx
            residual, stats = mapping.compute_inversion_error()
            assert_almost_equal(stats[1], 0, decimal=4)
            assert_almost_equal(stats[2], 0, decimal=4)


def test_invert_vector_field_3d():
    r"""
    Inverts a synthetic, analytically invertible, displacement field
    """
    import dipy.align.imwarp as imwarp
    import dipy.core.geometry as geometry
    shape = (64, 64, 64)
    ns = shape[0]
    nr = shape[1]
    nc = shape[2]

    # Create an arbitrary image-to-space transform

    # Select an arbitrary rotation axis
    axis = np.array([2.0, 0.5, 1.0])
    t = 2.5 #translation factor

    trans = np.array([[1, 0, 0, -t*ns],
                      [0, 1, 0, -t*nr],
                      [0, 0, 1, -t*nc],
                      [0, 0, 0, 1]])
    trans_inv = np.linalg.inv(trans)

    d, dinv = vfu.create_harmonic_fields_3d(ns, nr, nc, 0.2, 8)
    d = np.asarray(d).astype(floating)
    dinv = np.asarray(dinv).astype(floating)

    for theta in [-1 * np.pi/5.0, 0.0, np.pi/5.0]: #rotation angle
        for s in [0.5,  1.0, 2.0]: #scale
            rot = np.zeros(shape=(4,4))
            rot[:3, :3] = geometry.rodrigues_axis_rotation(axis, theta)
            rot[3,3] = 1.0
            scale = np.array([[1*s, 0, 0, 0],
                              [0, 1*s, 0, 0],
                              [0, 0, 1*s, 0],
                              [0, 0, 0, 1]])

            gt_affine = trans_inv.dot(scale.dot(rot.dot(trans)))
            gt_affine_inv = np.linalg.inv(gt_affine)
            dcopy = np.copy(d)

            #make sure the field remains invertible after the re-mapping
            vfu.reorient_vector_field_3d(dcopy, gt_affine) 

            # Note: the spacings are used just to check convergence, so they don't need
            # to be very accurate. Here we are passing (0.5 * s) to force the algorithm
            # to make more iterations: in ANTS, there is a hard-coded bound on the maximum
            # residual, that's why we cannot force more iteration by changing the parameters.
            # We will investigate this issue with more detail in the future.

            inv_approx = vfu.invert_vector_field_fixed_point_3d(dcopy, gt_affine_inv,
                                                                np.array([s, s, s])*0.5,
                                                                40, 1e-7)

            mapping = imwarp.DiffeomorphicMap(3, (nr,nc), gt_affine)
            mapping.forward = dcopy
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
    size = 32
    for reduce_r in [True, False]:
        nrows = size -1 if reduce_r else size
        for reduce_c in [True, False]:
            ncols = size -1 if reduce_c else size
            image = np.ndarray((size, size), dtype=floating)
            image[...] = np.random.randint(0, 10, np.size(image)).reshape((size, size))

            if reduce_r:
                image[-1, :] = 0
            if reduce_c:
                image[:, -1] = 0

            a = image[::2, ::2]
            b = image[1::2, ::2]
            c = image[::2, 1::2]
            d = image[1::2, 1::2]

            expected = 0.25*(a + b + c + d)

            if reduce_r:
                expected[-1,:]*=2
            if reduce_c:
                expected[:,-1]*=2

            actual = np.array(vfu.downsample_scalar_field_2d(image[:nrows, :ncols]))
            assert_array_almost_equal(expected, actual)
    

def test_downsample_displacement_field_2d():
    np.random.seed(2115556)
    size = 32
    for reduce_r in [True, False]:
        nrows = size -1 if reduce_r else size
        for reduce_c in [True, False]:
            ncols = size -1 if reduce_c else size
            field = np.ndarray((size, size, 2), dtype=floating)
            field[...] = np.random.randint(0, 10, np.size(field)).reshape((size, size, 2))

            if reduce_r:
                field[-1, :, :] = 0
            if reduce_c:
                field[:, -1, :] = 0

            a = field[::2, ::2, :]
            b = field[1::2, ::2, :]
            c = field[::2, 1::2, :]
            d = field[1::2, 1::2, :]

            expected = 0.25*(a + b + c + d)

            if reduce_r:
                expected[-1, :, :]*=2
            if reduce_c:
                expected[:, -1, :]*=2

            actual = np.array(vfu.downsample_displacement_field_2d(field[:nrows, :ncols, :]))
            assert_array_almost_equal(expected, actual)


def test_downsample_scalar_field_3d():
    np.random.seed(8315759)
    size = 32
    for reduce_s in [True, False]:
        nslices = size -1 if reduce_s else size
        for reduce_r in [True, False]:
            nrows = size -1 if reduce_r else size
            for reduce_c in [True, False]:
                ncols = size -1 if reduce_c else size
                image = np.ndarray((size, size, size), dtype=floating)
                image[...] = np.random.randint(0, 10, np.size(image)).reshape((size, size, size))

                if reduce_s:
                    image[-1, :, :] = 0
                if reduce_r:
                    image[:, -1, :] = 0
                if reduce_c:
                    image[:, :, -1] = 0

                a = image[::2, ::2, ::2]
                b = image[1::2, ::2, ::2]
                c = image[::2, 1::2, ::2]
                d = image[1::2, 1::2, ::2]
                aa = image[::2, ::2, 1::2]
                bb = image[1::2, ::2, 1::2]
                cc = image[::2, 1::2, 1::2]
                dd = image[1::2, 1::2, 1::2]

                expected = 0.125*(a + b + c + d + aa + bb + cc + dd)

                if reduce_s:
                    expected[-1, :, :] *= 2
                if reduce_r:
                    expected[:, -1, :] *= 2
                if reduce_c:
                    expected[:, :, -1] *= 2

                actual = np.array(vfu.downsample_scalar_field_3d(image[:nslices, :nrows, :ncols]))
                assert_array_almost_equal(expected, actual)


def test_downsample_displacement_field_3d():
    np.random.seed(8315759)
    size = 32
    for reduce_s in [True, False]:
        nslices = size -1 if reduce_s else size
        for reduce_r in [True, False]:
            nrows = size -1 if reduce_r else size
            for reduce_c in [True, False]:
                ncols = size -1 if reduce_c else size
                field = np.ndarray((size, size, size, 3), dtype=floating)
                field[...] = np.random.randint(0, 10, np.size(field)).reshape((size, size, size, 3))

                if reduce_s:
                    field[-1, :, :] = 0
                if reduce_r:
                    field[:, -1, :] = 0
                if reduce_c:
                    field[:, :, -1] = 0

                a = field[::2, ::2, ::2, :]
                b = field[1::2, ::2, ::2, :]
                c = field[::2, 1::2, ::2, :]
                d = field[1::2, 1::2, ::2, :]
                aa = field[::2, ::2, 1::2, :]
                bb = field[1::2, ::2, 1::2, :]
                cc = field[::2, 1::2, 1::2, :]
                dd = field[1::2, 1::2, 1::2, :]

                expected = 0.125*(a + b + c + d + aa + bb + cc + dd)

                if reduce_s:
                    expected[-1, :, :, :] *= 2
                if reduce_r:
                    expected[:, -1, :, :] *= 2
                if reduce_c:
                    expected[:, :, -1, :] *= 2

                actual = np.array(vfu.downsample_displacement_field_3d(field[:nslices, :nrows, :ncols]))
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
    test_interpolate_scalar_2d()
    test_interpolate_scalar_nn_2d()
    test_interpolate_scalar_nn_3d()
    test_interpolate_scalar_3d()
    #test_warping_2d()
    #test_warping_3d()
    #test_affine_warping_2d()
    #test_affine_warping_3d()
    #test_compose_vector_fields_2d()
    #test_compose_vector_fields_3d()
    #test_invert_vector_field_2d()
    #test_invert_vector_field_3d()
    #test_resample_vector_field_2d()
    #test_resample_vector_field_3d()
    #test_downsample_scalar_field_2d()
    #test_downsample_scalar_field_3d()
    #test_downsample_displacement_field_2d()
    #test_downsample_displacement_field_3d()
    #test_reorient_vector_field_2d()
    #test_reorient_vector_field_3d()
