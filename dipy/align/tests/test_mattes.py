import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
from dipy.core.ndindex import ndindex
from dipy.data import get_data
import dipy.align.vector_fields as vf
from dipy.align.transforms import regtransforms, Transform
from dipy.align.mattes import MattesBase, cubic_spline, cubic_spline_derivative
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           assert_almost_equal,
                           assert_equal)


def create_random_image_pair_2d(nr, nc, nvals):
    r""" Create a pair of images with an arbitrary, non-uniform joint
    distribution
    """
    sh = (nr, nc)
    static = np.random.randint(0, nvals, nr*nc).reshape(sh)

    # This is just a simple way of making joint the distribution non-uniform
    moving = static.copy()
    moving += np.random.randint(0, nvals//2, nr*nc).reshape(sh) - nvals//4

    # This is just a simple way of making joint the distribution non-uniform
    static = moving.copy()
    static += np.random.randint(0, nvals//2, nr*nc).reshape(sh) - nvals//4

    return static.astype(np.float64), moving.astype(np.float64)


def create_random_image_pair_3d(ns, nr, nc, nvals):
    r""" Create a pair of images with an arbitrary, non-uniform joint
    distribution
    """
    sh = (ns, nr, nc)
    static = np.random.randint(0, nvals, ns*nr*nc).reshape(sh)

    # This is just a simple way of making  the distribution non-uniform
    moving = static.copy()
    moving += np.random.randint(0, nvals//2, ns*nr*nc).reshape(sh) - nvals//4

    # This is just a simple way of making  the distribution non-uniform
    static = moving.copy()
    static += np.random.randint(0, nvals//2, ns*nr*nc).reshape(sh) - nvals//4

    return static.astype(np.float64), moving.astype(np.float64)


def test_cubic_spline():
    #Cubic spline as defined in [1] eq. (3)
    #
    #[1] Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K., & Eubank, W.
    #    PET-CT image registration in the chest using free-form deformations.
    #    IEEE Transactions on Medical Imaging, 22(1), 120-8, 2003.
    input = []
    expected = []
    epsilon = 1e-9
    for epsilon in [-1e-9, 0.0, 1e-9]:
        for t in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            x = t + epsilon
            input.append(x)
            absx = np.abs(x)
            sqrx = x * x
            if absx < 1:
                expected.append((4.0 - 6*sqrx + 3.0 * (absx**3))/6.0)
            elif absx < 2:
                expected.append(((2 - absx)**3)/6.0)
            else:
                expected.append(0.0)
    actual = cubic_spline(np.array(input, dtype=np.float64))
    assert_array_almost_equal(actual, np.array(expected, dtype=np.float64))


def test_cubic_spline_derivative():
    # Test derivative of the cubic spline, as defined in [1] eq. (3) by
    # comparing the analytical and numerical derivatives
    #
    #[1] Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K., & Eubank, W.
    #    PET-CT image registration in the chest using free-form deformations.
    #    IEEE Transactions on Medical Imaging, 22(1), 120-8, 2003.
    input = []
    expected = []
    epsilon = 1e-9
    for epsilon in [-1e-9, 0.0, 1e-9]:
        for t in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            x = t + epsilon
            input.append(x)
    h = 1e-6
    input = np.array(input)
    input_h = input + h
    s = np.array(cubic_spline(input))
    s_h = np.array(cubic_spline(input_h))
    expected = (s_h - s)/h
    actual = cubic_spline_derivative(input)
    print("max dif:%f\n", np.abs(expected-actual).max())
    assert_array_almost_equal(actual, expected)


def test_mattes_base():
    # Test the simple functionality of MattesBase,
    # the gradients and computation of the joint intensity distribution
    # will be tested independently
    for nbins in [15, 30, 50]:
        for min_int in [-10.0, 0.0, 10.0]:
            for intensity_range in [0.1, 1.0, 10.0]:
                factor = 1
                max_int = min_int + intensity_range
                M = MattesBase(nbins)
                # Make a pair of 4-pixel images, introduce +/- 1 values
                # that will be excluded using a mask
                static = np.array([min_int - 1.0, min_int,
                                   max_int, max_int + 1.0])
                # Multiply by an arbitrary value (make the ranges different)
                moving = factor * np.array([min_int, min_int - 1.0,
                                  max_int + 1.0, max_int])
                # Create a mask to exclude the invalid values (beyond min and
                # max computed above)
                static_mask = np.array([0, 1, 1, 0])
                moving_mask = np.array([1, 0, 0, 1])

                M.setup(static, moving, static_mask, moving_mask)

                # Test bin_normalize_static at the boundary
                normalized = M.bin_normalize_static(min_int)
                assert_almost_equal(normalized, M.padding)
                index = M.bin_index(normalized)
                assert_equal(index, M.padding)
                normalized = M.bin_normalize_static(max_int)
                assert_almost_equal(normalized, nbins - M.padding)
                index = M.bin_index(normalized)
                assert_equal(index, nbins - 1 - M.padding)

                # Test bin_normalize_moving at the boundary
                normalized = M.bin_normalize_moving(factor * min_int)
                assert_almost_equal(normalized, M.padding)
                index = M.bin_index(normalized)
                assert_equal(index, M.padding)
                normalized = M.bin_normalize_moving(factor * max_int)
                assert_almost_equal(normalized, nbins - M.padding)
                index = M.bin_index(normalized)
                assert_equal(index, nbins - 1 - M.padding)

                # Test bin_index not at the boundary
                delta_s = (max_int - min_int)/(nbins - 2*M.padding)
                delta_m = factor * (max_int - min_int)/(nbins - 2*M.padding)
                for i in range(nbins - 2*M.padding):
                    normalized = M.bin_normalize_static(min_int +
                                                        (i+0.5)*delta_s)
                    index = M.bin_index(normalized)
                    assert_equal(index, M.padding + i)

                    normalized = M.bin_normalize_moving(factor*min_int +
                                                        (i+0.5)*delta_m)
                    index = M.bin_index(normalized)
                    assert_equal(index, M.padding + i)


def test_mattes_densities_dense():
    # Test the computation of the joint intensity distribution
    # using a dense set of values (from a pair of images)
    np.random.seed(1246592)
    nbins = 32
    nr = 30
    nc = 35
    ns = 20
    nvals = 50

    for dim in [2, 3]:
        if dim == 2:
            shape = (nr, nc)
            static, moving = create_random_image_pair_2d(nr, nc, nvals)
        else:
            shape = (ns, nr, nc)
            static, moving = create_random_image_pair_3d(ns, nr, nc, nvals)

        # Initialize
        mbase = MattesBase(nbins)
        mbase.setup(static, moving)
        mbase.update_pdfs_dense(static, moving)
        actual_joint = mbase.joint
        actual_mmarginal = mbase.mmarginal
        actual_smarginal = mbase.smarginal

        # Compute the expected joint distribution
        expected_joint = np.zeros(shape=(nbins, nbins))
        for index in ndindex(shape):
            sval = mbase.bin_normalize_static(static[index])
            mval = mbase.bin_normalize_moving(moving[index])
            sbin = mbase.bin_index(sval)
            mbin = mbase.bin_index(mval)
            # The spline is centered at mval, will evaluate for all row
            spline_arg = np.array([i - mval for i in range(nbins)])
            contribution = cubic_spline(spline_arg)
            expected_joint[sbin, :] += contribution

        # Verify joint distribution
        expected_joint /= expected_joint.sum()
        assert_array_almost_equal(actual_joint, expected_joint)

        # Verify moving marginal
        expected_mmarginal = expected_joint.sum(0)
        expected_mmarginal /= expected_mmarginal.sum()
        assert_array_almost_equal(actual_mmarginal, expected_mmarginal)

        # Verivy static marginal
        expected_smarginal = expected_joint.sum(1)
        expected_smarginal /= expected_smarginal.sum()
        assert_array_almost_equal(actual_smarginal, expected_smarginal)


def test_mattes_densities_sparse():
    # Test the computation of the joint intensity distribution
    # from a list of pairs of corresponding intensities
    np.random.seed(3147702)
    nbins = 32
    nr = 30
    nc = 35
    nvals = 50

    shape = (nr, nc)
    static, moving = create_random_image_pair_2d(nr, nc, nvals)
    sval = static.reshape(-1)
    mval = moving.reshape(-1)

    # Initialize
    mbase = MattesBase(nbins)
    mbase.setup(static, moving)
    mbase.update_pdfs_sparse(sval, mval)
    actual_joint = mbase.joint
    actual_mmarginal = mbase.mmarginal
    actual_smarginal = mbase.smarginal

    # Compute the expected joint distribution
    expected_joint = np.zeros(shape=(nbins, nbins))
    for index in range(sval.shape[0]):
        sv = mbase.bin_normalize_static(sval[index])
        mv = mbase.bin_normalize_moving(mval[index])
        sbin = mbase.bin_index(sv)
        mbin = mbase.bin_index(mv)
        # The spline is centered at mval, will evaluate for all row
        spline_arg = np.array([i - mv for i in range(nbins)])
        contribution = cubic_spline(spline_arg)
        expected_joint[sbin, :] += contribution

    # Verify joint distribution
    expected_joint /= expected_joint.sum()
    assert_array_almost_equal(actual_joint, expected_joint)

    # Verify moving marginal
    expected_mmarginal = expected_joint.sum(0)
    expected_mmarginal /= expected_mmarginal.sum()
    assert_array_almost_equal(actual_mmarginal, expected_mmarginal)

    # Verivy static marginal
    expected_smarginal = expected_joint.sum(1)
    expected_smarginal /= expected_smarginal.sum()
    assert_array_almost_equal(actual_smarginal, expected_smarginal)


def setup_random_transform(transform, rfactor, nslices=45, sigma=1):
    r""" Creates a pair of images related to each other by an affine transform

    We warp the static image with a random transform so that the
    returned ground-truth transform will produce the static image
    when applied to the moving image. This will simply stack some copies of
    a T1 coronal slice image and add some zero slices up and down to
    reduce boundary artefacts when interpolating.

    Parameters
    ----------
    transform: instance of Transform
        defines the type of random transformation that will be created
    rfactor: float
        the factor to multiply the uniform(0,1) random noise that will be
        added to the identity parameters to create the random transform
    nslices: int
        number of slices to be stacked to form the volumes
    """
    dim = 2 if nslices ==1 else 3
    if transform.get_dim() !=dim:
        raise ValueError("Transform and requested volume have different dims.")
    np.random.seed(3147702)
    zero_slices = nslices//3

    fname = get_data('t1_coronal_slice')
    moving_slice = np.load(fname)
    moving_slice = moving_slice[40:180, 50:210]

    if nslices == 1:
        dim = 2
        moving = moving_slice
        warp_method = vf.warp_2d_affine
    else:
        dim = 3
        warp_method = vf.warp_3d_affine
        moving = np.zeros(shape=moving_slice.shape + (nslices,))
        moving[..., zero_slices:(2 * zero_slices)] = moving_slice[..., None]

    moving = sp.ndimage.filters.gaussian_filter(moving, sigma)

    moving_aff = np.eye(dim + 1)
    mmask = np.ones_like(moving, dtype=np.int32)

    # Create a transform by slightly perturbing the identity parameters
    theta = transform.get_identity_parameters()
    n = transform.get_number_of_parameters()
    theta += np.random.rand(n) * rfactor

    T = transform.param_to_matrix(theta)
    shape = np.array(moving.shape, dtype=np.int32)
    static = np.array(warp_method(moving.astype(np.float32), shape, T))
    static = static.astype(np.float64)
    static_aff = np.eye(dim + 1)
    smask = np.ones_like(static, dtype=np.int32)

    return static, moving, static_aff, moving_aff, smask, mmask, T


def test_joint_pdf_gradients():
    # Compare the analytical and numerical (finite differences) gradient of the
    # joint distribution (i.e. derivatives of each histogram cell) w.r.t. the
    # transform parameters. Since the histograms are discrete partitions of the
    # image intensities, the finite difference approximation is normally not
    # very close to the analytical derivatives. Other sources of error are the
    # interpolation used when warping the images and the boundary intensities
    # introduced when interpolating outside of the image (i.e. some "zeros" are
    # introduced at the boundary which affect the numerical derivatives but is
    # not taken into account by the analytical derivatives). Thus, we need to
    # relax the verification. Instead of looking for the analytical and
    # numerical gradients to be very close to each other, we will verify that
    # they approximately point in the same direction by testing the angle they
    # form is close to zero.
    h = 1e-4
    factors = {('TRANSLATION', 2):2.0,
               ('ROTATION', 2):0.1,
               ('RIGID', 2):0.1,
               ('SCALING', 2):0.01,
               ('AFFINE', 2):0.1,
               ('TRANSLATION', 3):2.0,
               ('ROTATION', 3):0.2,
               ('RIGID', 3):0.1,
               ('SCALING', 3):0.1,
               ('AFFINE', 3):0.1}
    for ttype in factors.keys():
        dim = ttype[1]
        if dim == 2:
            nslices = 1
            warp_method = vf.warp_2d_affine
        else:
            nslices = 45
            warp_method = vf.warp_3d_affine

        transform = regtransforms[ttype]
        factor = factors[ttype]
        theta = transform.get_identity_parameters()

        static, moving, static_aff, moving_aff, smask, mmask, T = \
                        setup_random_transform(transform, factor, nslices, 5.0)
        metric = MattesBase(32)
        metric.setup(static, moving, smask, mmask)

        # Compute the gradient at theta with the implementation under test
        T = transform.param_to_matrix(theta)
        shape = np.array(static.shape, dtype=np.int32)

        warped = warp_method(moving.astype(np.float32), shape, T)
        warped = np.array(warped)
        metric.update_pdfs_dense(static.astype(np.float64),
                                 warped.astype(np.float64))
        J0 = np.copy(metric.joint)
        # Now we have the joint distribution evaluated at theta
        grid_to_space = np.eye(dim + 1)
        spacing = np.ones(dim, dtype=np.float64)
        mgrad = vf.gradient(moving.astype(np.float32), moving_aff,
                            spacing, shape, grid_to_space)
        id = transform.get_identity_parameters()
        metric.update_gradient_dense(id, transform, static.astype(np.float64),
                                     warped.astype(np.float64), grid_to_space,
                                     mgrad, smask, mmask)
        actual = np.copy(metric.joint_grad)
        # Now we have the gradient of the joint distribution w.r.t. the
        # transform parameters

        # Compute the gradient using finite-diferences
        n = transform.get_number_of_parameters()
        expected = np.empty_like(actual)
        for i in range(n):
            dtheta = theta.copy()
            dtheta[i] += h
            # Update the joint distribution with the warped moving image
            T = transform.param_to_matrix(dtheta)
            shape = np.array(static.shape, dtype=np.int32)
            warped = warp_method(moving.astype(np.float32), shape, T)
            warped = np.array(warped)
            metric.update_pdfs_dense(static.astype(np.float64),
                                     warped.astype(np.float64))
            J1 =  np.copy(metric.joint)
            expected[...,i] = (J1 - J0) / h

        # Dot product and norms of gradients of each joint histogram cell
        # i.e. the derivatives of each cell w.r.t. all parameters
        P = (expected*actual).sum(2)
        enorms = np.sqrt((expected**2).sum(2))
        anorms = np.sqrt((actual**2).sum(2))
        prodnorms = enorms*anorms
        # Cosine of angle between the expected and actual gradients.
        # Exclude very small gradients
        P[prodnorms > 1e-6] /= (prodnorms[prodnorms > 1e-6])
        P[prodnorms <= 1e-6]=0
        # Verify that a large proportion of the gradients point almost in
        # the same direction. Disregard very small gradients
        mean_cosine = P[P!=0].mean()
        std_cosine = P[P!=0].std()
        assert_equal(mean_cosine > 0.9, True)
        assert_equal(std_cosine < 0.25, True)


def test_mi_gradient():
    # Test the gradient of mutual information using MattesMIMetric,
    # which extends MattesBase
    h = 1e-5
    factors = {('TRANSLATION', 2):2.0,
               ('ROTATION', 2):0.1,
               ('RIGID', 2):0.1,
               ('SCALING', 2):0.01,
               ('AFFINE', 2):0.1,
               ('TRANSLATION', 3):2.0,
               ('ROTATION', 3):0.2,
               ('RIGID', 3):0.1,
               ('SCALING', 3):0.1,
               ('AFFINE', 3):0.1}
    for ttype in factors.keys():
        transform = regtransforms[ttype]
        dim = ttype[1]
        if dim == 2:
            nslices = 1
            warp_method = vf.warp_2d_affine
        else:
            nslices = 45
            warp_method = vf.warp_3d_affine
        # Get data (pair of images related to each other by an known transform)
        factor = factors[ttype]
        static, moving, static_aff, moving_aff, smask, mmask, T = \
                        setup_random_transform(transform, factor, nslices, 5.0)
        smask=None
        mmask=None

        # Prepare a MattesBase instance
        # The computation of the metric is done in 3 steps:
        # 1.Compute the joint distribution
        # 2.Compute the gradient of the joint distribution
        # 3.Compute the metric's value and gradient using results from 1 and 2
        metric = MattesBase(32)
        metric.setup(static, moving, smask, mmask)

        # 1. Update the joint distribution
        metric.update_pdfs_dense(static.astype(np.float64),
                                 moving.astype(np.float64))

        # 2. Update the joint distribution gradient (the derivative of each
        # histogram cell w.r.t. the transform parameters). This requires
        # among other things, the spatial gradient of the moving image.
        theta = transform.get_identity_parameters().copy()
        grid_to_space = np.eye(dim + 1)
        spacing = np.ones(dim, dtype=np.float64)
        shape = np.array(static.shape, dtype=np.int32)
        mgrad = vf.gradient(moving.astype(np.float32), moving_aff,
                            spacing, shape, grid_to_space)
        metric.update_gradient_dense(theta, transform, static.astype(np.float64),
                                     moving.astype(np.float64), grid_to_space,
                                     mgrad, smask, mmask)

        # 3. Update the metric (in this case, the Mutual Information) and its
        # gradient, which is computed from the joint density and its gradient
        metric.update_mi_metric(update_gradient=True)

        # Now we can extract the value and gradient of the metric
        # This is the gradient according to the implementation under test
        val0 = metric.metric_val
        actual = np.copy(metric.metric_grad)

        # Compute the gradient using finite-diferences
        n = transform.get_number_of_parameters()
        expected = np.empty_like(actual)
        for i in range(n):
            dtheta = theta.copy()
            dtheta[i] += h

            T = transform.param_to_matrix(dtheta)
            shape = np.array(static.shape, dtype=np.int32)
            warped = np.array(warp_method(moving.astype(np.float32), shape, T))
            metric.update_pdfs_dense(static.astype(np.float64),
                                     warped.astype(np.float64))
            metric.update_mi_metric(update_gradient=False)
            val1 = metric.metric_val
            expected[i] = (val1 - val0) / h

        dp = expected.dot(actual)
        enorm = np.linalg.norm(expected)
        anorm = np.linalg.norm(actual)
        nprod = dp / (enorm * anorm)
        print(ttype)
        print("nprod:%f\n"%(nprod,))
        assert_equal(nprod >= 0.999, True)


if __name__ == '__main__':
    test_cubic_spline()
    test_cubic_spline_derivative()
    test_mattes_base()
    test_mattes_densities_dense()
    test_mattes_densities_sparse()
    test_joint_pdf_gradients()
    test_mi_gradient()
