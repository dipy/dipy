import numpy as np
from numpy.testing import (assert_equal,
                           assert_array_equal,
                           assert_array_almost_equal,
                           assert_raises)
from dipy.align import floating
from dipy.align import expectmax as em
from dipy.testing.decorators import set_random_number_generator


@set_random_number_generator(1346491)
def test_compute_em_demons_step_2d(rng):
    r"""
    Compares the output of the demons step in 2d against an analytical
    step. The fixed image is given by $F(x) = \frac{1}{2}||x - c_f||^2$, the
    moving image is given by $G(x) = \frac{1}{2}||x - c_g||^2$,
    $x, c_f, c_g \in R^{2}$

    References
    ----------
    [Vercauteren09] Vercauteren, T., Pennec, X., Perchant, A., & Ayache, N.
                    (2009). Diffeomorphic demons: efficient non-parametric
                    image registration. NeuroImage, 45(1 Suppl), S61-72.
                    doi:10.1016/j.neuroimage.2008.10.040
    """
    # Select arbitrary images' shape (same shape for both images)
    sh = (30, 20)

    # Select arbitrary centers
    c_f = np.asarray(sh) / 2
    c_g = c_f + 0.5

    # Compute the identity vector field I(x) = x in R^2
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    X = np.ndarray(sh + (2,), dtype=np.float64)
    O = np.ones(sh)
    X[..., 0] = x_0[:, None] * O
    X[..., 1] = x_1[None, :] * O

    # Compute the gradient fields of F and G
    grad_F = X - c_f
    grad_G = X - c_g

    # The squared norm of grad_G to be used later
    sq_norm_grad_G = np.sum(grad_G**2, -1)

    # Compute F and G
    F = 0.5 * np.sum(grad_F**2, -1)
    G = 0.5 * sq_norm_grad_G
    delta_field = G - F

    # Now select an arbitrary parameter for
    # $\sigma_x$ (eq 4 in [Vercauteren09])
    sigma_x_sq = 1.5

    # Set arbitrary values for $\sigma_i$ (eq. 4 in [Vercauteren09])
    # The original Demons algorithm used simply |F(x) - G(x)| as an
    # estimator, so let's use it as well
    sigma_i_sq = (F - G)**2

    # Select some pixels to have special values
    random_labels = rng.integers(0, 5, sh[0] * sh[1])
    random_labels = random_labels.reshape(sh)

    # this label is used to set sigma_i_sq == 0 below
    random_labels[sigma_i_sq == 0] = 2
    # this label is used to set gradient == 0 below
    random_labels[sq_norm_grad_G == 0] = 2

    expected = np.zeros_like(grad_G)
    # Pixels with sigma_i_sq = inf
    sigma_i_sq[random_labels == 0] = np.inf
    expected[random_labels == 0, ...] = 0

    # Pixels with gradient!=0 and sigma_i_sq=0
    sqnrm = sq_norm_grad_G[random_labels == 1]
    sigma_i_sq[random_labels == 1] = 0
    expected[random_labels == 1, 0] = (delta_field[random_labels == 1] *
                                       grad_G[random_labels == 1, 0] / sqnrm)
    expected[random_labels == 1, 1] = (delta_field[random_labels == 1] *
                                       grad_G[random_labels == 1, 1] / sqnrm)

    # Pixels with gradient=0 and sigma_i_sq=0
    sigma_i_sq[random_labels == 2] = 0
    grad_G[random_labels == 2, ...] = 0
    expected[random_labels == 2, ...] = 0

    # Pixels with gradient=0 and sigma_i_sq!=0
    grad_G[random_labels == 3, ...] = 0

    # Directly compute the demons step according to eq. 4 in [Vercauteren09]
    num = (sigma_x_sq * (F - G))[random_labels >= 3]
    den = (sigma_x_sq * sq_norm_grad_G + sigma_i_sq)[random_labels >= 3]

    # This is $J^{P}$ in eq. 4 [Vercauteren09]
    expected[random_labels >= 3] = -1 * np.array(grad_G[random_labels >= 3])
    expected[random_labels >= 3, ...] *= (num / den)[..., None]

    # Now compute it using the implementation under test

    actual = np.empty_like(expected, dtype=floating)
    em.compute_em_demons_step_2d(np.array(delta_field, dtype=floating),
                                 np.array(sigma_i_sq, dtype=floating),
                                 np.array(grad_G, dtype=floating),
                                 sigma_x_sq,
                                 actual)

    # Test sigma_i_sq == inf
    try:
        assert_array_almost_equal(actual[random_labels == 0],
                                  expected[random_labels == 0])
    except AssertionError:
        raise AssertionError("Failed for sigma_i_sq == inf")

    # Test sigma_i_sq == 0 and gradient != 0
    try:
        assert_array_almost_equal(actual[random_labels == 1],
                                  expected[random_labels == 1])
    except AssertionError:
        raise AssertionError("Failed for sigma_i_sq == 0 and gradient != 0")

    # Test sigma_i_sq == 0 and gradient == 0
    try:
        assert_array_almost_equal(actual[random_labels == 2],
                                  expected[random_labels == 2])
    except AssertionError:
        raise AssertionError("Failed for sigma_i_sq == 0 and gradient == 0")

    # Test sigma_i_sq != 0 and gradient == 0
    try:
        assert_array_almost_equal(actual[random_labels == 3],
                                  expected[random_labels == 3])
    except AssertionError:
        raise AssertionError("Failed for sigma_i_sq != 0 and gradient == 0 ")

    # Test sigma_i_sq != 0 and gradient != 0
    try:
        assert_array_almost_equal(actual[random_labels == 4],
                                  expected[random_labels == 4])
    except AssertionError:
        raise AssertionError("Failed for sigma_i_sq != 0 and gradient != 0")


@set_random_number_generator(1346491)
def test_compute_em_demons_step_3d(rng):
    r"""
    Compares the output of the demons step in 3d against an analytical
    step. The fixed image is given by $F(x) = \frac{1}{2}||x - c_f||^2$, the
    moving image is given by $G(x) = \frac{1}{2}||x - c_g||^2$,
    $x, c_f, c_g \in R^{3}$

    References
    ----------
    [Vercauteren09] Vercauteren, T., Pennec, X., Perchant, A., & Ayache, N.
                    (2009). Diffeomorphic demons: efficient non-parametric
                    image registration. NeuroImage, 45(1 Suppl), S61-72.
                    doi:10.1016/j.neuroimage.2008.10.040
    """

    # Select arbitrary images' shape (same shape for both images)
    sh = (20, 15, 10)

    # Select arbitrary centers
    c_f = np.asarray(sh) / 2
    c_g = c_f + 0.5

    # Compute the identity vector field I(x) = x in R^2
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    x_2 = np.asarray(range(sh[2]))
    X = np.ndarray(sh + (3,), dtype=np.float64)
    O = np.ones(sh)
    X[..., 0] = x_0[:, None, None] * O
    X[..., 1] = x_1[None, :, None] * O
    X[..., 2] = x_2[None, None, :] * O

    # Compute the gradient fields of F and G
    grad_F = X - c_f
    grad_G = X - c_g

    # The squared norm of grad_G to be used later
    sq_norm_grad_G = np.sum(grad_G**2, -1)

    # Compute F and G
    F = 0.5 * np.sum(grad_F**2, -1)
    G = 0.5 * sq_norm_grad_G
    delta_field = G - F

    # Now select an arbitrary parameter for
    # $\sigma_x$ (eq 4 in [Vercauteren09])
    sigma_x_sq = 1.5

    # Set arbitrary values for $\sigma_i$ (eq. 4 in [Vercauteren09])
    # The original Demons algorithm used simply |F(x) - G(x)| as an
    # estimator, so let's use it as well
    sigma_i_sq = (F - G)**2

    # Select some pixels to have special values
    random_labels = rng.integers(0, 5, sh[0] * sh[1] * sh[2])
    random_labels = random_labels.reshape(sh)

    # this label is used to set sigma_i_sq == 0 below
    random_labels[sigma_i_sq == 0] = 2
    # this label is used to set gradient == 0 below
    random_labels[sq_norm_grad_G == 0] = 2

    expected = np.zeros_like(grad_G)
    # Pixels with sigma_i_sq = inf
    sigma_i_sq[random_labels == 0] = np.inf
    expected[random_labels == 0, ...] = 0

    # Pixels with gradient!=0 and sigma_i_sq=0
    sqnrm = sq_norm_grad_G[random_labels == 1]
    sigma_i_sq[random_labels == 1] = 0
    expected[random_labels == 1, 0] = (delta_field[random_labels == 1] *
                                       grad_G[random_labels == 1, 0] / sqnrm)
    expected[random_labels == 1, 1] = (delta_field[random_labels == 1] *
                                       grad_G[random_labels == 1, 1] / sqnrm)
    expected[random_labels == 1, 2] = (delta_field[random_labels == 1] *
                                       grad_G[random_labels == 1, 2] / sqnrm)

    # Pixels with gradient=0 and sigma_i_sq=0
    sigma_i_sq[random_labels == 2] = 0
    grad_G[random_labels == 2, ...] = 0
    expected[random_labels == 2, ...] = 0

    # Pixels with gradient=0 and sigma_i_sq!=0
    grad_G[random_labels == 3, ...] = 0

    # Directly compute the demons step according to eq. 4 in [Vercauteren09]
    num = (sigma_x_sq * (F - G))[random_labels >= 3]
    den = (sigma_x_sq * sq_norm_grad_G + sigma_i_sq)[random_labels >= 3]

    # This is $J^{P}$ in eq. 4 [Vercauteren09]
    expected[random_labels >= 3] = -1 * np.array(grad_G[random_labels >= 3])
    expected[random_labels >= 3, ...] *= (num / den)[..., None]

    # Now compute it using the implementation under test
    actual = np.empty_like(expected, dtype=floating)
    em.compute_em_demons_step_3d(np.array(delta_field, dtype=floating),
                                 np.array(sigma_i_sq, dtype=floating),
                                 np.array(grad_G, dtype=floating),
                                 sigma_x_sq,
                                 actual)

    # Test sigma_i_sq == inf
    try:
        assert_array_almost_equal(actual[random_labels == 0],
                                  expected[random_labels == 0])
    except AssertionError:
        raise AssertionError("Failed for sigma_i_sq == inf")

    # Test sigma_i_sq == 0 and gradient != 0
    try:
        assert_array_almost_equal(actual[random_labels == 1],
                                  expected[random_labels == 1])
    except AssertionError:
        raise AssertionError("Failed for sigma_i_sq == 0 and gradient != 0")

    # Test sigma_i_sq == 0 and gradient == 0
    try:
        assert_array_almost_equal(actual[random_labels == 2],
                                  expected[random_labels == 2])
    except AssertionError:
        raise AssertionError("Failed for sigma_i_sq == 0 and gradient == 0")

    # Test sigma_i_sq != 0 and gradient == 0
    try:
        assert_array_almost_equal(actual[random_labels == 3],
                                  expected[random_labels == 3])
    except AssertionError:
        raise AssertionError("Failed for sigma_i_sq != 0 and gradient == 0 ")

    # Test sigma_i_sq != 0 and gradient != 0
    try:
        assert_array_almost_equal(actual[random_labels == 4],
                                  expected[random_labels == 4])
    except AssertionError:
        raise AssertionError("Failed for sigma_i_sq != 0 and gradient != 0")


@set_random_number_generator(1246592)
def test_quantize_positive_2d(rng):
    # an arbitrary number of quantization levels
    num_levels = 11
    # arbitrary test image shape (must contain at least 3 elements)
    img_shape = (15, 20)
    min_positive = 0.1
    max_positive = 1.0
    epsilon = 1e-8

    delta = (max_positive - min_positive + epsilon) / (num_levels - 1)
    true_levels = np.zeros((num_levels,), dtype=np.float32)
    # put the intensities at the centers of the bins
    true_levels[1:] = np.linspace(min_positive + delta * 0.5,
                                  max_positive - delta * 0.5, num_levels - 1)
    # generate a target quantization image
    true_quantization = np.empty(img_shape, dtype=np.int32)
    random_labels = rng.integers(0, num_levels,
                                      np.size(true_quantization))

    # make sure there is at least one element equal to 0, 1 and num_levels-1
    random_labels[0] = 0
    random_labels[1] = 1
    random_labels[2] = num_levels - 1
    true_quantization[...] = random_labels.reshape(img_shape)

    # make sure additive noise doesn't change the quantization result
    noise_amplitude = np.min([delta / 4.0, min_positive / 4.0])
    sz = np.size(true_quantization)
    noise = rng.random(sz).reshape(img_shape) * noise_amplitude
    noise = noise.astype(floating)
    input_image = np.ndarray(img_shape, dtype=floating)
    # assign intensities plus noise
    input_image[...] = true_levels[true_quantization] + noise
    # preserve original zeros
    input_image[true_quantization == 0] = 0
    # preserve min positive value
    input_image[true_quantization == 1] = min_positive
    # preserve max positive value
    input_image[true_quantization == num_levels - 1] = max_positive

    out, levels, hist = em.quantize_positive_2d(input_image, num_levels)
    levels = np.asarray(levels)
    assert_array_equal(out, true_quantization)
    assert_array_almost_equal(levels, true_levels)
    for i in range(num_levels):
        current_bin = np.asarray(true_quantization == i).sum()
        assert_equal(hist[i], current_bin)

    # test num_levels<2 and input image with zeros and non-zeros everywhere
    assert_raises(ValueError, em.quantize_positive_2d, input_image, 0)
    assert_raises(ValueError, em.quantize_positive_2d, input_image, 1)

    out, levels, hist = em.quantize_positive_2d(
        np.zeros(img_shape, dtype=floating), 2)
    assert_equal(out, np.zeros(img_shape, dtype=np.int32))

    out, levels, hist = em.quantize_positive_2d(
        np.ones(img_shape, dtype=floating), 2)
    assert_equal(out, np.ones(img_shape, dtype=np.int32))


@set_random_number_generator(1246592)
def test_quantize_positive_3d(rng):
    # an arbitrary number of quantization levels
    num_levels = 11
    # arbitrary test image shape (must contain at least 3 elements)
    img_shape = (5, 10, 15)
    min_positive = 0.1
    max_positive = 1.0
    epsilon = 1e-8

    delta = (max_positive - min_positive + epsilon) / (num_levels - 1)
    true_levels = np.zeros((num_levels,), dtype=np.float32)
    # put the intensities at the centers of the bins
    true_levels[1:] = np.linspace(min_positive + delta * 0.5,
                                  max_positive - delta * 0.5,
                                  num_levels - 1)
    # generate a target quantization image
    true_quantization = np.empty(img_shape, dtype=np.int32)
    random_labels = rng.integers(0, num_levels,
                                      np.size(true_quantization))

    # make sure there is at least one element equal to 0, 1 and num_levels-1
    random_labels[0] = 0
    random_labels[1] = 1
    random_labels[2] = num_levels - 1
    true_quantization[...] = random_labels.reshape(img_shape)

    # make sure additive noise doesn't change the quantization result
    noise_amplitude = np.min([delta / 4.0, min_positive / 4.0])
    sz = np.size(true_quantization)
    noise = rng.random(sz).reshape(img_shape) * noise_amplitude
    noise = noise.astype(floating)
    input_image = np.ndarray(img_shape, dtype=floating)
    # assign intensities plus noise
    input_image[...] = true_levels[true_quantization] + noise
    # preserve original zeros
    input_image[true_quantization == 0] = 0
    # preserve min positive value
    input_image[true_quantization == 1] = min_positive
    # preserve max positive value
    input_image[true_quantization == num_levels - 1] = max_positive

    out, levels, hist = em.quantize_positive_3d(input_image, num_levels)
    levels = np.asarray(levels)
    assert_array_equal(out, true_quantization)
    assert_array_almost_equal(levels, true_levels)
    for i in range(num_levels):
        current_bin = np.asarray(true_quantization == i).sum()
        assert_equal(hist[i], current_bin)

    # test num_levels<2 and input image with zeros and non-zeros everywhere
    assert_raises(ValueError, em.quantize_positive_3d, input_image, 0)
    assert_raises(ValueError, em.quantize_positive_3d, input_image, 1)

    out, levels, hist = em.quantize_positive_3d(np.zeros(img_shape,
                                                         dtype=floating), 2)
    assert_equal(out, np.zeros(img_shape, dtype=np.int32))

    out, levels, hist = em.quantize_positive_3d(np.ones(img_shape,
                                                        dtype=floating), 2)
    assert_equal(out, np.ones(img_shape, dtype=np.int32))


@set_random_number_generator(1246592)
def test_compute_masked_class_stats_2d(rng):
    shape = (32, 32)

    # Create random labels
    labels = np.ndarray(shape, dtype=np.int32)
    labels[...] = rng.integers(2, 10, np.size(labels)).reshape(shape)
    # now label 0 is not present and label 1 occurs once
    labels[0, 0] = 1

    # Create random values
    values = rng.standard_normal((shape[0], shape[1])).astype(floating)
    values *= labels
    values += labels

    expected_means = [0, values[0, 0]] + \
        [values[labels == i].mean() for i in range(2, 10)]
    expected_vars = [np.inf, np.inf] + \
        [values[labels == i].var() for i in range(2, 10)]

    mask = np.ones(shape, dtype=np.int32)
    means, std_dev = em.compute_masked_class_stats_2d(mask, values, 10, labels)
    assert_array_almost_equal(means, expected_means, decimal=4)
    assert_array_almost_equal(std_dev, expected_vars, decimal=4)


@set_random_number_generator(1246592)
def test_compute_masked_class_stats_3d(rng):
    shape = (32, 32, 32)

    # Create random labels
    labels = np.ndarray(shape, dtype=np.int32)
    labels[...] = rng.integers(2, 10, np.size(labels)).reshape(shape)

    # now label 0 is not present and label 1 occurs once
    labels[0, 0, 0] = 1

    # Create random values
    values = rng.standard_normal((shape[0], shape[1],
                                  shape[2])).astype(floating)
    values *= labels
    values += labels

    expected_means = [0, values[0, 0, 0]] + \
        [values[labels == i].mean() for i in range(2, 10)]
    expected_vars = [np.inf, np.inf] + \
        [values[labels == i].var() for i in range(2, 10)]

    mask = np.ones(shape, dtype=np.int32)
    means, std_dev = em.compute_masked_class_stats_3d(mask, values, 10, labels)
    assert_array_almost_equal(means, expected_means, decimal=4)
    assert_array_almost_equal(std_dev, expected_vars, decimal=4)
