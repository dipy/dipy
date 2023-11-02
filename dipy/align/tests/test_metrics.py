import itertools
import numpy as np
from scipy import ndimage
from dipy.align import floating
from dipy.align.metrics import SSDMetric, CCMetric, EMMetric
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           assert_raises)
from dipy.testing.decorators import set_random_number_generator


def test_exceptions():
    for invalid_dim in [-1, 0, 1, 4, 5]:
        assert_raises(ValueError, CCMetric, invalid_dim)
        assert_raises(ValueError, EMMetric, invalid_dim)
        assert_raises(ValueError, SSDMetric, invalid_dim)
    assert_raises(ValueError, SSDMetric, 3, step_type='unknown_metric_name')
    assert_raises(ValueError, EMMetric, 3, step_type='unknown_metric_name')

    def init_metric(shape, radius):
        dim = len(shape)
        metric = CCMetric(dim, radius=radius)
        metric.set_static_image(np.arange(np.prod(shape),
                                          dtype=float).reshape(shape),
                                np.eye(4), np.ones(dim), np.eye(3))
        metric.set_moving_image(np.arange(np.prod(shape),
                                dtype=float).reshape(shape),
                                np.eye(4), np.ones(dim), np.eye(3))
        return metric

    # Generate many shape combinations
    shapes_2d = itertools.product((5, 8), (8, 5))
    shapes_3d = itertools.product((5, 8), (8, 5), (30, 50))
    all_shapes = itertools.chain(shapes_2d, shapes_3d)
    # expected to fail for any dimension < 2*radius + 1.
    for shape in all_shapes:
        metric = init_metric(shape, 4)
        assert_raises(ValueError, metric.initialize_iteration)

    # expected to pass for any dimension == 2*radius + 1.
    metric = init_metric((9, 9), 4)
    metric.initialize_iteration()


@set_random_number_generator(7181309)
def test_EMMetric_image_dynamics(rng):
    metric = EMMetric(2)

    target_shape = (10, 10)
    # create a random image
    image = np.ndarray(target_shape, dtype=floating)
    image[...] = rng.integers(
        0, 10, np.size(image)).reshape(tuple(target_shape))
    # compute the expected binary mask
    expected = (image > 0).astype(np.int32)

    metric.use_static_image_dynamics(image, None)
    assert_array_equal(expected, metric.static_image_mask)

    metric.use_moving_image_dynamics(image, None)
    assert_array_equal(expected, metric.moving_image_mask)


def test_em_demons_step_2d():
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
    sh = (20, 10)

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
    sq_norm_grad_F = np.sum(grad_F**2, -1)
    sq_norm_grad_G = np.sum(grad_G**2, -1)

    # Compute F and G
    F = 0.5 * sq_norm_grad_F
    G = 0.5 * sq_norm_grad_G

    # Create an instance of EMMetric
    metric = EMMetric(2)
    metric.static_spacing = np.array([1.2, 1.2])
    # The $\sigma_x$ (eq. 4 in [Vercauteren09]) parameter is computed in ANTS
    # based on the image's spacing
    sigma_x_sq = np.sum(metric.static_spacing**2) / metric.dim
    # Set arbitrary values for $\sigma_i$ (eq. 4 in [Vercauteren09])
    # The original Demons algorithm used simply |F(x) - G(x)| as an
    # estimator, so let's use it as well
    sigma_i_sq = (F - G)**2
    # Set the properties relevant to the demons methods
    metric.smooth = 3.0
    metric.gradient_static = np.array(grad_F, dtype=floating)
    metric.gradient_moving = np.array(grad_G, dtype=floating)
    metric.static_image = np.array(F, dtype=floating)
    metric.moving_image = np.array(G, dtype=floating)
    metric.staticq_means_field = np.array(F, dtype=floating)
    metric.staticq_sigma_sq_field = np.array(sigma_i_sq, dtype=floating)
    metric.movingq_means_field = np.array(G, dtype=floating)
    metric.movingq_sigma_sq_field = np.array(sigma_i_sq, dtype=floating)

    # compute the step using the implementation under test
    actual_forward = metric.compute_demons_step(True)
    actual_backward = metric.compute_demons_step(False)

    # Now directly compute the demons steps according to eq 4 in
    # [Vercauteren09]
    num_fwd = sigma_x_sq * (G - F)
    den_fwd = sigma_x_sq * sq_norm_grad_F + sigma_i_sq
    # This is $J^{P}$ in eq. 4 [Vercauteren09]
    expected_fwd = -1 * np.array(grad_F)
    expected_fwd[..., 0] *= num_fwd / den_fwd
    expected_fwd[..., 1] *= num_fwd / den_fwd
    # apply Gaussian smoothing
    expected_fwd[..., 0] = ndimage.gaussian_filter(expected_fwd[..., 0], 3.0)
    expected_fwd[..., 1] = ndimage.gaussian_filter(expected_fwd[..., 1], 3.0)

    num_bwd = sigma_x_sq * (F - G)
    den_bwd = sigma_x_sq * sq_norm_grad_G + sigma_i_sq
    # This is $J^{P}$ in eq. 4 [Vercauteren09]
    expected_bwd = -1 * np.array(grad_G)
    expected_bwd[..., 0] *= num_bwd / den_bwd
    expected_bwd[..., 1] *= num_bwd / den_bwd
    # apply Gaussian smoothing
    expected_bwd[..., 0] = ndimage.gaussian_filter(expected_bwd[..., 0], 3.0)
    expected_bwd[..., 1] = ndimage.gaussian_filter(expected_bwd[..., 1], 3.0)

    assert_array_almost_equal(actual_forward, expected_fwd)
    assert_array_almost_equal(actual_backward, expected_bwd)


def test_em_demons_step_3d():
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
    sq_norm_grad_F = np.sum(grad_F**2, -1)
    sq_norm_grad_G = np.sum(grad_G**2, -1)

    # Compute F and G
    F = 0.5 * sq_norm_grad_F
    G = 0.5 * sq_norm_grad_G

    # Create an instance of EMMetric
    metric = EMMetric(3)
    metric.static_spacing = np.array([1.2, 1.2, 1.2])
    # The $\sigma_x$ (eq. 4 in [Vercauteren09]) parameter is computed in ANTS
    # based on the image's spacing
    sigma_x_sq = np.sum(metric.static_spacing**2) / metric.dim
    # Set arbitrary values for $\sigma_i$ (eq. 4 in [Vercauteren09])
    # The original Demons algorithm used simply |F(x) - G(x)| as an
    # estimator, so let's use it as well
    sigma_i_sq = (F - G)**2
    # Set the properties relevant to the demons methods
    metric.smooth = 3.0
    metric.gradient_static = np.array(grad_F, dtype=floating)
    metric.gradient_moving = np.array(grad_G, dtype=floating)
    metric.static_image = np.array(F, dtype=floating)
    metric.moving_image = np.array(G, dtype=floating)
    metric.staticq_means_field = np.array(F, dtype=floating)
    metric.staticq_sigma_sq_field = np.array(sigma_i_sq, dtype=floating)
    metric.movingq_means_field = np.array(G, dtype=floating)
    metric.movingq_sigma_sq_field = np.array(sigma_i_sq, dtype=floating)

    # compute the step using the implementation under test
    actual_forward = metric.compute_demons_step(True)
    actual_backward = metric.compute_demons_step(False)

    # Now directly compute the demons steps according to eq 4 in
    # [Vercauteren09]
    num_fwd = sigma_x_sq * (G - F)
    den_fwd = sigma_x_sq * sq_norm_grad_F + sigma_i_sq
    expected_fwd = -1 * np.array(grad_F)
    expected_fwd[..., 0] *= num_fwd / den_fwd
    expected_fwd[..., 1] *= num_fwd / den_fwd
    expected_fwd[..., 2] *= num_fwd / den_fwd
    # apply Gaussian smoothing
    expected_fwd[..., 0] = ndimage.gaussian_filter(expected_fwd[..., 0], 3.0)
    expected_fwd[..., 1] = ndimage.gaussian_filter(expected_fwd[..., 1], 3.0)
    expected_fwd[..., 2] = ndimage.gaussian_filter(expected_fwd[..., 2], 3.0)

    num_bwd = sigma_x_sq * (F - G)
    den_bwd = sigma_x_sq * sq_norm_grad_G + sigma_i_sq
    expected_bwd = -1 * np.array(grad_G)
    expected_bwd[..., 0] *= num_bwd / den_bwd
    expected_bwd[..., 1] *= num_bwd / den_bwd
    expected_bwd[..., 2] *= num_bwd / den_bwd
    # apply Gaussian smoothing
    expected_bwd[..., 0] = ndimage.gaussian_filter(expected_bwd[..., 0], 3.0)
    expected_bwd[..., 1] = ndimage.gaussian_filter(expected_bwd[..., 1], 3.0)
    expected_bwd[..., 2] = ndimage.gaussian_filter(expected_bwd[..., 2], 3.0)

    assert_array_almost_equal(actual_forward, expected_fwd)
    assert_array_almost_equal(actual_backward, expected_bwd)
