import numpy as np
from numpy.testing import assert_array_almost_equal
from dipy.align import floating
from dipy.align import crosscorr as cc
from dipy.testing.decorators import set_random_number_generator


def test_cc_factors_2d():
    r"""
    Compares the output of the optimized function to compute the cross-
    correlation factors against a direct (not optimized, but less error prone)
    implementation.
    """
    a = np.array(range(20*20), dtype=floating).reshape(20, 20)
    b = np.array(range(20*20)[::-1], dtype=floating).reshape(20, 20)
    a /= a.max()
    b /= b.max()
    for radius in [0, 1, 3, 6]:
        factors = np.asarray(cc.precompute_cc_factors_2d(a, b, radius))
        expected = np.asarray(cc.precompute_cc_factors_2d_test(a, b, radius))
        assert_array_almost_equal(factors, expected)


def test_cc_factors_3d():
    r"""
    Compares the output of the optimized function to compute the cross-
    correlation factors against a direct (not optimized, but less error prone)
    implementation.
    """
    a = np.array(range(20*20*20), dtype=floating).reshape(20, 20, 20)
    b = np.array(range(20*20*20)[::-1], dtype=floating).reshape(20, 20, 20)
    a /= a.max()
    b /= b.max()
    for radius in [0, 1, 3, 6]:
        factors = np.asarray(cc.precompute_cc_factors_3d(a, b, radius))
        expected = np.asarray(cc.precompute_cc_factors_3d_test(a, b, radius))
        assert_array_almost_equal(factors, expected, decimal=5)


@set_random_number_generator(1147572)
def test_compute_cc_steps_2d(rng):
    # Select arbitrary images' shape (same shape for both images)
    sh = (32, 32)
    radius = 2

    # Select arbitrary centers
    c_f = (np.asarray(sh)/2) + 1.25
    c_g = c_f + 2.5

    # Compute the identity vector field I(x) = x in R^2
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    X = np.ndarray(sh + (2,), dtype=np.float64)
    O = np.ones(sh)
    X[..., 0] = x_0[:, None] * O
    X[..., 1] = x_1[None, :] * O

    # Compute the gradient fields of F and G
    gradF = np.array(X - c_f, dtype=floating)
    gradG = np.array(X - c_g, dtype=floating)

    sz = np.size(gradF)
    Fnoise = rng.random(sz).reshape(gradF.shape) * gradF.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    gradF += Fnoise

    sz = np.size(gradG)
    Gnoise = rng.random(sz).reshape(gradG.shape) * gradG.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    gradG += Gnoise

    sq_norm_grad_G = np.sum(gradG**2, -1)

    F = np.array(0.5*np.sum(gradF**2, -1), dtype=floating)
    G = np.array(0.5*sq_norm_grad_G, dtype=floating)

    Fnoise = rng.random(np.size(F)).reshape(F.shape) * F.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    F += Fnoise

    Gnoise = rng.random(np.size(G)).reshape(G.shape) * G.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    G += Gnoise

    # precompute the cross correlation factors
    factors = cc.precompute_cc_factors_2d_test(F, G, radius)
    factors = np.array(factors, dtype=floating)

    # test the forward step against the exact expression
    I = factors[..., 0]
    J = factors[..., 1]
    sfm = factors[..., 2]
    sff = factors[..., 3]
    smm = factors[..., 4]
    expected = np.ndarray(shape=sh + (2,), dtype=floating)
    factor = (-2.0 * sfm / (sff * smm)) * (J - (sfm / sff) * I)
    expected[..., 0] = factor * gradF[..., 0]
    factor = (-2.0 * sfm / (sff * smm)) * (J - (sfm / sff) * I)
    expected[..., 1] = factor * gradF[..., 1]
    actual, energy = cc.compute_cc_forward_step_2d(gradF, factors, 0)
    assert_array_almost_equal(actual, expected)
    for radius in range(1, 5):
        expected[:radius, ...] = 0
        expected[:, :radius, ...] = 0
        expected[-radius::, ...] = 0
        expected[:, -radius::, ...] = 0
        actual, energy = cc.compute_cc_forward_step_2d(gradF, factors, radius)
        assert_array_almost_equal(actual, expected)

    # test the backward step against the exact expression
    factor = (-2.0 * sfm / (sff * smm)) * (I - (sfm / smm) * J)
    expected[..., 0] = factor * gradG[..., 0]
    factor = (-2.0 * sfm / (sff * smm)) * (I - (sfm / smm) * J)
    expected[..., 1] = factor * gradG[..., 1]
    actual, energy = cc.compute_cc_backward_step_2d(gradG, factors, 0)
    assert_array_almost_equal(actual, expected)
    for radius in range(1, 5):
        expected[:radius, ...] = 0
        expected[:, :radius, ...] = 0
        expected[-radius::, ...] = 0
        expected[:, -radius::, ...] = 0
        actual, energy = cc.compute_cc_backward_step_2d(gradG, factors, radius)
        assert_array_almost_equal(actual, expected)


@set_random_number_generator(12465825)
def test_compute_cc_steps_3d(rng):
    sh = (32, 32, 32)
    radius = 2

    # Select arbitrary centers
    c_f = (np.asarray(sh)/2) + 1.25
    c_g = c_f + 2.5

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
    gradF = np.array(X - c_f, dtype=floating)
    gradG = np.array(X - c_g, dtype=floating)

    sz = np.size(gradF)
    Fnoise = rng.random(sz).reshape(gradF.shape) * gradF.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    gradF += Fnoise

    sz = np.size(gradG)
    Gnoise = rng.random(sz).reshape(gradG.shape) * gradG.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    gradG += Gnoise

    sq_norm_grad_G = np.sum(gradG**2, -1)

    F = np.array(0.5*np.sum(gradF**2, -1), dtype=floating)
    G = np.array(0.5*sq_norm_grad_G, dtype=floating)

    Fnoise = rng.random(np.size(F)).reshape(F.shape) * F.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    F += Fnoise

    Gnoise = rng.random(np.size(G)).reshape(G.shape) * G.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    G += Gnoise

    # precompute the cross correlation factors
    factors = cc.precompute_cc_factors_3d_test(F, G, radius)
    factors = np.array(factors, dtype=floating)

    # test the forward step against the exact expression
    I = factors[..., 0]
    J = factors[..., 1]
    sfm = factors[..., 2]
    sff = factors[..., 3]
    smm = factors[..., 4]
    expected = np.ndarray(shape=sh + (3,), dtype=floating)
    factor = (-2.0 * sfm / (sff * smm)) * (J - (sfm / sff) * I)
    expected[..., 0] = factor * gradF[..., 0]
    expected[..., 1] = factor * gradF[..., 1]
    expected[..., 2] = factor * gradF[..., 2]
    actual, energy = cc.compute_cc_forward_step_3d(gradF, factors, 0)
    assert_array_almost_equal(actual, expected)
    for radius in range(1, 5):
        expected[:radius, ...] = 0
        expected[:, :radius, ...] = 0
        expected[:, :, :radius, :] = 0
        expected[-radius::, ...] = 0
        expected[:, -radius::, ...] = 0
        expected[:, :, -radius::, ...] = 0
        actual, energy = cc.compute_cc_forward_step_3d(gradF, factors, radius)
        assert_array_almost_equal(actual, expected)

    # test the backward step against the exact expression
    factor = (-2.0 * sfm / (sff * smm)) * (I - (sfm / smm) * J)
    expected[..., 0] = factor * gradG[..., 0]
    expected[..., 1] = factor * gradG[..., 1]
    expected[..., 2] = factor * gradG[..., 2]
    actual, energy = cc.compute_cc_backward_step_3d(gradG, factors, 0)
    assert_array_almost_equal(actual, expected)
    for radius in range(1, 5):
        expected[:radius, ...] = 0
        expected[:, :radius, ...] = 0
        expected[:, :, :radius, :] = 0
        expected[-radius::, ...] = 0
        expected[:, -radius::, ...] = 0
        expected[:, :, -radius::, ...] = 0
        actual, energy = cc.compute_cc_backward_step_3d(gradG, factors, radius)
        assert_array_almost_equal(actual, expected)
