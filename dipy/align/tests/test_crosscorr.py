from time import time
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_equal,
                           run_module_suite)
from dipy.align import floating
from dipy.align import crosscorr as cc
from dipy.denoise.denspeed import cpu_count


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


def test_compute_cc_steps_2d():
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
    np.random.seed(1147572)

    gradF = np.array(X - c_f, dtype=floating)
    gradG = np.array(X - c_g, dtype=floating)

    sz = np.size(gradF)
    Fnoise = np.random.ranf(sz).reshape(gradF.shape) * gradF.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    gradF += Fnoise

    sz = np.size(gradG)
    Gnoise = np.random.ranf(sz).reshape(gradG.shape) * gradG.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    gradG += Gnoise

    sq_norm_grad_G = np.sum(gradG**2, -1)

    F = np.array(0.5*np.sum(gradF**2, -1), dtype=floating)
    G = np.array(0.5*sq_norm_grad_G, dtype=floating)

    Fnoise = np.random.ranf(np.size(F)).reshape(F.shape) * F.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    F += Fnoise

    Gnoise = np.random.ranf(np.size(G)).reshape(G.shape) * G.max() * 0.1
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


def test_compute_cc_steps_3d():
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
    np.random.seed(12465825)
    gradF = np.array(X - c_f, dtype=floating)
    gradG = np.array(X - c_g, dtype=floating)

    sz = np.size(gradF)
    Fnoise = np.random.ranf(sz).reshape(gradF.shape) * gradF.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    gradF += Fnoise

    sz = np.size(gradG)
    Gnoise = np.random.ranf(sz).reshape(gradG.shape) * gradG.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    gradG += Gnoise

    sq_norm_grad_G = np.sum(gradG**2, -1)

    F = np.array(0.5*np.sum(gradF**2, -1), dtype=floating)
    G = np.array(0.5*sq_norm_grad_G, dtype=floating)

    Fnoise = np.random.ranf(np.size(F)).reshape(F.shape) * F.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    F += Fnoise

    Gnoise = np.random.ranf(np.size(G)).reshape(G.shape) * G.max() * 0.1
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


def test_cc_threads():
    radius = 4
    rstate = np.random.RandomState(1234)
    for ndim in [2, 3]:
        if ndim == 2:
            N = 128
            precomp_func = cc.precompute_cc_factors_2d
            forward_func = cc.compute_cc_forward_step_2d
            backward_func = cc.compute_cc_backward_step_2d
        elif ndim == 3:
            N = 48
            precomp_func = cc.precompute_cc_factors_3d
            forward_func = cc.compute_cc_forward_step_3d
            backward_func = cc.compute_cc_backward_step_3d
        # test data
        im_shape = (N, ) * ndim
        static = rstate.standard_normal(im_shape)
        moving = rstate.standard_normal(im_shape)
        grad = rstate.standard_normal(im_shape + (ndim, ))

        print('cpu count %d' % (cpu_count(),))

        print('1')
        t = time()
        factors = precomp_func(static, moving, radius, num_threads=1)
        duration_1core_pre = time() - t
        t = time()
        out_f, energy_f = forward_func(grad, factors, radius, num_threads=1)
        duration_1core_forward = time() - t
        t = time()
        out_b, energy_b = backward_func(grad, factors, radius, num_threads=1)
        duration_1core_backward = time() - t
        print("  pre: {} s".format(duration_1core_pre))
        print("  forward: {} s".format(duration_1core_forward))
        print("  back: {} s".format(duration_1core_backward))

        print('All')
        t = time()
        factors_all = precomp_func(static, moving, radius, num_threads=None)
        duration_all_core_pre = time() - t
        t = time()
        out_f_all, energy_f_all = forward_func(grad, factors_all, radius,
                                               num_threads=None)
        duration_all_core_forward = time() - t
        t = time()
        out_b_all, energy_b_all = backward_func(grad, factors_all, radius,
                                                num_threads=None)
        duration_all_core_backward = time() - t
        print("  pre: {} s".format(duration_all_core_pre))
        print("  forward: {} s".format(duration_all_core_forward))
        print("  back: {} s".format(duration_all_core_backward))

        print('2')
        t = time()
        factors2 = precomp_func(static, moving, radius, num_threads=2)
        duration_2core_pre = time() - t
        t = time()
        out_f2, energy_f2 = forward_func(grad, factors2, radius, num_threads=2)
        duration_2core_forward = time() - t
        t = time()
        out_b2, energy_b2 = backward_func(grad, factors2, radius,
                                          num_threads=2)
        duration_2core_backward = time() - t
        print("  pre: {} s".format(duration_2core_pre))
        print("  forward: {} s".format(duration_2core_forward))
        print("  back: {} s".format(duration_2core_backward))

        # verify same result regardless of threading
        assert_array_almost_equal(factors, factors2)
        assert_array_almost_equal(out_f, out_f2)
        assert_array_almost_equal(out_b, out_b2)
        assert_array_almost_equal(factors, factors_all)
        assert_array_almost_equal(out_f, out_f_all)
        assert_array_almost_equal(out_b, out_b_all)

        # Only verify speedups for the precomputation routine which is the
        # slowest of the three.  For the small sizes tested here, the other
        # routines may not always be faster in the multithreaded case.
        if cpu_count() > 2:
            assert_equal(duration_all_core_pre < duration_2core_pre, True)
            assert_equal(duration_2core_pre < duration_1core_pre, True)

        if cpu_count() == 2:
            assert_equal(duration_2core_pre < duration_1core_pre, True)


if __name__ == '__main__':
    run_module_suite()
