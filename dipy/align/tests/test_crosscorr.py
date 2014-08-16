import numpy as np
from dipy.align import floating
from numpy.testing import (assert_equal,
                           assert_almost_equal, 
                           assert_array_equal,
                           assert_array_almost_equal)
import dipy.align.crosscorr as cc

def test_cc_factors_2d():
    r"""
    Compares the output of the optimized function to compute the cross-
    correlation factors against a direct (not optimized, but less error prone)
    implementation. 
    """
    a = np.array(range(20*20), dtype=floating).reshape(20,20)
    b = np.array(range(20*20)[::-1], dtype=floating).reshape(20,20)
    factors = np.asarray(cc.precompute_cc_factors_2d(a,b,3))
    expected = np.asarray(cc.precompute_cc_factors_2d_test(a,b,3))
    assert_array_almost_equal(factors, expected)


def test_cc_factors_3d():
    r"""
    Compares the output of the optimized function to compute the cross-
    correlation factors against a direct (not optimized, but less error prone)
    implementation. 
    """
    a = np.array(range(20*20*20), dtype=floating).reshape(20,20,20)
    b = np.array(range(20*20*20)[::-1], dtype=floating).reshape(20,20,20)
    factors = np.asarray(cc.precompute_cc_factors_3d(a,b,3))
    expected = np.asarray(cc.precompute_cc_factors_3d_test(a,b,3))
    assert_array_almost_equal(factors, expected)


def test_compute_cc_steps_2d():
    #Select arbitrary images' shape (same shape for both images)
    sh = (32, 32)
    radius = 2

    #Select arbitrary centers
    c_f = (np.asarray(sh)/2) + 1.25
    c_g = c_f + 2.5

    #Compute the identity vector field I(x) = x in R^2
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    X = np.ndarray(sh + (2,), dtype = np.float64)
    O = np.ones(sh)
    X[...,0]= x_0[:, None] * O
    X[...,1]= x_1[None, :] * O

    #Compute the gradient fields of F and G
    grad_F = np.array(X - c_f, dtype = floating)
    grad_G = np.array(X - c_g, dtype = floating)

    sq_norm_grad_G = np.sum(grad_G**2,-1) 
    
    F = np.array(0.5*np.sum(grad_F**2,-1), dtype = floating)
    G = np.array(0.5*sq_norm_grad_G, dtype = floating)

    #precompute the cross correlation factors
    factors = cc.precompute_cc_factors_2d_test(F, G, radius)
    factors = np.array(factors, dtype = floating)

    #test the forward step against the exact expression
    I = factors[..., 0]
    J = factors[..., 1]
    sfm = factors[..., 2]
    sff = factors[..., 3]
    smm = factors[..., 4]
    expected = np.ndarray(shape = sh + (2,), dtype = floating)
    expected[...,0] = (-2.0 * sfm / (sff * smm)) * (J - (sfm / sff) * I) * grad_F[..., 0]
    expected[...,1] = (-2.0 * sfm / (sff * smm)) * (J - (sfm / sff) * I) * grad_F[..., 1]
    actual, energy = cc.compute_cc_forward_step_2d(grad_F, grad_G, factors)
    assert_array_almost_equal(actual, expected)

    #test the backward step against the exact expression
    expected[...,0] = (-2.0 * sfm / (sff * smm)) * (I - (sfm / smm) * J) * grad_G[..., 0]
    expected[...,1] = (-2.0 * sfm / (sff * smm)) * (I - (sfm / smm) * J) * grad_G[..., 1]
    actual, energy = cc.compute_cc_backward_step_2d(grad_F, grad_G, factors)
    assert_array_almost_equal(actual, expected)


def test_compute_cc_steps_3d():
    sh = (32, 32, 32)
    radius = 2

    #Select arbitrary centers
    c_f = (np.asarray(sh)/2) + 1.25
    c_g = c_f + 2.5

    #Compute the identity vector field I(x) = x in R^2
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    x_2 = np.asarray(range(sh[2]))
    X = np.ndarray(sh + (3,), dtype = np.float64)
    O = np.ones(sh)
    X[...,0]= x_0[:, None, None] * O
    X[...,1]= x_1[None, :, None] * O
    X[...,2]= x_2[None, None, :] * O
    #Compute the gradient fields of F and G
    grad_F = np.array(X - c_f, dtype = floating)
    grad_G = np.array(X - c_g, dtype = floating)

    sq_norm_grad_G = np.sum(grad_G**2,-1) 
    
    F = np.array(0.5*np.sum(grad_F**2,-1), dtype = floating)
    G = np.array(0.5*sq_norm_grad_G, dtype = floating)

    #precompute the cross correlation factors
    factors = cc.precompute_cc_factors_3d_test(F, G, radius)
    factors = np.array(factors, dtype = floating)

    #test the forward step against the exact expression
    I = factors[..., 0]
    J = factors[..., 1]
    sfm = factors[..., 2]
    sff = factors[..., 3]
    smm = factors[..., 4]
    expected = np.ndarray(shape = sh + (3,), dtype = floating)
    expected[...,0] = (-2.0 * sfm / (sff * smm)) * (J - (sfm / sff) * I) * grad_F[..., 0]
    expected[...,1] = (-2.0 * sfm / (sff * smm)) * (J - (sfm / sff) * I) * grad_F[..., 1]
    expected[...,2] = (-2.0 * sfm / (sff * smm)) * (J - (sfm / sff) * I) * grad_F[..., 2]
    actual, energy = cc.compute_cc_forward_step_3d(grad_F, grad_G, factors)
    assert_array_almost_equal(actual, expected)

    #test the backward step against the exact expression
    expected[...,0] = (-2.0 * sfm / (sff * smm)) * (I - (sfm / smm) * J) * grad_G[..., 0]
    expected[...,1] = (-2.0 * sfm / (sff * smm)) * (I - (sfm / smm) * J) * grad_G[..., 1]
    expected[...,2] = (-2.0 * sfm / (sff * smm)) * (I - (sfm / smm) * J) * grad_G[..., 2]
    actual, energy = cc.compute_cc_backward_step_3d(grad_F, grad_G, factors)
    assert_array_almost_equal(actual, expected)


if __name__=='__main__':
    test_cc_factors_2d()
    test_cc_factors_3d()
    test_compute_cc_steps_2d()
    test_compute_cc_steps_3d()
