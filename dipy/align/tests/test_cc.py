import numpy as np
from dipy.align import floating
from numpy.testing import (assert_equal,
                           assert_almost_equal, 
                           assert_array_equal,
                           assert_array_almost_equal)


def test_cc_factors_2d():
    r"""
    Compares the output of the optimized function to compute the cross-
    correlation factors against a direct (not optimized, but less error prone)
    implementation. 
    """
    import dipy.align.cc as cc
    a = np.array(range(20*20), dtype = floating).reshape(20,20)
    b = np.array(range(20*20)[::-1], dtype = floating).reshape(20,20)
    factors = np.asarray(cc.precompute_cc_factors_2d(a,b,3))
    expected = np.asarray(cc.precompute_cc_factors_2d_test(a,b,3))
    assert_array_almost_equal(factors, expected)


def test_cc_factors_3d():
    r"""
    Compares the output of the optimized function to compute the cross-
    correlation factors against a direct (not optimized, but less error prone)
    implementation. 
    """
    import dipy.align.cc as cc
    a = np.array(range(20*20*20), dtype = floating).reshape(20,20,20)
    b = np.array(range(20*20*20)[::-1], dtype = floating).reshape(20,20,20)
    factors = np.asarray(cc.precompute_cc_factors_3d(a,b,3))
    expected = np.asarray(cc.precompute_cc_factors_3d_test(a,b,3))
    assert_array_almost_equal(factors, expected)


if __name__=='__main__':
    test_cc_factors_2d()
    test_cc_factors_3d()
