import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal, assert_equal, assert_)
from dipy.direction.bingham import (bingham_odf, bingham_fit_odf,
                                    _bingham_fit_peak)
from dipy.data import get_sphere

sphere = get_sphere('repulsion724')
sphere = sphere.subdivide(2)


def test_bingham_fit():
    """ Tests for bingham function and single Bingham fit"""
    peak_dir = np.array([1, 0, 0])
    ma_axis = np.array([0, 1, 0])
    mi_axis = np.array([0, 0, 1])
    k1 = 2
    k2 = 6
    f0 = 3

    # Test if maximum amplitude is in the expected Bingham main direction
    # which should be perpendicular to both ma_axis and mi_axis
    odf_test = bingham_odf(f0, k1, k2, ma_axis, mi_axis, peak_dir)
    assert_almost_equal(odf_test, f0)

    # Test Bingham fit on full sampled GT Bingham function
    odf_gt = bingham_odf(f0, k1, k2, ma_axis, mi_axis, sphere.vertices)
    a0, c1, c2, mu0, mu1, mu2 = _bingham_fit_peak(odf_gt, peak_dir, sphere, 45)

    # check scalar parameters
    assert_almost_equal(a0, f0, decimal=3)
    assert_almost_equal(c1, k1, decimal=3)
    assert_almost_equal(c2, k2, decimal=3)

    # check if measured peak direction and dispersion axis are aligned to their
    # respective GT
    Mus = np.array([mu0, mu1, mu2])
    Mus_ref = np.array([peak_dir, ma_axis, mi_axis])
    assert_array_almost_equal(np.abs(np.diag(np.dot(Mus, Mus_ref))),
                              np.ones(3))

    # check the same for bingham_fit_odf
    fits, n = bingham_fit_odf(odf_gt, sphere, max_search_angle=45)
    assert_almost_equal(fits[0][0], f0, decimal=3)
    assert_almost_equal(fits[0][1], k1, decimal=3)
    assert_almost_equal(fits[0][2], k2, decimal=3)
    Mus = np.array([fits[0][3], fits[0][4], fits[0][5]])
    # I had to decrease the precision in the assert below because main peak
    # direction is now calculated (before the GT direction was given)
    assert_array_almost_equal(np.abs(np.diag(np.dot(Mus, Mus_ref))),
                              np.ones(3), decimal=5)
