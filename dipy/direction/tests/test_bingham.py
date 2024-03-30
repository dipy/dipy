import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal, assert_equal, assert_)
from dipy.direction.bingham import (bingham_odf)
from dipy.data import get_sphere

sphere = get_sphere('repulsion724')


def test_bingham_fit():
    """ Tests for bingham function and single Bingham fit"""
    ma_axis = np.array([1, 0, 0])
    mi_axis = np.array([0, 1, 0])
    k1 = 2
    k2 = 6
    f0 = 2

    # Test if maximum amplitude is in the expected Bingham main direction
    # which should be perpendicular to both ma_axis and mi_axis
    odf_test = bingham_odf(f0, k1, k2, ma_axis, mi_axis, np.array([0, 0, 1]))
    assert_almost_equal(odf_test, f0)

    # Test single Bingham fit on full sampled GT Bingham function
    odf_gt = bingham_odf(f0, k1, k2 , ma_axis, mi_axis, sphere.vertices)

