import numpy as np
from numpy.testing import (run_module_suite,
                           assert_equal,
                           assert_raises,
                           assert_array_almost_equal)
from dipy.denoise.denspeed import nlmeans_3d


def test_nlmeans():

    A = 100 * np.zeros((50, 50, 50)) #+ 5 * np.random.rand(50, 50, 50)

    assert_raises(ValueError, nlmeans_3d, A, sigma=5)

    A = 100 + np.zeros((50, 50, 50)) #+ 5 * np.random.rand(50, 50, 50)

    B = nlmeans_3d(A, sigma=1)#np.std(A))

    assert_array_almost_equal(A, B, 2)

    figure(1)

    imshow(A[..., 25])

    figure(2)

    imshow(B[..., 25])

    1/0

test_nlmeans()
