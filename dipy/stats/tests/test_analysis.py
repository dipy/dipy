import numpy as np
import numpy.testing as npt

from dipy.stats.analysis import (gaussian_weights, afq_profile)
from dipy.tracking.streamline import Streamlines


def test_gaussian_weights():
    # Some bogus x,y,z coordinates
    x = np.arange(10).astype(float)
    y = np.arange(10).astype(float)
    z = np.arange(10).astype(float)

    # Create a distribution for which we can predict the weights we would
    # expect to get:
    bundle = Streamlines([np.array([x, y, z]).T + 1,
                          np.array([x, y, z]).T - 1])
    # In this case, all nodes receives an equal weight of 0.5:
    w = gaussian_weights(bundle, n_points=10)
    npt.assert_almost_equal(w, np.ones((len(bundle), 10)) * 0.5)

    # Test when asked to return Mahalanobis, instead of weights
    w = gaussian_weights(bundle, n_points=10, return_mahalnobis=True)
    npt.assert_almost_equal(w, np.ones((len(bundle), 10)))

    # Here, some nodes are twice as far from the mean as others
    bundle = Streamlines([np.array([x, y, z]).T + 2,
                          np.array([x, y, z]).T + 1,
                          np.array([x, y, z]).T - 1,
                          np.array([x, y, z]).T - 2])
    w = gaussian_weights(bundle, n_points=10)

    # And their weights should be halved:
    npt.assert_almost_equal(w[0], w[1] / 2)
    npt.assert_almost_equal(w[-1], w[2] / 2)

    # Test the situation where all the streamlines have an identical node:
    arr1 = np.array([x, y, z]).T + 2
    arr2 = np.array([x, y, z]).T + 1
    arr3 = np.array([x, y, z]).T - 1
    arr4 = np.array([x, y, z]).T - 2

    arr1[0] = np.array([1, 1, 1])
    arr2[0] = np.array([1, 1, 1])
    arr3[0] = np.array([1, 1, 1])
    arr4[0] = np.array([1, 1, 1])

    bundle_w_id_node = Streamlines([arr1, arr2, arr3, arr4])
    w = gaussian_weights(Streamlines(bundle_w_id_node), n_points=10)
    # For this case, the result should be a weight of 1/n_streamlines in that
    # node for all streamlines:
    npt.assert_equal(w[:, 0],
                     np.ones(len(bundle_w_id_node)) * 1/len(bundle_w_id_node))

    # Test the situation where all the streamlines are copies of each other:
    bundle_w_copies = Streamlines([bundle[0], bundle[0], bundle[0], bundle[0]])
    w = gaussian_weights(bundle_w_copies, n_points=10)
    # In this case, the entire array should be equal to 1/n_streamlines:
    npt.assert_equal(w,
                     np.ones(w.shape) * 1/len(bundle_w_id_node))

    # Test with bundle of length 1:
    bundle_len_1 = Streamlines([bundle[0]])
    w = gaussian_weights(bundle_len_1, n_points=10)
    npt.assert_equal(w, np.ones(w.shape))

    bundle_len_1 = Streamlines([bundle[0]])
    w = gaussian_weights(bundle_len_1, n_points=10, return_mahalnobis=True)
    npt.assert_equal(w, np.ones(w.shape) * np.nan)


def test_afq_profile():
    data = np.ones((10, 10, 10))
    bundle = Streamlines()
    bundle.extend(np.array([[[0, 0., 0],
                             [1, 0., 0.],
                             [2, 0., 0.]]]))
    bundle.extend(np.array([[[0, 0., 0.],
                             [1, 0., 0],
                             [2, 0,  0.]]]))

    profile = afq_profile(data, bundle, np.eye(4))
    npt.assert_equal(profile, np.ones(100))

    profile = afq_profile(data, bundle, np.eye(4), n_points=10,
                          weights=None)
    npt.assert_equal(profile, np.ones(10))

    profile = afq_profile(data, bundle, np.eye(4),
                          weights=gaussian_weights, stat=np.median)

    npt.assert_equal(profile, np.ones(100))

    profile = afq_profile(data, bundle, np.eye(4), orient_by=bundle[0],
                          weights=gaussian_weights, stat=np.median)

    npt.assert_equal(profile, np.ones(100))

    profile = afq_profile(data, bundle, np.eye(4), n_points=10,
                          weights=None)
    npt.assert_equal(profile, np.ones(10))

    profile = afq_profile(data, bundle, np.eye(4), n_points=10,
                          weights=np.ones((2, 10)) * 0.5)
    npt.assert_equal(profile, np.ones(10))

    profile = afq_profile(data, bundle, np.eye(4), n_points=10,
                          stat=np.median)
    npt.assert_equal(profile, np.ones(10))

    # Disallow setting weights that don't sum to 1 across fibers/nodes:
    npt.assert_raises(ValueError, afq_profile,
                      data, bundle, np.eye(4),
                      n_points=10, weights=np.ones((2, 10)) * 0.6)

    # Test using an affine:
    affine = np.eye(4)
    affine[:, 3] = [-1, 100, -20, 1]
    # Transform the streamlines:
    bundle._data = bundle._data + affine[:3, 3]
    profile = afq_profile(data,
                          bundle,
                          affine,
                          n_points=10,
                          weights=None)

    npt.assert_equal(profile, np.ones(10))

    # Test for error-handling:
    empty_bundle = Streamlines([])
    npt.assert_raises(ValueError, afq_profile, data, empty_bundle, np.eye(4))
