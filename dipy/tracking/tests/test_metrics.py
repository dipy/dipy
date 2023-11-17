""" Testing track_metrics module """
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_equal, assert_array_almost_equal

from dipy.tracking import metrics as tm
from dipy.tracking import distances as pf
from dipy.utils.deprecator import ExpiredDeprecationError
from dipy.testing.decorators import set_random_number_generator


def test_downsample_deprecated():
    streamline = [np.array([[0, 0, 0], [1, 1, 1]])]
    npt.assert_raises(ExpiredDeprecationError, tm.downsample, streamline, 12)


@set_random_number_generator()
def test_splines(rng):
    # create a helix
    t = np.linspace(0, 1.75*2*np.pi, 100)
    x = np.sin(t)
    y = np.cos(t)
    z = t
    # add noise
    x += rng.normal(scale=0.1, size=x.shape)
    y += rng.normal(scale=0.1, size=y.shape)
    z += rng.normal(scale=0.1, size=z.shape)
    xyz = np.vstack((x, y, z)).T
    # get the B-splines smoothed result
    tm.spline(xyz, 3, 2, -1)


def test_segment_intersection():
    xyz = np.array([[1, 1, 1], [2, 2, 2], [2, 2, 2]])
    center = [10, 4, 10]
    radius = 1
    assert_equal(tm.intersect_sphere(xyz, center, radius), False)
    xyz = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    center = [10, 10, 10]
    radius = 2
    assert_equal(tm.intersect_sphere(xyz, center, radius), False)
    xyz = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    center = [2.1, 2, 2.2]
    radius = 2
    assert_equal(tm.intersect_sphere(xyz, center, radius), True)


def test_normalized_3vec():
    vec = [1, 2, 3]
    l2n = np.sqrt(np.dot(vec, vec))
    assert_array_almost_equal(l2n, pf.norm_3vec(vec))
    nvec = pf.normalized_3vec(vec)
    assert_array_almost_equal(np.array(vec) / l2n, nvec)
    vec = np.array([[1, 2, 3]])
    assert_equal(vec.shape, (1, 3))
    assert_equal(pf.normalized_3vec(vec).shape, (3,))


def test_inner_3vecs():
    vec1 = [1, 2.3, 3]
    vec2 = [2, 3, 4.3]
    assert_array_almost_equal(np.inner(vec1, vec2), pf.inner_3vecs(vec1, vec2))
    vec2 = [2, -3, 4.3]
    assert_array_almost_equal(np.inner(vec1, vec2), pf.inner_3vecs(vec1, vec2))


def test_add_sub_3vecs():
    vec1 = np.array([1, 2.3, 3])
    vec2 = np.array([2, 3, 4.3])
    assert_array_almost_equal(vec1 - vec2, pf.sub_3vecs(vec1, vec2))
    assert_array_almost_equal(vec1 + vec2, pf.add_3vecs(vec1, vec2))
    vec2 = [2, -3, 4.3]
    assert_array_almost_equal(vec1 - vec2, pf.sub_3vecs(vec1, vec2))
    assert_array_almost_equal(vec1 + vec2, pf.add_3vecs(vec1, vec2))


def test_winding():
    t = np.array([[63.90763092, 66.25634766, 74.84692383],
                  [63.19578171, 65.95800018, 74.77872467],
                  [61.79797363, 64.91297913, 75.04083252],
                  [60.22916412, 64.11988068, 75.12763214],
                  [59.47861481, 63.50800323, 75.25228882],
                  [58.29077911, 62.88838959, 75.59411621],
                  [57.40341568, 62.48369217, 75.46385193],
                  [56.08355713, 61.64668274, 75.50260162],
                  [54.88656616, 60.34751129, 75.49420929],
                  [52.57548523, 58.3325882, 76.18450928],
                  [50.99916077, 56.06463623, 76.07842255],
                  [50.2379303, 54.92457962, 76.14080811],
                  [49.29185867, 54.21960449, 76.04216003],
                  [48.56259918, 53.58783722, 75.95063782],
                  [48.13407516, 53.19916534, 75.91035461],
                  [47.29430389, 52.12264252, 76.05912018]], dtype=np.float32)
    assert_equal(np.isnan(tm.winding(t)), False)
