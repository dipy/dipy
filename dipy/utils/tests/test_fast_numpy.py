import timeit

import numpy as np
from numpy.testing import assert_, assert_almost_equal
from dipy.utils.fast_numpy import random, random_point_within_circle


def test_random():
    # Test that random numbers are between 0 and 1 and that the mean is 0.5.
    for _ in range(10):
        vec = random()
        assert_(vec < 1 and vec > 0)

    random_number_list = []
    for _ in range(10000):
        random_number_list.append(random())
    assert_almost_equal(np.mean(random_number_list), 0.5, decimal=1)


def test_random_point_within_circle():
    # Test that the random point is within the circle
    for r in np.arange(0, 5, 0.2):
        pts = random_point_within_circle(r)
        assert_(np.linalg.norm(pts) <= r)
