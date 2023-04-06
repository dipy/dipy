import timeit

import numpy as np
from numpy.testing import assert_, assert_almost_equal, assert_array_equal
from dipy.utils.fast_numpy import (random, norm, normalize, dot, cross,
                                   random_vector, random_perpendicular_vector,
                                   random_point_within_circle)


def test_random():
    # Test that random numbers are between 0 and 1 and that the mean is 0.5.
    for _ in range(10):
        vec = random()
        assert_(vec < 1 and vec > 0)

    random_number_list = []
    for _ in range(10000):
        random_number_list.append(random())
    assert_almost_equal(np.mean(random_number_list), 0.5, decimal=1)


def test_norm():
    # Test that the norm equal numpy norm.
    for _ in range(10):
        vec = np.random.random(3)
        assert_almost_equal(norm(vec), np.linalg.norm(vec))


def test_normalize():
    # Test that the normalize vector as a norm of 1.
    for _ in range(10):
        vec = np.random.random(3)
        normalize(vec)
        assert_almost_equal(np.linalg.norm(vec), 1)


def test_dot():
    # Test that dot is faster and equal to numpy.dot.
    for _ in range(10):
        vec1 = np.random.random(3)
        vec2 = np.random.random(3)
        assert_almost_equal(dot(vec1, vec2), np.dot(vec1, vec2))

    vec1 = np.random.random(3)
    vec2 = np.random.random(3)

    def __dot():
        dot(vec1, vec2)

    def __npdot():
        np.dot(vec1, vec2)

    number = 100000
    time_dot = timeit.timeit(__dot, number=number)
    time_npdot = timeit.timeit(__npdot, number=number)
    assert_(time_dot < time_npdot)


def test_cross():
    # Test that cross is faster and equal to numpy.cross.
    out = np.zeros(3, dtype=float)
    for _ in range(10):
        vec1 = np.random.random(3)
        vec2 = np.random.random(3)
        cross(out, vec1, vec2)
        assert_array_equal(out, np.cross(vec1, vec2))

    vec1 = np.random.random(3)
    vec2 = np.random.random(3)

    def __cross():
        cross(out, vec1, vec2)

    def __npcross():
        np.cross(vec1, vec2)

    number = 10000
    time_cross = timeit.timeit(__cross, number=number)
    time_npcross = timeit.timeit(__npcross, number=number)
    assert_(time_cross < time_npcross)


def test_random_vector():
    # Test that that the random vector is of norm 1
    test = np.zeros(3, dtype=float)
    for _ in range(10):
        random_vector(test)
        assert_almost_equal(np.linalg.norm(test), 1)
        assert_(np.all(test >= -1))
        assert_(np.all(test <= 1))


def test_random_perpendicular_vector():
    # Test that the random vector is of norm 1 and perpendicular
    test = np.zeros(3, dtype=float)
    vec = np.zeros(3, dtype=float)
    for _ in range(10):
        random_vector(vec)
        random_perpendicular_vector(test, vec)
        assert_almost_equal(np.linalg.norm(test), 1)
        assert_(np.all(test >= -1))
        assert_(np.all(test <= 1))
        assert_almost_equal(np.dot(vec, test), 0)


def test_random_point_within_circle():
    # Test that the random point is within the circle
    for r in np.arange(0, 5, 0.2):
        pts = random_point_within_circle(r)
        assert_(np.linalg.norm(pts) <= r)
