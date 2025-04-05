import timeit

cimport numpy as cnp
import numpy as np
from numpy.testing import (assert_, assert_almost_equal, assert_raises,
                            assert_array_equal)
from dipy.utils.fast_numpy import random, seed
from dipy.utils.fast_numpy cimport (
    cross,
    dot,
    norm,
    normalize,
    random_vector,
    random_perpendicular_vector, take,
    random_point_within_circle,
    take,
)


def test_norm():
    # Test that the norm equal numpy norm.
    cdef double[:] vec_view
    for _ in range(10):
        vec_view = vec = np.random.random(3)
        assert_almost_equal(norm(&vec_view[0]), np.linalg.norm(vec))


def test_normalize():
    # Test that the normalize vector as a norm of 1.
    cdef double[:] vec_view
    for _ in range(10):
        vec_view = vec = np.random.random(3)
        normalize(&vec_view[0])
        assert_almost_equal(np.linalg.norm(vec), 1)


def test_dot():
    # Test that dot is faster and equal to numpy.dot.
    cdef double[:] vec_view1
    cdef double[:] vec_view2
    for _ in range(10):
        vec_view1 = vec1 = np.random.random(3)
        vec_view2 = vec2 = np.random.random(3)
        assert_almost_equal(dot(&vec_view1[0], &vec_view2[0]),
            np.dot(vec1, vec2))

    vec_view1 = vec1 = np.random.random(3)
    vec_view2 = vec2 = np.random.random(3)

    def __dot():
        dot(&vec_view1[0], &vec_view2[0])

    def __npdot():
        np.dot(vec1, vec2)

    number = 100000
    time_dot = timeit.timeit(__dot, number=number)
    time_npdot = timeit.timeit(__npdot, number=number)
    assert_(time_dot < time_npdot)


# TODO: Check why this test fails
# def test_cross():
#     # Test that cross is faster and equal to numpy.cross.
#     cdef double[:] vec1
#     cdef double[:] vec2
#     cdef double[:] out
#     out = np.zeros(3, dtype=float)
#     for _ in range(10):
#         vec1 = np.random.random(3)
#         vec2 = np.random.random(3)
#         cross(&out[0], &vec1[0], &vec2[0])
#         assert_array_equal(out, np.cross(vec1, vec2))

#     vec1 = np.random.random(3)
#     vec2 = np.random.random(3)

#     def __cross():
#         cross(&out[0], &vec1[0], &vec2[0])

#     def __npcross():
#         np.cross(vec1, vec2)

#     number = 10000
#     time_cross = timeit.timeit(__cross, number=number)
#     time_npcross = timeit.timeit(__npcross, number=number)
#     assert_(time_cross < time_npcross)


def test_random_vector():
    # Test that that the random vector is of norm 1
    cdef double[:] test = np.zeros(3, dtype=float)
    for _ in range(10):
        random_vector(&test[0])
        assert_almost_equal(np.linalg.norm(test), 1)
        assert_(np.all(test >= np.double(-1)))
        assert_(np.all(test <= np.double(1)))


def test_random_perpendicular_vector():
    # Test that the random vector is of norm 1 and perpendicular
    cdef double[:] test = np.zeros(3, dtype=float)
    cdef double[:] vec = np.zeros(3, dtype=float)
    for _ in range(10):
        random_vector(&vec[0])
        random_perpendicular_vector(&test[0], &vec[0])
        assert_almost_equal(np.linalg.norm(test), 1)
        assert_(np.all(test >= np.double(-1)))
        assert_(np.all(test <= np.double(1)))
        assert_almost_equal(np.dot(vec, test), 0)


def test_random():
    # Test that random numbers are between 0 and 1 and that the mean is 0.5.
    for _ in range(10):
        vec = random()
        assert_(vec < 1 and vec > 0)

    random_number_list = []
    for _ in range(10000):
        random_number_list.append(random())
    assert_almost_equal(np.mean(random_number_list), 0.5, decimal=1)

    # Test the random generator seed input
    for s in [0, 1, np.iinfo(np.uint32).max]:
        seed(s)
    for s in [np.iinfo(np.uint32).max + 1, -1]:
        assert_raises(OverflowError, seed, s)


def test_random_point_within_circle():
    # Test that the random point is within the circle
    for r in np.arange(0, 5, 0.2):
        pts = random_point_within_circle(r)
        assert_(np.linalg.norm(pts) <= r)


def test_take():
    # Test that the take function is equal to numpy.take
    cdef int n_indices = 5
    cdef double[:] odf = np.random.random(10)
    cdef cnp.npy_intp[:] indices = np.random.randint(0, 10, n_indices, dtype=np.intp)
    cdef double[:] values_out = np.zeros(n_indices, dtype=float)
    take(&odf[0], &indices[0], n_indices, &values_out[0])
    assert_array_equal(values_out, np.take(odf, indices))