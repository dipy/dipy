"""Tests for dipy.reconst.qti module"""

import numpy as np
import numpy.testing as npt
import dipy.reconst.qti as qti


def test_from_3x3_to_6x1():
    T = np.array([[1, 0, 2],
                  [0, 0, 3],
                  [2, 3, 2]]).astype(float)
    V = np.array([[1, 0, 2, np.sqrt(2) * 3, np.sqrt(2) * 2, 0]]).T
    npt.assert_array_almost_equal(qti.from_3x3_to_6x1(T), V)
    npt.assert_raises(ValueError, qti.from_3x3_to_6x1, T[0:1])
    npt.assert_warns(Warning, qti.from_3x3_to_6x1, T + np.arange(3))
    return


def test_from_6x1_to_3x3():
    T = np.array([[1, 0, 2],
                  [0, 0, 3],
                  [2, 3, 2]]).astype(float)
    V = np.array([[1, 0, 2, np.sqrt(2) * 3, np.sqrt(2) * 2, 0]]).T
    npt.assert_array_almost_equal(qti.from_6x1_to_3x3(V), T)
    npt.assert_raises(ValueError, qti.from_6x1_to_3x3, T)
    return


def test_from_6x6_to_21x1():
    V = np.arange(21).astype(float)
    C = 1 / np.sqrt(2)
    T = np.array(
        [[V[0], C * V[5], C * V[4], C * V[6], C * V[7], C * V[8]],
         [C * V[5], V[1], C * V[3], C * V[9], C * V[10], C * V[11]],
         [C * V[4], C * V[3], V[2], C * V[12], C * V[13], C * V[14]],
         [C * V[6], C * V[9], C * V[12], V[15], C * V[18], C * V[20]],
         [C * V[7], C * V[10], C * V[13], C * V[18], V[16], C * V[19]],
         [C * V[8], C * V[11], C * V[14], C * V[20], C * V[19], V[17]]])
    V = V[:, np.newaxis]
    npt.assert_array_almost_equal(qti.from_6x6_to_21x1(T), V)
    npt.assert_raises(ValueError, qti.from_6x6_to_21x1, T[0:1])
    npt.assert_warns(Warning, qti.from_6x6_to_21x1, T + np.arange(6))
    return


def test_from_21x1_to_6x6():
    V = np.arange(21).astype(float)
    C = 1 / np.sqrt(2)
    T = np.array(
        [[V[0], C * V[5], C * V[4], C * V[6], C * V[7], C * V[8]],
         [C * V[5], V[1], C * V[3], C * V[9], C * V[10], C * V[11]],
         [C * V[4], C * V[3], V[2], C * V[12], C * V[13], C * V[14]],
         [C * V[6], C * V[9], C * V[12], V[15], C * V[18], C * V[20]],
         [C * V[7], C * V[10], C * V[13], C * V[18], V[16], C * V[19]],
         [C * V[8], C * V[11], C * V[14], C * V[20], C * V[19], V[17]]])
    V = V[:, np.newaxis]
    npt.assert_array_almost_equal(qti.from_21x1_to_6x6(V), T)
    npt.assert_raises(ValueError, qti.from_21x1_to_6x6, T)
    return
