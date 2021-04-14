"""Tests for dipy.reconst.qti module"""

import numpy as np
import numpy.testing as npt

import dipy.reconst.qti as qti
from dipy.core.gradients import GradientTable
from dipy.sims.voxel import vec2vec_rotmat


def test_from_3x3_to_6x1():
    V = np.arange(1, 7)[:, np.newaxis].astype(float)
    T = np.array(([1, 4.24264069, 3.53553391],
                  [4.24264069, 2, 2.82842712],
                  [3.53553391, 2.82842712, 3]))
    npt.assert_array_almost_equal(qti.from_3x3_to_6x1(T), V)
    npt.assert_array_almost_equal(
        qti.from_3x3_to_6x1(qti.from_6x1_to_3x3(V)), V)
    npt.assert_raises(ValueError, qti.from_3x3_to_6x1, T[0:1])
    npt.assert_warns(Warning, qti.from_3x3_to_6x1, T + np.arange(3))
    return


def test_from_6x1_to_3x3():
    V = np.arange(1, 7)[:, np.newaxis].astype(float)
    T = np.array(([1, 4.24264069, 3.53553391],
                  [4.24264069, 2, 2.82842712],
                  [3.53553391, 2.82842712, 3]))
    npt.assert_array_almost_equal(qti.from_6x1_to_3x3(V), T)
    npt.assert_array_almost_equal(
        qti.from_6x1_to_3x3(qti.from_3x3_to_6x1(T)), T)
    npt.assert_raises(ValueError, qti.from_6x1_to_3x3, T)
    return


def test_from_6x6_to_21x1():
    V = np.arange(1, 22)[:, np.newaxis].astype(float)
    T = np.array((
        [1, 4.24264069, 3.53553391, 4.94974747, 5.65685425, 6.36396103],
        [4.24264069, 2, 2.82842712, 7.07106781, 7.77817459, 8.48528137],
        [3.53553391, 2.82842712, 3, 9.19238816, 9.89949494, 10.60660172],
        [4.94974747, 7.07106781, 9.19238816, 16, 13.43502884, 14.8492424],
        [5.65685425, 7.77817459, 9.89949494, 13.43502884, 17, 14.14213562],
        [6.36396103, 8.48528137, 10.60660172, 14.8492424, 14.14213562, 18]))
    npt.assert_array_almost_equal(qti.from_6x6_to_21x1(T), V)
    npt.assert_array_almost_equal(
        qti.from_6x6_to_21x1(qti.from_21x1_to_6x6(V)), V)
    npt.assert_raises(ValueError, qti.from_6x6_to_21x1, T[0:1])
    npt.assert_warns(Warning, qti.from_6x6_to_21x1, T + np.arange(6))
    return


def test_from_21x1_to_6x6():
    V = np.arange(1, 22)[:, np.newaxis].astype(float)
    T = np.array((
        [1, 4.24264069, 3.53553391, 4.94974747, 5.65685425, 6.36396103],
        [4.24264069, 2, 2.82842712, 7.07106781, 7.77817459, 8.48528137],
        [3.53553391, 2.82842712, 3, 9.19238816, 9.89949494, 10.60660172],
        [4.94974747, 7.07106781, 9.19238816, 16, 13.43502884, 14.8492424],
        [5.65685425, 7.77817459, 9.89949494, 13.43502884, 17, 14.14213562],
        [6.36396103, 8.48528137, 10.60660172, 14.8492424, 14.14213562, 18]))
    npt.assert_array_almost_equal(qti.from_21x1_to_6x6(V), T)
    npt.assert_array_almost_equal(
        qti.from_21x1_to_6x6(qti.from_6x6_to_21x1(T)), T)
    npt.assert_raises(ValueError, qti.from_21x1_to_6x6, T)
    return


def test_helper_tensors():
    npt.assert_array_equal(qti.e_iso, np.eye(3) / 3)
    npt.assert_array_equal(qti.E_iso, np.eye(6) / 3)
    npt.assert_array_equal(
        qti.E_bulk, np.matmul(
            qti.from_3x3_to_6x1(qti.e_iso),
            qti.from_3x3_to_6x1(qti.e_iso).T))
    npt.assert_array_equal(qti.E_shear, qti.E_iso - qti.E_bulk)
    npt.assert_array_equal(qti.E_tsym, qti.E_bulk + .4 * qti.E_shear)
    return


def test_dtd_covariance():

    # Isotropic diffusion tensors with varying sizes
    DTD = np.array(
        [[[1., 0, 0], [0, 1., 0], [0, 0, 1.]],
         [[.5, 0, 0], [0, .5, 0], [0, 0, .5]],
         [[.1, 0, 0], [0, .1, 0], [0, 0, .1]]])
    C = np.zeros((6, 6))
    C[0:3, 0:3] = 0.13555556
    npt.assert_almost_equal(qti.dtd_covariance(DTD), C)

    # Anisotropic diffusion tensors with varying orientations
    DTD = np.zeros((6, 3, 3))
    evals = np.array([1, 0, 0])
    phi = (1 + np.sqrt(5)) / 2
    directions = np.array(
        [[0, 1, phi],
         [0, 1, -phi],
         [1, phi, 0],
         [1, -phi, 0],
         [phi, 0, 1],
         [phi, 0, -1]]) / np.linalg.norm([0, 1, phi])
    for i in range(6):
        R = vec2vec_rotmat(np.array([1, 0, 0]), directions[i])
        DTD[i] = np.matmul(R, np.matmul(np.eye(3) * evals, R.T))
    C = np.eye(6) * 2 / 15
    C[0:3, 0:3] = np.array(
        [[4 / 45, -2 / 45, -2 / 45],
         [-2 / 45, 4 / 45, -2 / 45],
         [-2 / 45, -2 / 45, 4 / 45]])
    npt.assert_almost_equal(qti.dtd_covariance(DTD), C)
    return


def test_qti_signal():
    # Test input validation
    # Test signal generation
    return


def test_design_matrix():
    btens = np.array([np.eye(3, 3) for i in range(3)])
    btens[0, 1, 1] = 0
    btens[0, 2, 2] = 0
    btens[1, 0, 0] = 0
    X = qti.design_matrix(btens)
    npt.assert_almost_equal(X, np.array(
        [[1., 1., 1.],
         [-1., -0., -1.],
         [-0., -1., -1.],
         [-0., -1., -1.],
         [-0., -0., -0.],
         [-0., -0., -0.],
         [-0., -0., -0.],
         [0.5, 0., 0.5],
         [0., 0.5, 0.5],
         [0., 0.5, 0.5],
         [0., 0.70710678, 0.70710678],
         [0., 0., 0.70710678],
         [0., 0., 0.70710678],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]).T)
    return


def test_ols_fit():
    #
    return


def test_ols_fit():
    #
    return


def test_qti_model():
    #
    return


def test_qti_fit():
    #
    return
