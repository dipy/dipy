import numpy as np
from numpy.testing import assert_array_almost_equal
from dipy.reconst.wigner import (
    z_rot_mat,
    rot_mat,
    change_of_basis_matrix,
    _cc2rc,
    wigner_d_matrix,
    wigner_D_matrix,
    so3_rfft,
    so3_rifft,
    complex_mm,
)


def test_z_rot_mat():
    # Test for zero angle rotation, should return identity matrix
    angle = np.pi / 2
    l = 1
    expected_90_rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    assert_array_almost_equal(z_rot_mat(angle, l), expected_90_rot)


def test_rot_mat():
    # Test for identity rotation (no rotation)
    alpha, beta, gamma = 0, 0, 0
    l = 2
    J = np.eye(2 * l + 1)
    expected_identity = np.eye(2 * l + 1)
    assert_array_almost_equal(
        rot_mat(alpha, beta, gamma, l, J), expected_identity)


def test_change_of_basis_matrix():
    # Test for no change scenario
    l = 2
    frm = to = ("real", "quantum", "centered", "cs")
    expected_identity = np.eye(2 * l + 1)
    assert_array_almost_equal(
        change_of_basis_matrix(l, frm, to), expected_identity)


def test_cc2rc():
    l = 0
    B = _cc2rc(l)
    expected_matrix = 1
    assert_array_almost_equal(B, expected_matrix)


def test_wigner_d_matrix():
    # Test for beta = 0, should return identity
    l = 2
    beta = 0
    expected_identity = np.eye(2 * l + 1)
    assert_array_almost_equal(wigner_d_matrix(l, beta), expected_identity)


def test_wigner_D_matrix():
    l = 2
    alpha, beta, gamma = 0, 0, 0
    expected_identity = np.eye(2 * l + 1)
    assert_array_almost_equal(wigner_D_matrix(
        l, alpha, beta, gamma), expected_identity)


def test_so3_rfft():
    x = np.random.rand(2, 4, 4, 4)  # Assuming b_in = 2
    transformed = so3_rfft(x)
    # Based on b_out = b_in = 2, and nspec calculation
    expected_shape = (10, 2, 2)
    assert transformed.shape == expected_shape


def test_so3_rifft():
    b_out = 2
    nspec = b_out * (4 * b_out**2 - 1) // 3
    
    # Generate input x with the correct shape
    x = np.random.rand(nspec, 2, 2) + 1j * np.random.rand(nspec, 2, 2)
    
    # Perform the inverse transform
    transformed = so3_rifft(x, b_out=b_out)  
    
    # Assuming b_out = 2
    expected_shape = (2, 4, 4, 4)  
    assert transformed.shape == expected_shape
