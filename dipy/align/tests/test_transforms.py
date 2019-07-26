from dipy.align.transforms import regtransforms, Transform
import numpy as np
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           assert_almost_equal,
                           assert_equal,
                           assert_raises)


def test_number_of_parameters():
    expected_params = {('TRANSLATION', 2): 2,
                       ('TRANSLATION', 3): 3,
                       ('ROTATION', 2): 1,
                       ('ROTATION', 3): 3,
                       ('RIGID', 2): 3,
                       ('RIGID', 3): 6,
                       ('SCALING', 2): 1,
                       ('SCALING', 3): 1,
                       ('AFFINE', 2): 6,
                       ('AFFINE', 3): 12}

    for ttype, transform in regtransforms.items():
        assert_equal(
            transform.get_number_of_parameters(),
            expected_params[ttype])


def test_param_to_matrix_2d():
    rng = np.random.RandomState()
    # Test translation matrix 2D
    transform = regtransforms[('TRANSLATION', 2)]
    dx, dy = rng.uniform(size=(2,))
    theta = np.array([dx, dy])
    expected = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
    actual = transform.param_to_matrix(theta)
    assert_array_equal(actual, expected)

    # Test rotation matrix 2D
    transform = regtransforms[('ROTATION', 2)]
    angle = rng.uniform()
    theta = np.array([angle])
    ct = np.cos(angle)
    st = np.sin(angle)
    expected = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
    actual = transform.param_to_matrix(theta)
    assert_array_almost_equal(actual, expected)

    # Test rigid matrix 2D
    transform = regtransforms[('RIGID', 2)]
    angle, dx, dy = rng.uniform(size=(3,))
    theta = np.array([angle, dx, dy])
    ct = np.cos(angle)
    st = np.sin(angle)
    expected = np.array([[ct, -st, dx], [st, ct, dy], [0, 0, 1]])
    actual = transform.param_to_matrix(theta)
    assert_array_almost_equal(actual, expected)

    # Test rigid matrix 2D
    transform = regtransforms[('SCALING', 2)]
    factor = rng.uniform()
    theta = np.array([factor])
    expected = np.array([[factor, 0, 0], [0, factor, 0], [0, 0, 1]])
    actual = transform.param_to_matrix(theta)
    assert_array_almost_equal(actual, expected)

    # Test affine 2D
    transform = regtransforms[('AFFINE', 2)]
    theta = rng.uniform(size=(6,))
    expected = np.eye(3)
    expected[0, :] = theta[:3]
    expected[1, :] = theta[3:6]
    actual = transform.param_to_matrix(theta)
    assert_array_almost_equal(actual, expected)

    # Verify that ValueError is raised if incorrect number of parameters
    for transform in regtransforms.values():
        n = transform.get_number_of_parameters()
        # Set incorrect number of parameters
        theta = np.zeros(n + 1, dtype=np.float64)
        assert_raises(ValueError, transform.param_to_matrix, theta)


def test_param_to_matrix_3d():
    rng = np.random.RandomState()
    # Test translation matrix 3D
    transform = regtransforms[('TRANSLATION', 3)]
    dx, dy, dz = rng.uniform(size=(3,))
    theta = np.array([dx, dy, dz])
    expected = np.array([[1, 0, 0, dx],
                         [0, 1, 0, dy],
                         [0, 0, 1, dz],
                         [0, 0, 0, 1]])
    actual = transform.param_to_matrix(theta)
    assert_array_equal(actual, expected)

    # Test rotation matrix 3D
    transform = regtransforms[('ROTATION', 3)]
    theta = rng.uniform(size=(3,))
    ca = np.cos(theta[0])
    sa = np.sin(theta[0])
    cb = np.cos(theta[1])
    sb = np.sin(theta[1])
    cc = np.cos(theta[2])
    sc = np.sin(theta[2])

    X = np.array([[1, 0, 0],
                  [0, ca, -sa],
                  [0, sa, ca]])
    Y = np.array([[cb, 0, sb],
                  [0, 1, 0],
                  [-sb, 0, cb]])
    Z = np.array([[cc, -sc, 0],
                  [sc, cc, 0],
                  [0, 0, 1]])

    R = Z.dot(X.dot(Y))  # Apply in order: Y, X, Z (Y goes to the right)
    expected = np.eye(4)
    expected[:3, :3] = R[:3, :3]
    actual = transform.param_to_matrix(theta)
    assert_array_almost_equal(actual, expected)

    # Test rigid matrix 3D
    transform = regtransforms[('RIGID', 3)]
    theta = rng.uniform(size=(6,))
    ca = np.cos(theta[0])
    sa = np.sin(theta[0])
    cb = np.cos(theta[1])
    sb = np.sin(theta[1])
    cc = np.cos(theta[2])
    sc = np.sin(theta[2])

    X = np.array([[1, 0, 0],
                  [0, ca, -sa],
                  [0, sa, ca]])
    Y = np.array([[cb, 0, sb],
                  [0, 1, 0],
                  [-sb, 0, cb]])
    Z = np.array([[cc, -sc, 0],
                  [sc, cc, 0],
                  [0, 0, 1]])

    R = Z.dot(X.dot(Y))  # Apply in order: Y, X, Z (Y goes to the right)
    expected = np.eye(4)
    expected[:3, :3] = R[:3, :3]
    expected[:3, 3] = theta[3:6]
    actual = transform.param_to_matrix(theta)
    assert_array_almost_equal(actual, expected)

    # Test scaling matrix 3D
    transform = regtransforms[('SCALING', 3)]
    factor = rng.uniform()
    theta = np.array([factor])
    expected = np.array([[factor, 0, 0, 0],
                         [0, factor, 0, 0],
                         [0, 0, factor, 0],
                         [0, 0, 0, 1]])
    actual = transform.param_to_matrix(theta)
    assert_array_almost_equal(actual, expected)

    # Test affine 3D
    transform = regtransforms[('AFFINE', 3)]
    theta = rng.uniform(size=(12,))
    expected = np.eye(4)
    expected[0, :] = theta[:4]
    expected[1, :] = theta[4:8]
    expected[2, :] = theta[8:12]
    actual = transform.param_to_matrix(theta)
    assert_array_almost_equal(actual, expected)

    # Verify that ValueError is raised if incorrect number of parameters
    for transform in regtransforms.values():
        n = transform.get_number_of_parameters()
        # Set incorrect number of parameters
        theta = np.zeros(n + 1, dtype=np.float64)
        assert_raises(ValueError, transform.param_to_matrix, theta)


def test_identity_parameters():
    for transform in regtransforms.values():
        dim = transform.get_dim()
        theta = transform.get_identity_parameters()

        expected = np.eye(dim + 1)
        actual = transform.param_to_matrix(theta)
        assert_array_almost_equal(actual, expected)


def test_jacobian_functions():
    rng = np.random.RandomState()
    # Compare the analytical Jacobians with their numerical approximations
    h = 1e-8
    nsamples = 50

    for transform in regtransforms.values():
        n = transform.get_number_of_parameters()
        dim = transform.get_dim()

        expected = np.empty((dim, n))
        theta = rng.uniform(size=(n,))
        T = transform.param_to_matrix(theta)

        for j in range(nsamples):
            x = 255 * (rng.uniform(size=(dim,)) - 0.5)
            actual = transform.jacobian(theta, x)

            # Approximate with finite differences
            x_hom = np.ones(dim + 1)
            x_hom[:dim] = x[:]
            for i in range(n):
                dtheta = theta.copy()
                dtheta[i] += h
                dT = np.array(transform.param_to_matrix(dtheta))
                g = (dT - T).dot(x_hom) / h
                expected[:, i] = g[:dim]

            assert_array_almost_equal(actual, expected, decimal=5)

    # Test ValueError is raised when theta parameter doesn't have the right
    # length
    for transform in regtransforms.values():
        n = transform.get_number_of_parameters()
        # Wrong number of parameters
        theta = np.zeros(n + 1)
        x = np.zeros(dim)
        assert_raises(ValueError, transform.jacobian, theta, x)


def test_invalid_transform():
    # Note: users should not attempt to use the base class Transform:
    # they should get an instance of one of its derived classes from the
    # regtransforms dictionary (the base class is not contained there)
    # If for some reason the user instantiates it and attempts to use it,
    # however, it will raise exceptions when attempting to retrieve its
    # Jacobian, identity parameters or its matrix representation. It will
    # return -1 if queried about its dimension or number of parameters
    transform = Transform()
    theta = np.ndarray(3)
    x = np.ndarray(3)
    assert_raises(ValueError, transform.jacobian, theta, x)
    assert_raises(ValueError, transform.get_identity_parameters)
    assert_raises(ValueError, transform.param_to_matrix, theta)

    expected = -1
    actual = transform.get_number_of_parameters()
    assert_equal(actual, expected)

    actual = transform.get_dim()
    assert_equal(actual, expected)


if __name__ == '__main__':
    test_number_of_parameters()
    test_jacobian_functions()
    test_param_to_matrix_2d()
    test_param_to_matrix_3d()
    test_identity_parameters()
    test_invalid_transform()
