from dipy.align.transforms import *
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           assert_almost_equal,
                           assert_equal)


def test_number_of_parameters():
    transforms = ['TRANSLATION', 'ROTATION', 'RIGID', 'SCALING', 'AFFINE']
    expected_2d = [2, 1, 3, 1, 6]
    expected_3d = [3, 3, 6, 1, 12]

    actual_2d = []
    actual_3d = []
    for transform in transforms:
        t = transform_type[transform]
        actual_2d.append(number_of_parameters(t, 2))
        actual_3d.append(number_of_parameters(t, 3))

    assert_array_equal(actual_2d, expected_2d)
    assert_array_equal(actual_3d, expected_3d)


def test_param_to_matrix_2d():
    # Test translation matrix 2D
    dim = 2
    ttype = transform_type['TRANSLATION']
    dx, dy = np.random.rand(), np.random.rand()
    theta = np.array([dx, dy])
    expected = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
    actual = np.empty_like(expected)
    param_to_matrix(ttype, dim, theta, actual)
    assert_array_equal(actual, expected)

    # Test rotation matrix 2D
    dim = 2
    ttype = transform_type['ROTATION']
    angle = np.random.rand()
    theta = np.array([angle])
    ct = np.cos(angle)
    st = np.sin(angle)
    expected = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
    actual = np.empty_like(expected)
    param_to_matrix(ttype, dim, theta, actual)
    assert_array_almost_equal(actual, expected)

    # Test rigid matrix 2D
    dim = 2
    ttype = transform_type['RIGID']
    angle, dx, dy = np.random.rand(), np.random.rand(), np.random.rand()
    theta = np.array([angle, dx, dy])
    ct = np.cos(angle)
    st = np.sin(angle)
    expected = np.array([[ct, -st, dx], [st, ct, dy], [0, 0, 1]])
    actual = np.empty_like(expected)
    param_to_matrix(ttype, dim, theta, actual)
    assert_array_almost_equal(actual, expected)

    # Test rigid matrix 2D
    dim = 2
    ttype = transform_type['SCALING']
    factor = np.random.rand()
    theta = np.array([factor])
    expected = np.array([[factor, 0, 0], [0, factor, 0], [0, 0, 1]])
    actual = np.empty_like(expected)
    param_to_matrix(ttype, dim, theta, actual)
    assert_array_almost_equal(actual, expected)

    # Test affine 2D
    dim = 2
    ttype = transform_type['AFFINE']
    theta = np.random.rand(6)
    expected = np.eye(3)
    expected[0,:] = theta[:3]
    expected[1,:] = theta[3:6]
    actual = np.empty_like(expected)
    param_to_matrix(ttype, dim, theta, actual)
    assert_array_almost_equal(actual, expected)


def test_param_to_matrix_3d():
    # Test translation matrix 3D
    dim = 3
    ttype = transform_type['TRANSLATION']
    dx, dy, dz = np.random.rand(3)
    theta = np.array([dx, dy, dz])
    expected = np.array([[1, 0, 0, dx],
                         [0, 1, 0, dy],
                         [0, 0, 1, dz],
                         [0, 0, 0, 1]])
    actual = np.empty_like(expected)
    param_to_matrix(ttype, dim, theta, actual)
    assert_array_equal(actual, expected)

    # Test rotation matrix 3D
    dim = 3
    ttype = transform_type['ROTATION']
    theta = np.random.rand(3)
    ca = np.cos(theta[0])
    sa = np.sin(theta[0])
    cb = np.cos(theta[1])
    sb = np.sin(theta[1])
    cc = np.cos(theta[2])
    sc = np.sin(theta[2])

    X = np.array([[1,  0,  0 ],
                  [0, ca, -sa],
                  [0, sa,  ca]])
    Y = np.array([[cb,  0, sb],
                  [0,   1,  0],
                  [-sb, 0, cb]])
    Z = np.array([[cc, -sc, 0],
                  [sc,  cc, 0],
                  [0 ,   0, 1]])

    R = Z.dot(X.dot(Y)) # Apply in order: Y, X, Z (Y goes to the right)
    expected = np.eye(dim + 1)
    expected[:3, :3] = R[:3, :3]
    actual = np.empty_like(expected)
    param_to_matrix(ttype, dim, theta, actual)
    assert_array_almost_equal(actual, expected)

    # Test rigid matrix 3D
    dim = 3
    ttype = transform_type['RIGID']
    theta = np.random.rand(6)
    ca = np.cos(theta[0])
    sa = np.sin(theta[0])
    cb = np.cos(theta[1])
    sb = np.sin(theta[1])
    cc = np.cos(theta[2])
    sc = np.sin(theta[2])

    X = np.array([[1,  0,  0 ],
                  [0, ca, -sa],
                  [0, sa,  ca]])
    Y = np.array([[cb,  0, sb],
                  [0,   1,  0],
                  [-sb, 0, cb]])
    Z = np.array([[cc, -sc, 0],
                  [sc,  cc, 0],
                  [0 ,   0, 1]])

    R = Z.dot(X.dot(Y)) # Apply in order: Y, X, Z (Y goes to the right)
    expected = np.eye(dim + 1)
    expected[:3, :3] = R[:3, :3]
    expected[:3, 3] = theta[3:6]
    actual = np.empty_like(expected)
    param_to_matrix(ttype, dim, theta, actual)
    assert_array_almost_equal(actual, expected)

    # Test scaling matrix 2D
    dim = 3
    ttype = transform_type['SCALING']
    factor = np.random.rand()
    theta = np.array([factor])
    expected = np.array([[factor, 0, 0, 0],
                         [0, factor, 0, 0],
                         [0, 0, factor, 0],
                         [0, 0, 0, 1]])
    actual = np.empty_like(expected)
    param_to_matrix(ttype, dim, theta, actual)
    assert_array_almost_equal(actual, expected)

    # Test affine 2D
    dim = 3
    ttype = transform_type['AFFINE']
    theta = np.random.rand(12)
    expected = np.eye(4)
    expected[0,:] = theta[:4]
    expected[1,:] = theta[4:8]
    expected[2,:] = theta[8:12]
    actual = np.empty_like(expected)
    param_to_matrix(ttype, dim, theta, actual)
    assert_array_almost_equal(actual, expected)


def test_get_identity_parameters():
    transforms = transform_type.keys()
    for dim in [2, 3]:
        for transform in transforms:
            t = transform_type[transform]
            n = number_of_parameters(t, dim)
            theta = np.empty(n)
            get_identity_parameters(t, dim, theta)

            expected = np.eye(dim + 1)
            actual = np.empty_like(expected)
            param_to_matrix(t, dim, theta, actual)
            assert_array_almost_equal(actual, expected)


def test_eval_jacobian_function():
    r""" Compare the analytical Jacobians with their numerical approximations
    """
    transforms = transform_type.keys()
    h = 1e-8
    nsamples = 50

    for dim in [2, 3]:
        for transform in transforms:
            ttype = transform_type[transform]
            n = number_of_parameters(ttype, dim)

            expected = np.empty((dim, n))
            actual = np.empty((dim, n))
            theta = np.random.rand(n)
            T = np.ndarray((dim+1, dim+1))
            param_to_matrix(ttype, dim, theta, T)

            for j in range(nsamples):
                x = 255 * (np.random.rand(dim+1) - 0.5)
                x[dim] = 1
                eval_jacobian_function(ttype, dim, theta, x, actual)

                # Approximate with finite differences
                for i in range(n):
                    dtheta = theta.copy()
                    dtheta[i] += h
                    dT = np.empty_like(T)
                    param_to_matrix(ttype, dim, dtheta, dT)
                    g = (dT - T).dot(x) / h
                    expected[:,i] = g[:dim]

                assert_array_almost_equal(actual, expected, decimal=5)


if __name__=='__main__':
    test_number_of_parameters()
    test_eval_jacobian_function()
    test_param_to_matrix_2d()
    test_param_to_matrix_3d()
    test_get_identity_parameters()
