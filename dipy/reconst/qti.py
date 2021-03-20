"""Classes and functions for fitting the covariance tensor model of q-space
trajectory imaging."""

import numpy as np


def from_3x3_to_6x1(T):
    """Convert a symmetric 3 x 3 tensor into a 6 x 1 tensor.

    Parameters
    ----------
    T : np.ndarray
        Symmetric 3 x 3 tensor.

    Returns
    -------    
    V : np.ndarray
        T converted into a 6 x 1 tensor.
    """
    if T.shape != (3, 3):
        raise ValueError('The shape of the input array must be (3, 3).')
    if not np.all(np.isclose(T, T.T)):
        raise ValueError('The input array must be symmetric.')
    C = np.sqrt(2)
    V = np.array([[T[0, 0],
                   T[1, 1],
                   T[2, 2],
                   C * T[1, 2],
                   C * T[0, 2],
                   C * T[0, 1]]]).T
    return V


def from_6x1_to_3x3(V):
    """Convert a 6 x 1 tensor into a symmetric 3 x 3 tensor.

    Parameters
    ----------
    V : np.ndarray
        6 x 1 tensor.

    Returns
    ------- 
    T : np.ndarray
        V converted into a symmetric 3 x 3 tensor.
    """
    if V.shape != (6, 1):
        raise ValueError('The shape of the input array must be (6, 1).')
    C = 1 / np.sqrt(2)
    T = np.array([[V[0, 0], C * V[5, 0], C * V[4, 0]],
                  [C * V[5, 0], V[1, 0], C * V[3, 0]],
                  [C * V[4, 0], C * V[3, 0], V[2, 0]]])
    return T


def from_6x6_to_21x1(T):
    """Convert a symmetric 6 x 6 tensor into a 21 x 1 tensor.

    Parameters
    ----------
    T : np.ndarray
        Symmetric 6 x 1 tensor.

    Returns
    ------- 
    V : np.ndarray
        T converted into a 21 x 1 tensor.
    """
    if T.shape != (6, 6):
        raise ValueError('The shape of the input array must be (6, 6).')
    if not np.all(np.isclose(T, T.T)):
        raise ValueError('The input array must be symmetric.')
    C = np.sqrt(2)
    V = np.array([[T[0, 0], T[1, 1], T[2, 2],
                   C * T[1, 2], C * T[0, 2], C * T[0, 1],
                   C * T[0, 3], C * T[0, 4], C * T[0, 5],
                   C * T[1, 3], C * T[1, 4], C * T[1, 5],
                   C * T[2, 3], C * T[2, 4], C * T[2, 5],
                   T[3, 3], T[4, 4], T[5, 5],
                   C * T[3, 4], C * T[4, 5], C * T[5, 3]]]).T
    return V


def from_21x1_to_6x6(V):
    """Convert a 21 x 1 tensor into a symmetric 6 x 6 tensor.

    Parameters
    ----------
    V : np.ndarray
        21 x 1 tensor.

    Returns
    ------- 
    T : np.ndarray
        V converted into a symmetric 6 x 6 tensor.
    """
    if V.shape != (21, 1):
        raise ValueError('The shape of the input array must be (21, 1).')
    V = V[:, 0]  # Code is easier to read without extra dimension
    C = 1 / np.sqrt(2)
    T = np.array(
        [[V[0], C * V[5], C * V[4], C * V[6], C * V[7], C * V[8]],
         [C * V[5], V[1], C * V[3], C * V[9], C * V[10], C * V[11]],
         [C * V[4], C * V[3], V[2], C * V[12], C * V[13], C * V[14]],
         [C * V[6], C * V[9], C * V[12], V[15], C * V[18], C * V[20]],
         [C * V[7], C * V[10], C * V[13], C * V[18], V[16], C * V[19]],
         [C * V[8], C * V[11], C * V[14], C * V[20], C * V[19], V[17]]])
    return T
