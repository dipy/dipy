import numpy as np
from nibabel.affines import apply_affine
from dipy.tracking.distances import bundles_distances_mdf
from scipy.optimize import fmin_powell


def rotation_vec2mat(r):
    """ R = rotation_vec2mat(r)

    The rotation matrix is given by the Rodrigues formula:

    R = Id + sin(theta)*Sn + (1-cos(theta))*Sn^2

    with:

           0  -nz  ny
    Sn =   nz   0 -nx
          -ny  nx   0

    where n = r / ||r||

    In case the angle ||r|| is very small, the above formula may lead
    to numerical instabilities. We instead use a Taylor expansion
    around theta=0:

    R = I + sin(theta)/tetha Sr + (1-cos(theta))/teta2 Sr^2

    leading to:

    R = I + (1-theta2/6)*Sr + (1/2-theta2/24)*Sr^2
    """
    theta = np.linalg.norm(r)
    if theta > 1e-30:
        n = r / theta
        Sn = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
        R = np.eye(3) + np.sin(theta) * Sn + \
            (1 - np.cos(theta)) * np.dot(Sn, Sn)
    else:
        Sr = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        theta2 = theta * theta
        R = np.eye(3) + (1 - theta2 / 6.) * \
            Sr + (.5 - theta2 / 24.) * np.dot(Sr, Sr)
    return R


def matrix44(t, dtype=np.double):
    """ Compose a 4x4 transformation matrix

    Parameters
    -----------
    t : ndarray
        t is a vector of of affine transformation parameters with 
        size at least 6.
        If size < 6, error.
        If size == 6, t is interpreted as translation + rotation.
        If size == 7, t is interpreted as translation + rotation +
        isotropic scaling.
        If 7 < size < 12, error.
        If size >= 12, t is interpreted as translation + rotation +
        scaling + pre-rotation.

    Returns
    -------
    T : ndarray
        
    """
    if isinstance(t, list):
        t = np.array(t)
    size = t.size
    T = np.eye(4, dtype=dtype)

    # Degrees to radians
    rads = np.deg2rad(t[3:6])

    R = rotation_vec2mat(rads)

    if size == 6:
        T[0:3, 0:3] = R
    elif size == 7:
        T[0:3, 0:3] = t[6] * R
    else:
        S = np.diag(np.exp(t[6:9]))
        Q = rotation_vec2mat(t[9:12])
        # Beware: R*s*Q
        T[0:3, 0:3] = np.dot(R, np.dot(S, Q))
    T[0:3, 3] = t[0:3]
    return T


def transform_streamlines(streamlines, mat):
    """ Apply affine transformation to streamlines

    Parameters
    ----------
    streamlines : list
        List of 2D ndarrays of shape[-1]==3

    Returns
    -------
    new_streamlines : list
        List of the transformed 2D ndarrays of shape[-1]==3
    """

    return [apply_affine(mat, s) for s in streamlines]


def mdf_optimization_sum(t, static, moving):
    """ MDF distance optimization function (SUM)

    We minimize the distance between moving streamlines as they align
    with the static streamlines.

    Parameters
    -----------
    t : ndarray
        t is a vector of of affine transformation parameters with 
        size at least 6. If size < 6, returns an error.
        If size == 6, t is interpreted as translation + rotation.
        If size == 7, t is interpreted as translation + rotation +
        isotropic scaling. If 7 < size < 12, error.
        If size >= 12, t is interpreted as translation + rotation +
        scaling + pre-rotation.

    static : list
        Static streamlines

    moving : list
        Moving streamlines. These will be transform to align with
        the static streamlines

    Returns
    -------
    cost: float

    """

    aff = matrix44(t)
    moving = transform_streamlines(moving, aff)
    d01 = bundles_distances_mdf(static, moving)
    return np.sum(d01)


def mdf_optimization_min(t, static, moving):
    """ MDF distance optimization function (SUM)

    We minimize the distance between moving streamlines as they align
    with the static streamlines.

    Parameters
    -----------
    t : ndarray
        t is a vector of of affine transformation parameters with 
        size at least 6. If size < 6, returns an error.
        If size == 6, t is interpreted as translation + rotation.
        If size == 7, t is interpreted as translation + rotation +
        isotropic scaling. If 7 < size < 12, error.
        If size >= 12, t is interpreted as translation + rotation +
        scaling + pre-rotation.

    static : list
        Static streamlines

    moving : list
        Moving streamlines. These will be transform to align with
        the static streamlines

    Returns
    -------
    cost: float

    """

    aff = matrix44(t)
    moving = transform_streamlines(moving, aff)
    d01 = bundles_distances_mdf(static, moving)
    return np.sum(np.min(d01, axis=0)) + np.sum(np.min(d01, axis=1))


def center_streamlines(streamlines):
    """ Move streamlines to the origin 

    Parameters
    ----------
    streamlines : list
        List of 2D ndarrays of shape[-1]==3
            
    Returns
    -------
    new_streamlines : list
        List of 2D ndarrays of shape[-1]==3
    inv_shift : ndarray
        Translation in x,y,z to go back in the initial position

    """
    center = np.mean(np.concatenate(streamlines, axis=0), axis=0)
    return [s - center for s in streamlines], center


class LinearRegistration(object):

    def __init__(self, cost_func, reg_type='rigid', xtol=10 ** (-6),
                 ftol=10 ** (-6), maxiter=10 ** 6):

        self.cost_func = cost_func
        self.xopt = None
        self.xtol = xtol
        self.ftol = ftol
        self.maxiter = maxiter

        if reg_type == 'rigid':
            self.initial = np.zeros(6).tolist()
        if reg_type == 'rigid+scale':
            self.initial = np.zeros(7).tolist()

    def optimize(self):
        self.xopt = fmin_powell(self.cost_func,
                                self.initial,
                                (self.static, self.moving),
                                xtol = self.xtol,
                                ftol = self.ftol,
                                maxiter = self.maxiter)                                

        return self.xopt

    def transform(self, static, moving):
        self.static = static
        self.moving = moving
        xopt = self.optimize()
        mat = matrix44(xopt)
        self.mat = mat
        self.moved = transform_streamlines(self.moving, mat)
        return self.moved
