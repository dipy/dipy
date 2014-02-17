import numpy as np
from nibabel.affines import apply_affine
from dipy.tracking.distances import bundles_distances_mdf
from scipy.optimize import fmin_powell, fmin, fmin_l_bfgs_b
from dipy.tracking.metrics import downsample


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
    """ MDF distance optimization function

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


def bundle_min_distance(t, static, moving):

    """ MDF-based pairwise distance optimization function

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
    return (np.sum(np.min(d01, axis=0)) + np.sum(np.min(d01, axis=1))) ** 2



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


class StreamlineRigidRegistration(object):

    def __init__(self, similarity, xtol=10 ** (-6),
                 ftol=10 ** (-6), maxiter=10 ** 6, full_output=False,
                 disp=False, algorithm='powell', bounds=None):

        self.similarity = similarity
        self.xtol = xtol
        self.ftol = ftol
        self.maxiter = maxiter
        self.full_output = full_output
        self.disp = disp
        self.initial = np.zeros(6).tolist()
        self.algorithm = algorithm

        if self.algorithm == 'powell':
            self.fmin = fmin_powell
        if self.algorithm == 'simplex':
            self.fmin = fmin
        if self.algorithm == 'l_bfgs':
            self.fmin = fmin_l_bfgs_b

        self.bounds = bounds

    def optimize(self, static, moving):

        msg = 'need to have the same number of points.'
        if static[0].shape[0] != static[-1].shape[0]:
            raise ValueError('Static streamlines ' + msg)

        if moving[0].shape[0] != moving[-1].shape[0]:
            raise ValueError('Moving streamlines ' + msg)

        if static[0].shape[0] != moving[-1].shape[0]:
            raise ValueError('Static and moving streamlines ' + msg)

        static_centered, static_shift = center_streamlines(static)
        moving_centered, moving_shift = center_streamlines(moving)

        if self.algorithm != 'l_bfgs':

            optimum = self.fmin(self.similarity,
                                self.initial,
                                (static_centered, moving_centered),
                                xtol=self.xtol,
                                ftol=self.ftol,
                                maxiter=self.maxiter,
                                full_output=self.full_output,
                                disp=self.disp,
                                retall=True)

        if self.algorithm == 'l_bfgs':

            optimum = self.fmin(self.similarity,
                                self.initial,
                                None,
                                (static_centered, moving_centered),
                                approx_grad=True,
                                bounds=self.bounds,
                                disp=self.disp)


        if self.full_output:

            if self.algorithm == 'powell':

                xopt, fopt, direc, iterations, funcs, warnflag, allvecs = optimum

            if self.algorithm == 'simplex':

                xopt, fopt, iterations, funcs, warnflag, allvecs = optimum

        else:

            if self.algorithm != 'l_bfgs':

                xopt, fopt, allvecs = optimum[0], None, optimum[1]


        if self.algorithm == 'l_bfgs':

            xopt, fopt, dictionary = optimum
            if self.full_output:
                print('function evaluations', dictionary['funcalls'])
                print('number of iterations', dictionary['nit'])
                print('fopt', fopt)
            allvecs = [xopt]

        opt_mat = matrix44(xopt)
        static_mat = matrix44([static_shift[0], static_shift[1],
                               static_shift[2], 0, 0, 0])

        moving_mat = matrix44([-moving_shift[0], -moving_shift[1],
                               -moving_shift[2], 0, 0, 0])

        mat = compose_transformations(moving_mat, opt_mat, static_mat)

        mat_history = []
        for vecs in allvecs:
            mat_history.append(compose_transformations(moving_mat,
                                                       matrix44(vecs),
                                                       static_mat))

        return StreamlineRegistrationParams(mat, xopt, fopt, mat_history)


class StreamlineRegistrationParams(object):

    def __init__(self, matopt, xopt, fopt, matopt_history):

        self.matrix = matopt
        self.xopt = xopt
        self.fopt = fopt
        self.matrix_history = matopt_history

    def transform(self, streamlines):

        return transform_streamlines(streamlines, self.matrix)


def compose_transformations(*mats):
    """ Compose multiple transformations in one 4x4 matrix

    Parameters
    -----------

    mat1 : array, (4, 4)
    mat2 : array, (4, 4)
    ...
    matN : array, (4, 4)

    Returns
    -------
    matN x mat2 x mat1 : array, (4, 4)
    """

    prev = mats[0]

    for mat in mats[1:]:

        prev = np.dot(mat, prev)

    return prev


def vectorize_streamlines(streamlines, no_pts):
    """ Resample all streamlines to the same number of points
    """
    return [downsample(s, no_pts) for s in streamlines]
