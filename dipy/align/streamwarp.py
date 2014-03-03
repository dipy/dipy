import numpy as np
from nibabel.affines import apply_affine
from dipy.tracking.distances import bundles_distances_mdf
from scipy.optimize import fmin_powell, fmin_l_bfgs_b
from dipy.tracking.metrics import downsample
from dipy.align.bmd import _bundle_minimum_distance_rigid


class StreamlineRigidRegistration(object):

    def __init__(self, similarity=None, xtol=10 ** (-6),
                 ftol=10 ** (-6), maxiter=10 ** 6,
                 disp=False, algorithm='L_BFGS_B', bounds=None, fast=True):

        self.similarity = similarity
        self.xtol = xtol
        self.ftol = ftol
        self.maxiter = maxiter
        self.disp = disp
        self.initial = np.ones(6).tolist()
        self.algorithm = algorithm
        if self.algorithm not in ['Powell', 'L_BFGS_B']:
            raise ValueError('Not approriate algorithm')
        self.bounds = bounds
        self.fast = fast

        if self.similarity is None:
            if self.fast:
                self.similarity = bundle_min_distance_fast
            else:
                self.similarity = bundle_min_distance


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

        static_centered_pts, st_idx = unlist_streamlines(static_centered)
        static_centered_pts = np.ascontiguousarray(static_centered_pts,
                                                   dtype=np.float32)

        moving_centered_pts, mv_idx = unlist_streamlines(moving_centered)

        block_size = st_idx[0]

        if self.fast:
            args = (static_centered_pts, moving_centered_pts, block_size)
        else:
            args = (static_centered, moving_centered)


        if self.algorithm == 'Powell':

            optimum = fmin_powell(self.similarity,
                                  self.initial,
                                  args,
                                  xtol=self.xtol,
                                  ftol=self.ftol,
                                  maxiter=self.maxiter,
                                  full_output=True,
                                  disp=False,
                                  retall=True)

        if self.algorithm == 'L_BFGS_B':

            optimum = fmin_l_bfgs_b(self.similarity,
                                    self.initial,
                                    None,
                                    args,
                                    approx_grad=True,
                                    bounds=self.bounds,
                                    m=100,
                                    factr=10,
                                    pgtol=1e-16,
                                    epsilon=1e-3)


        if self.algorithm == 'Powell':

            xopt, fopt, direc, iterations, funcs, warnflag, allvecs = optimum

            if self.disp:
                print('Powell :')
                print('xopt', xopt)
                print('fopt', fopt)
                print('iter', iterations)
                print('funcs', funcs)
                print('warn', warnflag)

        if self.algorithm == 'L_BFGS_B':

            xopt, fopt, dictionary = optimum
            funcs = dictionary['funcalls']
            warnflag = dictionary['warnflag']
            iterations = dictionary['nit']
            grad = dictionary['grad']

            if self.disp:
                print('L_BFGS_B :')
                print('xopt', xopt)
                print('fopt', fopt)
                print('iter', iterations)
                print('grad', grad)
                print('funcs', funcs)
                print('warn', warnflag)
                print('msg', dictionary['task'])

        #print('static_shift', static_shift)
        #print('moving_shift', moving_shift)

        opt_mat = matrix44(xopt)
        static_mat = matrix44([static_shift[0], static_shift[1],
                               static_shift[2], 0, 0, 0])

        moving_mat = matrix44([-moving_shift[0], -moving_shift[1],
                               -moving_shift[2], 0, 0, 0])

        mat = compose_transformations(moving_mat, opt_mat, static_mat)

        #print('mat', mat)

        mat_history = []

        if self.algorithm == 'Powell':

            for vecs in allvecs:
                mat_history.append(compose_transformations(moving_mat,
                                                           matrix44(vecs),
                                                           static_mat))

        return StreamlineRegistrationMap(mat, xopt, fopt,
                                            mat_history, funcs, iterations)


class StreamlineRegistrationMap(object):

    def __init__(self, matopt, xopt, fopt, matopt_history, funcs, iterations):

        self.matrix = matopt
        self.xopt = xopt
        self.fopt = fopt
        self.matrix_history = matopt_history
        self.funcs = funcs
        self.iterations = iterations

    def transform(self, streamlines):

        return transform_streamlines(streamlines, self.matrix)


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

    R = I + sin(theta)/theta Sr + (1-cos(theta))/teta2 Sr^2

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


def bundle_sum_distance(t, static, moving):
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
    
    rows, cols = d01.shape
    return 0.25 * (np.sum(np.min(d01, axis=0)) / float(cols) +
                   np.sum(np.min(d01, axis=1)) / float(rows)) ** 2


def bundle_min_distance_fast(t, static, moving, block_size):

    aff = matrix44(t)
    moving = np.dot(aff[:3, :3], moving.T).T + aff[:3, 3]
    moving = np.ascontiguousarray(moving, dtype=np.float32)

    rows = static.shape[0] / block_size
    cols = moving.shape[0] / block_size
    d01 = np.zeros((rows, cols), dtype=np.float32)
    
    _bundle_minimum_distance_rigid(static, moving,
                                   rows,
                                   cols,
                                   block_size,
                                   d01)

    return 0.25 * (np.sum(np.min(d01, axis=0)) / float(cols) +
                   np.sum(np.min(d01, axis=1)) / float(rows)) ** 2



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


def unlist_streamlines(streamlines):
    """ Return the streamlines not as a list but as an array and an offset

    Parameters
    ----------
    streamlines: sequence

    Returns
    -------
    points : array
    offsets : array

    """

    points = np.concatenate(streamlines, axis=0)
    offsets = np.zeros(len(streamlines), dtype='i8')

    curr_pos = 0
    prev_pos = 0
    for (i, s) in enumerate(streamlines):

            prev_pos = curr_pos
            curr_pos += s.shape[0]
            points[prev_pos:curr_pos] = s
            offsets[i] = curr_pos

    return points, offsets


def relist_streamlines(points, offsets):
    """ Given a representation of a set of streamlines as a large array and
    an offsets array return the streamlines as a list of smaller arrays.

    Parameters
    -----------
    points : array
    offsets : array

    Returns
    -------
    streamlines: sequence
    """

    streamlines = []

    streamlines.append(points[0 : offsets[0]])

    for i in range(len(offsets) - 1):
        streamlines.append(points[offsets[i] : offsets[i + 1]])

    return streamlines






