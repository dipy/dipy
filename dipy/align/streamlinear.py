import abc
import numpy as np
from nibabel.affines import apply_affine
from nibabel.quaternions import quat2angle_axis, mat2quat
from scipy.linalg import det
from dipy.core.optimize import Optimizer
from dipy.tracking.metrics import downsample
from dipy.align.bmd import (_bundle_minimum_distance_rigid,
                            _bundle_minimum_distance_rigid_nomat,
                            bundles_distance_matrix_mdf)

MAX_DIST = 1e10
LOG_MAX_DIST = np.log(MAX_DIST)


def threshold(x, th):
    return np.maximum(np.minimum(x, th), -th)


class StreamlineDistanceMetric(object):

    def __init__(self):
        """ An abstract class for the metric used for streamline registration

        If the two sets of streamlines match exactly then method ``distance``
        of this object should be minimum.
        """
        self.static = None
        self.moving = None

    @abc.abstractmethod
    def set_static(self, static):
        self.static = static

    @abc.abstractmethod
    def set_moving(self, moving):
        self.moving = moving

    @abc.abstractmethod
    def distance(self, xopt):
        """ calculate distance for current set of parameters
        """
        return None


class BundleMinDistance(StreamlineDistanceMetric):

    def distance(self, xopt):
        return bundle_min_distance(xopt, self.static, self.moving)


class BundleMinDistanceFast(StreamlineDistanceMetric):

    def set_static(self, static):
        static_centered_pts, st_idx = unlist_streamlines(static)
        self.static_centered_pts = np.ascontiguousarray(static_centered_pts,
                                                        dtype=np.float64)
        self.block_size = st_idx[0]

    def set_moving(self, moving):
        self.moving_centered_pts, mv_idx = unlist_streamlines(moving)

    def distance(self, xopt):
        return bundle_min_distance_fast(xopt,
                                        self.static_centered_pts,
                                        self.moving_centered_pts,
                                        self.block_size)


class BundleSumDistance(StreamlineDistanceMetric):

    def distance(self, xopt):
        return bundle_sum_distance(xopt, self.static, self.moving)


class StreamlineLinearRegistration(object):

    def __init__(self, metric=None, x0=None, method='L-BFGS-B',
                 bounds=None, fast=True, disp=False, options=None,
                 evolution=False):
        r""" Linear registration of 2 sets of streamlines [Garyfallidis14]_.

        Parameters
        ----------
        metric : StreamlineDistanceMetric,
            If None and fast is False then the BMD distance is used. If fast
            is True then a faster implementation of BMD is used. Otherwise,
            use the given distance metric.

        x0 : None or array
            Initial parametrization. If None ``x0=np.ones(6)``.
            If x0 has 6 elements then only translation and rotation is performed
            (rigid). If x0 has 7 elements also isotropic scaling is performed
            (similarity). If x0 has 12 elements then translation, rotation,
            scaling and shearing is performed (affine).

        method : str,
            'L_BFGS_B' or 'Powell' optimizers can be used. Default is 'L_BFGS_B'.

        bounds : list of tuples or None,
            If method == 'L_BFGS_B' then we can use bounded optimization.
            For example for the six parameters of rigid rotation we can set
            the bounds = [(-30, 30), (-30, 30), (-30, 30),
                          (-45, 45), (-45, 45), (-45, 45)]
            That means that we have set the bounds for the three translations
            and three rotation axes (in degrees).

        fast : boolean
            Allows faster execution. Currently works only with rigid
            registration. Default True.

        options : None or dict,
            Extra options to be used with the selected method.

        evolution : boolean
            If True save the transformation for each iteration of the
            optimizer. Default is False. Supported only with Scipy >= 0.11.

        Methods
        -------
        optimize(static, moving)

        References
        ----------
        .. [Garyfallidis14] Garyfallidis et al., "Direct native-space fiber
                            bundle alignment for group comparisons", ISMRM,
                            2014.

        """

        self.metric = metric
        self.x0 = x0
        if self.x0 is None:
            self.x0 = np.ones(6)

        if self.metric is None:
            if fast and (len(self.x0) == 6):
                self.metric = BundleMinDistanceFast()
            else:
                self.metric = BundleMinDistance()

        self.disp = disp
        self.method = method
        if self.method not in ['Powell', 'L-BFGS-B']:
            raise ValueError('Only Powell and L-BFGS-B can be used')
        self.bounds = bounds
        self.fast = fast
        self.options = options
        self.evolution = evolution

    def optimize(self, static, moving):
        """ Find the minimum of the provided metric.

        Parameters
        ----------

        static : streamlines

        moving : streamlines

        Returns
        -------

        map : StreamlineRegistrationMap

        """

        msg = 'need to have the same number of points.'
        if static[0].shape[0] != static[-1].shape[0]:
            raise ValueError('Static streamlines ' + msg)

        if moving[0].shape[0] != moving[-1].shape[0]:
            raise ValueError('Moving streamlines ' + msg)

        if static[0].shape[0] != moving[-1].shape[0]:
            raise ValueError('Static and moving streamlines ' + msg)

        static_centered, static_shift = center_streamlines(static)
        moving_centered, moving_shift = center_streamlines(moving)

        self.metric.set_static(static_centered)
        self.metric.set_moving(moving_centered)

        distance = self.metric.distance

        if self.method == 'Powell':

            if self.options is None:
                self.options = {'xtol': 1e-6, 'ftol':1e-6, 'maxiter':1e6}

            opt = Optimizer(distance, self.x0.tolist(),
                            method=self.method, options=self.options,
                            evolution=self.evolution)

        if self.method == 'L-BFGS-B':

            if self.options is None:
                self.options={'maxcor':10, 'ftol':1e-7, 'gtol':1e-5, 'eps':1e-8}

            opt = Optimizer(distance, self.x0.tolist(),
                            method=self.method,
                            bounds=self.bounds, options=self.options,
                            evolution=self.evolution)

        if self.disp:
            opt.info

        opt_mat = matrix44(opt.xopt)

        static_mat = matrix44([static_shift[0], static_shift[1],
                               static_shift[2], 0, 0, 0])

        moving_mat = matrix44([-moving_shift[0], -moving_shift[1],
                               -moving_shift[2], 0, 0, 0])

        mat = compose_transformations(moving_mat, opt_mat, static_mat)

        mat_history = []

        if opt.evolution is not None:
            for vecs in opt.evolution:
                mat_history.append(compose_transformations(moving_mat,
                                                           matrix44(vecs),
                                                           static_mat))

        return StreamlineRegistrationMap(mat, opt.xopt, opt.fopt,
                                         mat_history, opt.nfev, opt.nit)


class StreamlineRegistrationMap(object):

    def __init__(self, matopt, xopt, fopt, matopt_history, funcs, iterations):
        r""" A map holding the optimum affine matrix and some other parameters
        of the optimization

        Parameters
        ----------

        matrix : array,
            4x4 affine matrix which transforms the moving to the static
            streamlines

        xopt : array,
            1d array with the parameters of the transformation after centering

        fopt : float,
            final value of the metric

        matrix_history : array
            All transformation matrices created during the optimization

        funcs : int,
            Number of function evaluations of the optimizer

        iterations : int
            Number of iterations of the optimizer

        Methods
        -------
        transform()

        """

        self.matrix = matopt
        self.xopt = xopt
        self.fopt = fopt
        self.matrix_history = matopt_history
        self.funcs = funcs
        self.iterations = iterations


    def transform(self, streamlines):
        """ Apply ``self.matrix`` to the streamlines
        """

        return transform_streamlines(streamlines, self.matrix)


def bundle_sum_distance(t, static, moving):
    """ MDF distance optimization function (SUM)

    We minimize the distance between moving streamlines as they align
    with the static streamlines.

    Parameters
    -----------
    t : ndarray
        t is a vector of of affine transformation parameters with
        size at least 6. 
        If size is 6, t is interpreted as translation + rotation.
        If size is 7, t is interpreted as translation + rotation +
        isotropic scaling. 
        If size is 12, t is interpreted as translation + rotation +
        scaling + shearing.

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
    d01 = bundles_distance_matrix_mdf(static, moving)
    return np.sum(d01)


def bundle_min_distance(t, static, moving):
    """ MDF-based pairwise distance optimization function (MIN)

    We minimize the distance between moving streamlines as they align
    with the static streamlines.

    Parameters
    -----------
    t : ndarray
        t is a vector of of affine transformation parameters with
        size at least 6. 
        If size is 6, t is interpreted as translation + rotation.
        If size is 7, t is interpreted as translation + rotation +
        isotropic scaling. 
        If size is 12, t is interpreted as translation + rotation +
        scaling + shearing.

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
    d01 = bundles_distance_matrix_mdf(static, moving)

    rows, cols = d01.shape
    return 0.25 * (np.sum(np.min(d01, axis=0)) / float(cols) +
                   np.sum(np.min(d01, axis=1)) / float(rows)) ** 2


def bundle_min_distance_fast(t, static, moving, block_size):
    """ Faster implementation of the ``bundle_min_distance``
    """

    aff = matrix44(t)
    moving = np.dot(aff[:3, :3], moving.T).T + aff[:3, 3]
    moving = np.ascontiguousarray(moving, dtype=np.float64)

    rows = static.shape[0] / block_size
    cols = moving.shape[0] / block_size

    return _bundle_minimum_distance_rigid_nomat(static, moving,
                                                rows,
                                                cols,
                                                block_size)


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

    R = I + sin(theta)/theta Sr + (1-cos(theta))/theta2 Sr^2

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


def rotation_mat2vec(R):
    """ Rotation vector from rotation matrix `R`

    Parameters
    ----------
    R : (3,3) array-like
        Rotation matrix

    Returns
    -------
    vec : (3,) array
        Rotation vector, where norm of `vec` is the angle ``theta``, and the
        axis of rotation is given by ``vec / theta``
    """
    ax, angle = quat2angle_axis(mat2quat(R))
    return ax * angle


def matrix44(t, dtype=np.double):
    """ Compose a 4x4 transformation matrix

    Parameters
    -----------
    t : ndarray
        t is a vector of of affine transformation parameters with
        size at least 6.
        If size is 6, t is interpreted as translation + rotation.
        If size is 7, t is interpreted as translation + rotation +
        isotropic scaling.        
        If size is 12, t is interpreted as translation + rotation +
        scaling + shearing.

    Returns
    -------
    T : ndarray

    """
    if isinstance(t, list):
        t = np.array(t)
    size = t.size

    if size not in [6, 7, 12]:
        raise ValueError('Accepted number of parameters is 6, 7 and 12')

    T = np.eye(4, dtype=dtype)

    # Degrees to radians    
    rads = np.deg2rad(t[3:6])

    T[0:3, 3] = threshold(t[0:3], MAX_DIST)
    R = rotation_vec2mat(rads)
    if size == 6:
        T[0:3, 0:3] = R
    elif size == 7:
        T[0:3, 0:3] = t[6] * R
    elif size == 12:
        S = np.diag(threshold(t[6:9], MAX_DIST))
        # Q = rotation_vec2mat(t[9:12])
        kx, ky, kz = t[9:12]
        #shear matrix
        Q = np.array([[1, kx * kz, kx],
                      [ky, 1, 0],
                      [0, kz, 1]])
        # Beware: R*s*Q
        T[0:3, 0:3] = np.dot(R, np.dot(S, Q))
    
    return T


def from_matrix44_rigid(mat):
    """ Given a 4x4 rigid matrix return vector with 3 translations and 3
    rotation angles in degrees.
    """

    vec = np.zeros(6)
    vec[:3] = mat[:3, 3]

    R = mat[:3, :3]
    if det(R) < 0:
        R = -R
    vec[3:6] = rotation_mat2vec(R)
    vec[3:6] = np.rad2deg(vec[3:6])

    return vec


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

    streamlines.append(points[0: offsets[0]])

    for i in range(len(offsets) - 1):
        streamlines.append(points[offsets[i]: offsets[i + 1]])

    return streamlines
