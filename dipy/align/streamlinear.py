import abc
import numpy as np
from nibabel.quaternions import quat2angle_axis, mat2quat
from scipy.linalg import det
from dipy.utils.six import with_metaclass
from dipy.core.optimize import Optimizer
from dipy.align.bundlemin import (_bundle_minimum_distance,
                                  distance_matrix_mdf)
from dipy.tracking.streamline import (transform_streamlines,
                                      unlist_streamlines,
                                      center_streamlines)
from dipy.core.geometry import (rodrigues_axis_rotation,
                                compose_transformations,
                                compose_matrix)

MAX_DIST = 1e10
LOG_MAX_DIST = np.log(MAX_DIST)


class StreamlineDistanceMetric(with_metaclass(abc.ABCMeta, object)):

    def __init__(self):
        """ An abstract class for the metric used for streamline registration

        If the two sets of streamlines match exactly then method ``distance``
        of this object should be minimum.
        """
        self.static = None
        self.moving = None

    @abc.abstractmethod
    def setup(self, static, moving):
        pass

    @abc.abstractmethod
    def distance(self, xopt):
        """ calculate distance for current set of parameters
        """
        pass


class BundleMinDistanceMetric(StreamlineDistanceMetric):
    """ Bundle-based Minimum Distance aka BMD

    This is the cost function used by the StreamlineLinearRegistration

    Methods
    -------
    setup(static, moving)
    distance(xopt)

    References
    ----------
    .. [Garyfallidis14] Garyfallidis et al., "Direct native-space fiber
                        bundle alignment for group comparisons", ISMRM,
                        2014.
    """

    def setup(self, static, moving):
        """ Setup static and moving sets of streamlines

        Parameters
        ----------
        static : streamlines
            Fixed or reference set of streamlines.
        moving : streamlines
            Moving streamlines.

        Notes
        -----
        Call this after the object is initiated and before distance.
        """

        self._set_static(static)
        self._set_moving(moving)

    def _set_static(self, static):
        static_centered_pts, st_idx = unlist_streamlines(static)
        self.static_centered_pts = np.ascontiguousarray(static_centered_pts,
                                                        dtype=np.float64)
        self.block_size = st_idx[0]

    def _set_moving(self, moving):
        self.moving_centered_pts, _ = unlist_streamlines(moving)

    def distance(self, xopt):
        """ Distance calculated from this Metric

        Parameters
        ----------
        xopt : sequence
            List of affine parameters as an 1D vector,

        """
        return bundle_min_distance_fast(xopt,
                                        self.static_centered_pts,
                                        self.moving_centered_pts,
                                        self.block_size)


class BundleMinDistanceMatrixMetric(StreamlineDistanceMetric):
    """ Bundle-based Minimum Distance aka BMD

    This is the cost function used by the StreamlineLinearRegistration

    Methods
    -------
    setup(static, moving)
    distance(xopt)

    Notes
    -----
    The difference with BundleMinDistanceMetric is that this creates
    the entire distance matrix and therefore requires more memory.

    """

    def setup(self, static, moving):
        """ Setup static and moving sets of streamlines

        Parameters
        ----------
        static : streamlines
            Fixed or reference set of streamlines.
        moving : streamlines
            Moving streamlines.

        Notes
        -----
        Call this after the object is initiated and before distance.

        The difference between this class and
        """
        self.static = static
        self.moving = moving

    def distance(self, xopt):
        """ Distance calculated from this Metric

        Parameters
        ----------
        xopt : sequence
            List of affine parameters as an 1D vector
        """
        return bundle_min_distance(xopt, self.static, self.moving)


class BundleSumDistanceMatrixMetric(BundleMinDistanceMatrixMetric):
    """ Bundle-based Sum Distance aka BMD

    This is a cost function that can be used by the
    StreamlineLinearRegistration class.

    Methods
    -------
    setup(static, moving)
    distance(xopt)

    Notes
    -----
    The difference with BundleMinDistanceMatrixMetric is that it uses
    uses the sum of the distance matrix and not the sum of mins.
    """

    def distance(self, xopt):
        """ Distance calculated from this Metric

        Parameters
        ----------
        xopt : sequence
            List of affine parameters as an 1D vector
        """
        return bundle_sum_distance(xopt, self.static, self.moving)


class StreamlineLinearRegistration(object):

    def __init__(self, metric=None, x0=None, method='L-BFGS-B',
                 bounds=None, verbose=False, options=None,
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
            If x0 has 6 elements then only translation and rotation is
            performed (rigid). If x0 has 7 elements also isotropic scaling is
            performed (similarity). If x0 has 12 elements then translation,
            rotation, scaling and shearing is performed (affine).

        method : str,
            'L_BFGS_B' or 'Powell' optimizers can be used. Default is
            'L_BFGS_B'.

        bounds : list of tuples or None,
            If method == 'L_BFGS_B' then we can use bounded optimization.
            For example for the six parameters of rigid rotation we can set
            the bounds = [(-30, 30), (-30, 30), (-30, 30),
                          (-45, 45), (-45, 45), (-45, 45)]
            That means that we have set the bounds for the three translations
            and three rotation axes (in degrees).

        verbose : bool,
            If True then information about the optimization is shown.

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
            self.metric = BundleMinDistanceMetric()

        self.verbose = verbose
        self.method = method
        if self.method not in ['Powell', 'L-BFGS-B']:
            raise ValueError('Only Powell and L-BFGS-B can be used')
        self.bounds = bounds
        self.options = options
        self.evolution = evolution

    def optimize(self, static, moving):
        """ Find the minimum of the provided metric.

        Parameters
        ----------

        static : streamlines
            Reference or fixed set of streamlines.
        moving : streamlines
            Moving set of streamlines.

        Returns
        -------
        map : StreamlineRegistrationMap

        """

        msg = 'need to have the same number of points. Use '
        msg += 'set_number_of_points from dipy.tracking.streamline'

        if not np.all(np.array(list(map(len, static))) == static[0].shape[0]):
            raise ValueError('Static streamlines ' + msg)

        if not np.all(np.array(list(map(len, moving))) == moving[0].shape[0]):
            raise ValueError('Moving streamlines ' + msg)

        if not np.all(np.array(list(map(len, moving))) == static[0].shape[0]):
            raise ValueError('Static and moving streamlines ' + msg)

        static_centered, static_shift = center_streamlines(static)
        moving_centered, moving_shift = center_streamlines(moving)

        self.metric.setup(static_centered, moving_centered)

        distance = self.metric.distance

        if self.method == 'Powell':

            if self.options is None:
                self.options = {'xtol': 1e-6, 'ftol': 1e-6, 'maxiter': 1e6}

            opt = Optimizer(distance, self.x0.tolist(),
                            method=self.method, options=self.options,
                            evolution=self.evolution)

        if self.method == 'L-BFGS-B':

            if self.options is None:
                self.options = {'maxcor': 10, 'ftol': 1e-7,
                                'gtol': 1e-5, 'eps': 1e-8}

            opt = Optimizer(distance, self.x0.tolist(),
                            method=self.method,
                            bounds=self.bounds, options=self.options,
                            evolution=self.evolution)

        if self.verbose:
            opt.print_summary()

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

        srm = StreamlineRegistrationMap(mat, opt.xopt, opt.fopt,
                                        mat_history, opt.nfev, opt.nit)
        del opt
        return srm


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

    def transform(self, moving):
        """ Transform moving streamlines to the static.

        Parameters
        ----------
        moving : streamlines

        Returns
        -------
        moved : streamlines

        Notes
        -----

        All this does is apply ``self.matrix`` to the input streamlines.
        """

        return transform_streamlines(moving, self.matrix)


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
    d01 = distance_matrix_mdf(static, moving)
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
        Moving streamlines.

    Returns
    -------
    cost: float

    """
    aff = matrix44(t)
    moving = transform_streamlines(moving, aff)
    d01 = distance_matrix_mdf(static, moving)

    rows, cols = d01.shape
    return 0.25 * (np.sum(np.min(d01, axis=0)) / float(cols) +
                   np.sum(np.min(d01, axis=1)) / float(rows)) ** 2


def bundle_min_distance_fast(t, static, moving, block_size):
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

    static : ndarray
        All the points of the static streamlines. With order of streamlines
        intact.

    moving : ndarray
        All the points of the moving streamlines. With order of streamlines
        intact.

    block_size : int
        Number of points per streamline. All streamlines in statict and moving
        should have the same number of points.

    Returns
    -------
    cost: float

    Notes
    -----
    Faster implementation of the ``bundle_min_distance``. This is to
    be used after you have called ``unlist_streamlines`` which returns all the
    points of all the streamlines as one ndarray.

    """

    aff = matrix44(t)
    moving = np.dot(aff[:3, :3], moving.T).T + aff[:3, 3]
    moving = np.ascontiguousarray(moving, dtype=np.float64)

    rows = static.shape[0] / block_size
    cols = moving.shape[0] / block_size

    return _bundle_minimum_distance(static, moving,
                                    rows,
                                    cols,
                                    block_size)


def rotation_vec2mat(r):
    r"""  The rotation matrix is given by the Rodrigues formula:

    Parameters
    ----------
    r : (3,) array
        Rotation vector

    Returns
    -------
    R : (3, 3) array
        Rotation matrix

    """
    theta = np.linalg.norm(r)

    return rodrigues_axis_rotation(r, np.rad2deg(theta))


def rotation_mat2vec(R):
    """ Rotation vector from rotation matrix `R`

    Parameters
    ----------
    R : (3, 3) array-like
        Rotation matrix

    Returns
    -------
    vec : (3,) array
        Rotation vector, where norm of `vec` is the angle ``theta``, and the
        axis of rotation is given by ``vec / theta``
    """
    ax, angle = quat2angle_axis(mat2quat(R))
    return ax * angle


def _threshold(x, th):
    return np.maximum(np.minimum(x, th), -th)


def matrix44(t, dtype=np.double, cm=False):
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

    if cm:

        scale, shear, angles, translate = (None, ) * 4
        if size == [6, 7, 12]:
            translate = t[:3]
            angles = t[3: 6]
        if size == 7:
            scale = np.array((t[6],) * 3)
        if size == 12:
            scale = t[6: 9]
            shear = t[9: 12]

        return compose_matrix(scale=scale, shear=shear,
                              angles=angles,
                              translate=translate)
    else:

        T[0:3, 3] = _threshold(t[0:3], MAX_DIST)
        R = rotation_vec2mat(rads)
        if size == 6:
            T[0:3, 0:3] = R
        elif size == 7:
            T[0:3, 0:3] = t[6] * R
        elif size == 12:
            S = np.diag(_threshold(t[6:9], MAX_DIST))
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
