import abc
import numpy as np
from dipy.utils.six import with_metaclass
from dipy.core.optimize import Optimizer
from dipy.align.bundlemin import (_bundle_minimum_distance,
                                  distance_matrix_mdf)
from dipy.tracking.streamline import (transform_streamlines,
                                      unlist_streamlines,
                                      center_streamlines)
from dipy.core.geometry import (compose_transformations,
                                compose_matrix,
                                decompose_matrix)
from dipy.utils.six import string_types

MAX_DIST = 1e10
LOG_MAX_DIST = np.log(MAX_DIST)


class StreamlineDistanceMetric(with_metaclass(abc.ABCMeta, object)):

    def __init__(self, num_threads=None):
        """ An abstract class for the metric used for streamline registration

        If the two sets of streamlines match exactly then method ``distance``
        of this object should be minimum.

        Parameters
        ----------
        num_threads : int
            Number of threads. If None (default) then all available threads
            will be used. Only metrics using OpenMP will use this variable.
        """
        self.static = None
        self.moving = None
        self.num_threads = num_threads

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
        num_threads : int
            Number of threads. If None (default) then all available threads
            will be used.

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
                                        self.block_size,
                                        self.num_threads)


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

        Num_threads is not used in this class. Use ``BundleMinDistanceMetric``
        for a faster, threaded and less memory hungry metric
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

    def __init__(self, metric=None, x0="rigid", method='L-BFGS-B',
                 bounds=None, verbose=False, options=None,
                 evolution=False, num_threads=None):
        r""" Linear registration of 2 sets of streamlines [Garyfallidis14]_.

        Parameters
        ----------
        metric : StreamlineDistanceMetric,
            If None and fast is False then the BMD distance is used. If fast
            is True then a faster implementation of BMD is used. Otherwise,
            use the given distance metric.

        x0 : array or int or str
            Initial parametrization for the optimization.

            If 1D array with:
                a) 6 elements then only rigid registration is parformed with
                the 3 first elements for translation and 3 for rotation.
                b) 7 elements also isotropic scaling is performed (similarity).
                c) 12 elements then translation, rotation (in degrees),
                scaling and shearing is performed (affine).

                Here is an example of x0 with 12 elements:
                ``x0=np.array([0, 10, 0, 40, 0, 0, 2., 1.5, 1, 0.1, -0.5, 0])``

                This has translation (0, 10, 0), rotation (40, 0, 0) in
                degrees, scaling (2., 1.5, 1) and shearing (0.1, -0.5, 0).

            If int:
                a) 6
                    ``x0 = np.array([0, 0, 0, 0, 0, 0])``
                b) 7
                    ``x0 = np.array([0, 0, 0, 0, 0, 0, 1.])``
                c) 12
                    ``x0 = np.array([0, 0, 0, 0, 0, 0, 1., 1., 1, 0, 0, 0])``

            If str:
                a) "rigid"
                    ``x0 = np.array([0, 0, 0, 0, 0, 0])``
                b) "similarity"
                    ``x0 = np.array([0, 0, 0, 0, 0, 0, 1.])``
                c) "affine"
                    ``x0 = np.array([0, 0, 0, 0, 0, 0, 1., 1., 1, 0, 0, 0])``

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

        num_threads : int
            Number of threads. If None (default) then all available threads
            will be used. Only metrics using OpenMP will use this variable.

        References
        ----------
        .. [Garyfallidis14] Garyfallidis et al., "Direct native-space fiber
                            bundle alignment for group comparisons", ISMRM,
                            2014.

        """

        self.x0 = self._set_x0(x0)
        self.metric = metric

        if self.metric is None:
            self.metric = BundleMinDistanceMetric(num_threads=num_threads)

        self.verbose = verbose
        self.method = method
        if self.method not in ['Powell', 'L-BFGS-B']:
            raise ValueError('Only Powell and L-BFGS-B can be used')
        self.bounds = bounds
        self.options = options
        self.evolution = evolution

    def optimize(self, static, moving, mat=None):
        """ Find the minimum of the provided metric.

        Parameters
        ----------
        static : streamlines
            Reference or fixed set of streamlines.
        moving : streamlines
            Moving set of streamlines.
        mat : array
            Transformation (4, 4) matrix to start the registration. ``mat``
            is applied to moving. Default value None which means that initial
            transformation will be generated by shifting the centers of moving
            and static sets of streamlines to the origin.

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

        if mat is None:
            static_centered, static_shift = center_streamlines(static)
            moving_centered, moving_shift = center_streamlines(moving)
            static_mat = compose_matrix44([static_shift[0], static_shift[1],
                                           static_shift[2], 0, 0, 0])

            moving_mat = compose_matrix44([-moving_shift[0], -moving_shift[1],
                                           -moving_shift[2], 0, 0, 0])
        else:
            static_centered = static
            moving_centered = transform_streamlines(moving, mat)
            static_mat = np.eye(4)
            moving_mat = mat

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
                                'gtol': 1e-5, 'eps': 1e-8,
                                'maxiter': 100}

            opt = Optimizer(distance, self.x0.tolist(),
                            method=self.method,
                            bounds=self.bounds, options=self.options,
                            evolution=self.evolution)

        if self.verbose:
            opt.print_summary()

        opt_mat = compose_matrix44(opt.xopt)

        mat = compose_transformations(moving_mat, opt_mat, static_mat)

        mat_history = []

        if opt.evolution is not None:
            for vecs in opt.evolution:
                mat_history.append(
                    compose_transformations(moving_mat,
                                            compose_matrix44(vecs),
                                            static_mat))

        srm = StreamlineRegistrationMap(mat, opt.xopt, opt.fopt,
                                        mat_history, opt.nfev, opt.nit)
        del opt
        return srm

    def _set_x0(self, x0):
        """ check if input is of correct type"""

        if hasattr(x0, 'ndim'):

            if len(x0) not in [6, 7, 12]:
                msg = 'Only 1D arrays of 6, 7 and 12 elements are allowed'
                raise ValueError(msg)
            if x0.ndim != 1:
                raise ValueError("Array should have only one dimension")
            return x0

        if isinstance(x0, string_types):
            if x0.lower() == 'rigid':
                return np.zeros(6)

            if x0.lower() == 'similarity':
                return np.array([0, 0, 0, 0, 0, 0, 1.])

            if x0.lower() == 'affine':
                return np.array([0, 0, 0, 0, 0, 0, 1., 1., 1., 0, 0, 0])

        if isinstance(x0, int):
            if x0 not in [6, 7, 12]:
                msg = 'Only 6, 7 and 12 are accepted as integers'
                raise ValueError(msg)
            else:
                if x0 == 6:
                    return np.zeros(6)
                if x0 == 7:
                    return np.array([0, 0, 0, 0, 0, 0, 1.])
                if x0 == 12:
                    return np.array([0, 0, 0, 0, 0, 0, 1., 1., 1., 0, 0, 0])

        raise ValueError('Wrong input')


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


def bundle_sum_distance(t, static, moving, num_threads=None):
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

    aff = compose_matrix44(t)
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

    num_threads : int
        Number of threads. If None (default) then all available threads
        will be used.

    Returns
    -------
    cost: float

    """
    aff = compose_matrix44(t)
    moving = transform_streamlines(moving, aff)
    d01 = distance_matrix_mdf(static, moving)

    rows, cols = d01.shape
    return 0.25 * (np.sum(np.min(d01, axis=0)) / float(cols) +
                   np.sum(np.min(d01, axis=1)) / float(rows)) ** 2


def bundle_min_distance_fast(t, static, moving, block_size, num_threads):
    """ MDF-based pairwise distance optimization function (MIN)

    We minimize the distance between moving streamlines as they align
    with the static streamlines.

    Parameters
    -----------
    t : array
        1D array. t is a vector of of affine transformation parameters with
        size at least 6.
        If size is 6, t is interpreted as translation + rotation.
        If size is 7, t is interpreted as translation + rotation +
        isotropic scaling.
        If size is 12, t is interpreted as translation + rotation +
        scaling + shearing.

    static : array
        N*M x 3 array. All the points of the static streamlines. With order of
        streamlines intact. Where N is the number of streamlines and M
        is the number of points per streamline.

    moving : array
        K*M x 3 array. All the points of the moving streamlines. With order of
        streamlines intact. Where K is the number of streamlines and M
        is the number of points per streamline.

    block_size : int
        Number of points per streamline. All streamlines in static and moving
        should have the same number of points M.

    num_threads : int
        Number of threads. If None (default) then all available threads
        will be used.

    Returns
    -------
    cost: float

    Notes
    -----
    This is a faster implementation of ``bundle_min_distance``, which requires
    that all the points of each streamline are allocated into an ndarray
    (of shape N*M by 3, with N the number of points per streamline and M the
    number of streamlines). This can be done by calling
    `dipy.tracking.streamlines.unlist_streamlines`.

    """

    aff = compose_matrix44(t)
    moving = np.dot(aff[:3, :3], moving.T).T + aff[:3, 3]
    moving = np.ascontiguousarray(moving, dtype=np.float64)

    rows = static.shape[0] / block_size
    cols = moving.shape[0] / block_size

    return _bundle_minimum_distance(static, moving,
                                    rows,
                                    cols,
                                    block_size,
                                    num_threads)


def _threshold(x, th):
    return np.maximum(np.minimum(x, th), -th)


def compose_matrix44(t, dtype=np.double):
    """ Compose a 4x4 transformation matrix

    Parameters
    -----------
    t : ndarray
        This is a 1D vector of of affine transformation parameters with
        size at least 6.
        If size is 6, t is interpreted as translation + rotation.
        If size is 7, t is interpreted as translation + rotation +
        isotropic scaling.
        If size is 12, t is interpreted as translation + rotation +
        scaling + shearing.

    Returns
    -------
    T : ndarray
        Homogeneous transformation matrix of size 4x4.

    """
    if isinstance(t, list):
        t = np.array(t)
    size = t.size

    if size not in [6, 7, 12]:
        raise ValueError('Accepted number of parameters is 6, 7 and 12')

    scale, shear, angles, translate = (None, ) * 4
    if size in [6, 7, 12]:
        translate = _threshold(t[0:3], MAX_DIST)
        angles = np.deg2rad(t[3:6])
    if size == 7:
        scale = np.array((t[6],) * 3)
    if size == 12:
        scale = t[6: 9]
        shear = t[9: 12]
    return compose_matrix(scale=scale, shear=shear,
                          angles=angles,
                          translate=translate)


def decompose_matrix44(mat, size=12):
    """ Given a 4x4 homogeneous matrix return the parameter vector

    Parameters
    -----------
    mat : array
        Homogeneous 4x4 transformation matrix
    size : int
        Size of output vector. 6 for rigid, 7 for similarity and 12
        for affine. Default is 12.

    Returns
    -------
    t : ndarray
        One dimensional ndarray of 6, 7 or 12 affine parameters.

    """
    scale, shear, angles, translate, _ = decompose_matrix(mat)

    t = np.zeros(12)
    t[:3] = translate
    t[3: 6] = np.rad2deg(angles)
    if size == 6:
        return t[:6]
    if size == 7:
        t[6] = np.mean(scale)
        return t[:7]
    if size == 12:
        t[6: 9] = scale
        t[9: 12] = shear
        return t

    raise ValueError('Size can be 6, 7 or 12')
