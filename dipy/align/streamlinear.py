import logging
import abc
import numpy as np
from dipy.core.optimize import Optimizer
from dipy.align.bundlemin import (_bundle_minimum_distance,
                                  _bundle_minimum_distance_asymmetric,
                                  distance_matrix_mdf)
from dipy.tracking.streamline import (transform_streamlines,
                                      unlist_streamlines,
                                      center_streamlines,
                                      set_number_of_points,
                                      select_random_set_of_streamlines,
                                      length,
                                      Streamlines)
from dipy.segment.clustering import qbx_and_merge
from dipy.core.geometry import (compose_transformations,
                                compose_matrix,
                                decompose_matrix)
from time import time

DEFAULT_BOUNDS = [(-35, 35), (-35, 35), (-35, 35),
                  (-45, 45), (-45, 45), (-45, 45),
                  (0.6, 1.4), (0.6, 1.4), (0.6, 1.4),
                  (-10, 10), (-10, 10), (-10, 10)]

logger = logging.getLogger(__name__)


class StreamlineDistanceMetric(object, metaclass=abc.ABCMeta):

    def __init__(self, num_threads=None):
        """ An abstract class for the metric used for streamline registration.

        If the two sets of streamlines match exactly then method ``distance``
        of this object should be minimum.

        Parameters
        ----------
        num_threads : int, optional
            Number of threads to be used for OpenMP parallelization. If None
            (default) the value of OMP_NUM_THREADS environment variable is used
            if it is set, otherwise all available threads are used. If < 0 the
            maximal number of threads minus |num_threads + 1| is used (enter -1
            to use as many threads as possible). 0 raises an error. Only
            metrics using OpenMP will use this variable.

        """
        self.static = None
        self.moving = None
        self.num_threads = num_threads

    @abc.abstractmethod
    def setup(self, static, moving):
        pass

    @abc.abstractmethod
    def distance(self, xopt):
        """ calculate distance for current set of parameters.
        """
        pass


class BundleMinDistanceMetric(StreamlineDistanceMetric):
    """ Bundle-based Minimum Distance aka BMD

    This is the cost function used by the StreamlineLinearRegistration.

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
        """ Setup static and moving sets of streamlines.

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
        """ Distance calculated from this Metric.

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
        """ Setup static and moving sets of streamlines.

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
        """ Distance calculated from this Metric.

        Parameters
        ----------
        xopt : sequence
            List of affine parameters as an 1D vector
        """
        return bundle_min_distance(xopt, self.static, self.moving)


class BundleMinDistanceAsymmetricMetric(BundleMinDistanceMetric):
    """ Asymmetric Bundle-based Minimum distance.

    This is a cost function that can be used by the
    StreamlineLinearRegistration class.

    """

    def distance(self, xopt):
        """ Distance calculated from this Metric.

        Parameters
        ----------
        xopt : sequence
            List of affine parameters as an 1D vector

        """
        return bundle_min_distance_asymmetric_fast(xopt,
                                                   self.static_centered_pts,
                                                   self.moving_centered_pts,
                                                   self.block_size)


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
                 bounds=None, verbose=False, options=None, evolution=False,
                 num_threads=None):
        r""" Linear registration of 2 sets of streamlines [Garyfallidis15]_.

        Parameters
        ----------
        metric : StreamlineDistanceMetric,
            If None and fast is False then the BMD distance is used. If fast
            is True then a faster implementation of BMD is used. Otherwise,
            use the given distance metric.

        x0 : array or int or str
            Initial parametrization for the optimization.

            If 1D array with:
                a) 6 elements then only rigid registration is performed with
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

        verbose : bool, optional.
            If True, if True then information about the optimization is shown.
            Default: False.

        options : None or dict,
            Extra options to be used with the selected method.

        evolution : boolean
            If True save the transformation for each iteration of the
            optimizer. Default is False. Supported only with Scipy >= 0.11.

        num_threads : int, optional
            Number of threads to be used for OpenMP parallelization. If None
            (default) the value of OMP_NUM_THREADS environment variable is used
            if it is set, otherwise all available threads are used. If < 0 the
            maximal number of threads minus |num_threads + 1| is used (enter -1
            to use as many threads as possible). 0 raises an error. Only
            metrics using OpenMP will use this variable.

        References
        ----------
        .. [Garyfallidis15] Garyfallidis et al. "Robust and efficient linear
           registration of white-matter fascicles in the space of streamlines",
           NeuroImage, 117, 124--140, 2015

        .. [Garyfallidis14] Garyfallidis et al., "Direct native-space fiber
           bundle alignment for group comparisons", ISMRM, 2014.

        .. [Garyfallidis17] Garyfallidis et al. Recognition of white matter
           bundles using local and global streamline-based
           registration and clustering, Neuroimage, 2017.

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
        """ check if input is of correct type."""

        if hasattr(x0, 'ndim'):

            if len(x0) not in [3, 6, 7, 9, 12]:
                m_ = 'Only 1D arrays of 3, 6, 7, 9 and 12 elements are allowed'
                raise ValueError(m_)
            if x0.ndim != 1:
                raise ValueError("Array should have only one dimension")
            return x0

        if isinstance(x0, str):

            if x0.lower() == 'translation':
                return np.zeros(3)

            if x0.lower() == 'rigid':
                return np.zeros(6)

            if x0.lower() == 'similarity':
                return np.array([0, 0, 0, 0, 0, 0, 1.])

            if x0.lower() == 'scaling':
                return np.array([0, 0, 0, 0, 0, 0, 1., 1., 1.])

            if x0.lower() == 'affine':
                return np.array([0, 0, 0, 0, 0, 0, 1., 1., 1., 0, 0, 0])

        if isinstance(x0, int):
            if x0 not in [3, 6, 7, 9, 12]:
                msg = 'Only 3, 6, 7, 9 and 12 are accepted as integers'
                raise ValueError(msg)
            else:
                if x0 == 3:
                    return np.zeros(3)
                if x0 == 6:
                    return np.zeros(6)
                if x0 == 7:
                    return np.array([0, 0, 0, 0, 0, 0, 1.])
                if x0 == 9:
                    return np.array([0, 0, 0, 0, 0, 0, 1., 1., 1.])
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
    """ MDF distance optimization function (SUM).

    We minimize the distance between moving streamlines as they align
    with the static streamlines.

    Parameters
    ----------
    t : ndarray
        t is a vector of affine transformation parameters with
        size at least 6.
        If the size is 6, t is interpreted as translation + rotation.
        If the size is 7, t is interpreted as translation + rotation +
        isotropic scaling.
        If size is 12, t is interpreted as translation + rotation +
        scaling + shearing.

    static : list
        Static streamlines

    moving : list
        Moving streamlines. These will be transformed to align with
        the static streamlines

    num_threads : int, optional
        Number of threads. If -1 then all available threads will be used.

    Returns
    -------
    cost: float

    """

    aff = compose_matrix44(t)
    moving = transform_streamlines(moving, aff)
    d01 = distance_matrix_mdf(static, moving)
    return np.sum(d01)


def bundle_min_distance(t, static, moving):
    """ MDF-based pairwise distance optimization function (MIN).

    We minimize the distance between moving streamlines as they align
    with the static streamlines.

    Parameters
    ----------
    t : ndarray
        t is a vector of affine transformation parameters with
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
    aff = compose_matrix44(t)
    moving = transform_streamlines(moving, aff)
    d01 = distance_matrix_mdf(static, moving)

    rows, cols = d01.shape
    return 0.25 * (np.sum(np.min(d01, axis=0)) / float(cols) +
                   np.sum(np.min(d01, axis=1)) / float(rows)) ** 2


def bundle_min_distance_fast(t, static, moving, block_size, num_threads=None):
    """ MDF-based pairwise distance optimization function (MIN).

    We minimize the distance between moving streamlines as they align
    with the static streamlines.

    Parameters
    ----------
    t : array
        1D array. t is a vector of affine transformation parameters with
        size at least 6.
        If the size is 6, t is interpreted as translation + rotation.
        If the size is 7, t is interpreted as translation + rotation +
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

    num_threads : int, optional
        Number of threads to be used for OpenMP parallelization. If None
        (default) the value of OMP_NUM_THREADS environment variable is used
        if it is set, otherwise all available threads are used. If < 0 the
        maximal number of threads minus |num_threads + 1| is used (enter -1 to
        use as many threads as possible). 0 raises an error.

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

    rows = static.shape[0] // block_size
    cols = moving.shape[0] // block_size

    return _bundle_minimum_distance(static, moving,
                                    rows,
                                    cols,
                                    block_size,
                                    num_threads)


def bundle_min_distance_asymmetric_fast(t, static, moving, block_size):
    """ MDF-based pairwise distance optimization function (MIN).

    We minimize the distance between moving streamlines as they align
    with the static streamlines.

    Parameters
    ----------
    t : array
        1D array. t is a vector of affine transformation parameters with
        size at least 6.
        If the size is 6, t is interpreted as translation + rotation.
        If the size is 7, t is interpreted as translation + rotation +
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

    Returns
    -------
    cost: float

    """
    aff = compose_matrix44(t)
    moving = np.dot(aff[:3, :3], moving.T).T + aff[:3, 3]
    moving = np.ascontiguousarray(moving, dtype=np.float64)

    rows = static.shape[0] // block_size
    cols = moving.shape[0] // block_size

    return _bundle_minimum_distance_asymmetric(static, moving,
                                               rows,
                                               cols,
                                               block_size)


def remove_clusters_by_size(clusters, min_size=0):
    ob = filter(lambda c: len(c) >= min_size, clusters)

    centroids = Streamlines()
    for cluster in ob:
        centroids.append(cluster.centroid)

    return centroids


def progressive_slr(static, moving, metric, x0, bounds, method='L-BFGS-B',
                    verbose=False, num_threads=None):
    """ Progressive SLR.

    This is an utility function that allows for example to do affine
    registration using Streamline-based Linear Registration (SLR)
    [Garyfallidis15]_ by starting with translation first, then rigid,
    then similarity, scaling and finally affine.

    Similarly, if for example, you want to perform rigid then you start with
    translation first. This progressive strategy can helps with finding the
    optimal parameters of the final transformation.

    Parameters
    ----------
    static : Streamlines
    moving : Streamlines
    metric : StreamlineDistanceMetric
    x0 : string
        Could be any of 'translation', 'rigid', 'similarity', 'scaling',
        'affine'
    bounds : array
        Boundaries of registration parameters. See variable `DEFAULT_BOUNDS`
        for example.
    method : string
        L_BFGS_B' or 'Powell' optimizers can be used. Default is 'L_BFGS_B'.
    verbose :  bool, optional.
        If True, log messages. Default:
    num_threads : int, optional
        Number of threads to be used for OpenMP parallelization. If None
        (default) the value of OMP_NUM_THREADS environment variable is used
        if it is set, otherwise all available threads are used. If < 0 the
        maximal number of threads minus |num_threads + 1| is used (enter -1 to
        use as many threads as possible). 0 raises an error. Only metrics
        using OpenMP will use this variable.

    References
    ----------
    .. [Garyfallidis15] Garyfallidis et al. "Robust and efficient linear
        registration of white-matter fascicles in the space of streamlines",
        NeuroImage, 117, 124--140, 2015

    """
    if verbose:
        logger.info('Progressive Registration is Enabled')

    if x0 == 'translation' or x0 == 'rigid' or \
       x0 == 'similarity' or x0 == 'scaling' or x0 == 'affine':
        if verbose:
            logger.info(' Translation  (3 parameters)...')
        slr_t = StreamlineLinearRegistration(metric=metric,
                                             x0='translation',
                                             bounds=bounds[:3],
                                             method=method)

        slm_t = slr_t.optimize(static, moving)

    if x0 == 'rigid' or x0 == 'similarity' or \
       x0 == 'scaling' or x0 == 'affine':

        x_translation = slm_t.xopt
        x = np.zeros(6)
        x[:3] = x_translation
        if verbose:
            logger.info(' Rigid  (6 parameters) ...')
        slr_r = StreamlineLinearRegistration(metric=metric,
                                             x0=x,
                                             bounds=bounds[:6],
                                             method=method)
        slm_r = slr_r.optimize(static, moving)

    if x0 == 'similarity' or x0 == 'scaling' or x0 == 'affine':

        x_rigid = slm_r.xopt
        x = np.zeros(7)
        x[:6] = x_rigid
        x[6] = 1.
        if verbose:
            logger.info(' Similarity (7 parameters) ...')
        slr_s = StreamlineLinearRegistration(metric=metric,
                                             x0=x,
                                             bounds=bounds[:7],
                                             method=method)
        slm_s = slr_s.optimize(static, moving)

    if x0 == 'scaling' or x0 == 'affine':

        x_similarity = slm_s.xopt
        x = np.zeros(9)
        x[:6] = x_similarity[:6]
        x[6:] = np.array((x_similarity[6],) * 3)
        if verbose:
            logger.info(' Scaling (9 parameters) ...')

        slr_c = StreamlineLinearRegistration(metric=metric,
                                             x0=x,
                                             bounds=bounds[:9],
                                             method=method)
        slm_c = slr_c.optimize(static, moving)

    if x0 == 'affine':

        x_scaling = slm_c.xopt
        x = np.zeros(12)
        x[:9] = x_scaling[:9]
        x[9:] = np.zeros(3)
        if verbose:
            logger.info(' Affine (12 parameters) ...')

        slr_a = StreamlineLinearRegistration(metric=metric,
                                             x0=x,
                                             bounds=bounds[:12],
                                             method=method)
        slm_a = slr_a.optimize(static, moving)

    if x0 == 'translation':
        slm = slm_t
    elif x0 == 'rigid':
        slm = slm_r
    elif x0 == 'similarity':
        slm = slm_s
    elif x0 == 'scaling':
        slm = slm_c
    elif x0 == 'affine':
        slm = slm_a
    else:
        raise ValueError('Incorrect SLR transform')

    return slm


def slr_with_qbx(static, moving,
                 x0='affine',
                 rm_small_clusters=50,
                 maxiter=100,
                 select_random=None,
                 verbose=False,
                 greater_than=50,
                 less_than=250,
                 qbx_thr=[40, 30, 20, 15],
                 nb_pts=20,
                 progressive=True, rng=None, num_threads=None):
    """ Utility function for registering large tractograms.

    For efficiency, we apply the registration on cluster centroids and remove
    small clusters.

    Parameters
    ----------
    static : Streamlines
    moving : Streamlines

    x0 : str, optional.
        rigid, similarity or affine transformation model (default affine)

    rm_small_clusters : int, optional
        Remove clusters that have less than `rm_small_clusters` (default 50)

    select_random : int, optional.
        If not, None selects a random number of streamlines to apply clustering
        Default None.

    verbose : bool, optional
        If True, logs information about optimization. Default: False

    greater_than : int, optional
            Keep streamlines that have length greater than
            this value (default 50)

    less_than : int, optional
            Keep streamlines have length less than this value (default 250)

    qbx_thr : variable int
            Thresholds for QuickBundlesX (default [40, 30, 20, 15])

    np_pts : int, optional
            Number of points for discretizing each streamline (default 20)

    progressive : boolean, optional
            (default True)

    rng : RandomState
        If None creates RandomState in function.

    num_threads : int, optional
        Number of threads to be used for OpenMP parallelization. If None
        (default) the value of OMP_NUM_THREADS environment variable is used
        if it is set, otherwise all available threads are used. If < 0 the
        maximal number of threads minus |num_threads + 1| is used (enter -1 to
        use as many threads as possible). 0 raises an error. Only metrics
        using OpenMP will use this variable.

    Notes
    -----
    The order of operations is the following. First short or long streamlines
    are removed. Second, the tractogram or a random selection of the tractogram
    is clustered with QuickBundles. Then SLR [Garyfallidis15]_ is applied.

    References
    ----------
    .. [Garyfallidis15] Garyfallidis et al. "Robust and efficient linear
    registration of white-matter fascicles in the space of streamlines",
    NeuroImage, 117, 124--140, 2015
    .. [Garyfallidis14] Garyfallidis et al., "Direct native-space fiber
            bundle alignment for group comparisons", ISMRM, 2014.
    .. [Garyfallidis17] Garyfallidis et al. Recognition of white matter
    bundles using local and global streamline-based registration and
    clustering, Neuroimage, 2017.

    """
    if rng is None:
        rng = np.random.RandomState()

    if verbose:
        logger.info('Static streamlines size {}'.format(len(static)))
        logger.info('Moving streamlines size {}'.format(len(moving)))

    def check_range(streamline, gt=greater_than, lt=less_than):

        if (length(streamline) > gt) & (length(streamline) < lt):
            return True
        else:
            return False

    streamlines1 = Streamlines(static[np.array([check_range(s)
                                                for s in static])])
    streamlines2 = Streamlines(moving[np.array([check_range(s)
                                                for s in moving])])
    if verbose:
        logger.info('Static streamlines after length reduction {}'
                    .format(len(streamlines1)))
        logger.info('Moving streamlines after length reduction {}'
                    .format(len(streamlines2)))

    if select_random is not None:
        rstreamlines1 = select_random_set_of_streamlines(streamlines1,
                                                         select_random,
                                                         rng=rng)
    else:
        rstreamlines1 = streamlines1

    rstreamlines1 = set_number_of_points(rstreamlines1, nb_pts)

    rstreamlines1._data.astype('f4')

    cluster_map1 = qbx_and_merge(rstreamlines1, thresholds=qbx_thr, rng=rng)
    qb_centroids1 = remove_clusters_by_size(cluster_map1, rm_small_clusters)

    if select_random is not None:
        rstreamlines2 = select_random_set_of_streamlines(streamlines2,
                                                         select_random,
                                                         rng=rng)
    else:
        rstreamlines2 = streamlines2

    rstreamlines2 = set_number_of_points(rstreamlines2, nb_pts)
    rstreamlines2._data.astype('f4')

    cluster_map2 = qbx_and_merge(rstreamlines2, thresholds=qbx_thr, rng=rng)

    qb_centroids2 = remove_clusters_by_size(cluster_map2, rm_small_clusters)

    if verbose:
        t = time()

    if not progressive:
        slr = StreamlineLinearRegistration(x0=x0,
                                           options={'maxiter': maxiter},
                                           num_threads=num_threads)
        slm = slr.optimize(qb_centroids1, qb_centroids2)
    else:
        bounds = DEFAULT_BOUNDS

        slm = progressive_slr(qb_centroids1, qb_centroids2,
                              x0=x0, metric=None,
                              bounds=bounds, num_threads=num_threads)

    if verbose:
        logger.info('QB static centroids size %d' % len(qb_centroids1,))
        logger.info('QB moving centroids size %d' % len(qb_centroids2,))
        duration = time() - t
        logger.info('SLR finished in  %0.3f seconds.' % (duration,))
        if slm.iterations is not None:
            logger.info('SLR iterations: %d ' % (slm.iterations,))

    moved = slm.transform(moving)

    return moved, slm.matrix, qb_centroids1, qb_centroids2


# In essence whole_brain_slr can be thought as a combination of
# SLR on QuickBundles centroids and some thresholding see
# Garyfallidis et al. Recognition of white matter
# bundles using local and global streamline-based registration and
# clustering, NeuroImage, 2017.
whole_brain_slr = slr_with_qbx


def _threshold(x, th):
    return np.maximum(np.minimum(x, th), -th)


def compose_matrix44(t, dtype=np.double):
    """ Compose a 4x4 transformation matrix.

    Parameters
    ----------
    t : ndarray
        This is a 1D vector of affine transformation parameters with
        size at least 3.
        If the size is 3, t is interpreted as translation.
        If the size is 6, t is interpreted as translation + rotation.
        If the size is 7, t is interpreted as translation + rotation +
        isotropic scaling.
        If the size is 9, t is interpreted as translation + rotation +
        anisotropic scaling.
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

    if size not in [3, 6, 7, 9, 12]:
        raise ValueError('Accepted number of parameters is 3, 6, 7, 9 and 12')

    MAX_DIST = 1e10
    scale, shear, angles, translate = (None, ) * 4
    translate = _threshold(t[0:3], MAX_DIST)
    if size in [6, 7, 9, 12]:
        angles = np.deg2rad(t[3:6])
    if size == 7:
        scale = np.array((t[6],) * 3)
    if size in [9, 12]:
        scale = t[6: 9]
    if size == 12:
        shear = t[9: 12]
    return compose_matrix(scale=scale, shear=shear,
                          angles=angles,
                          translate=translate)


def decompose_matrix44(mat, size=12):
    """ Given a 4x4 homogeneous matrix return the parameter vector.

    Parameters
    ----------
    mat : array
        Homogeneous 4x4 transformation matrix
    size : int
        Size of the output vector. 3, for translation, 6 for rigid,
        7 for similarity, 9 for scaling and 12 for affine. Default is 12.

    Returns
    -------
    t : ndarray
        One dimensional ndarray of 3, 6, 7, 9 or 12 affine parameters.

    """
    scale, shear, angles, translate, _ = decompose_matrix(mat)

    t = np.zeros(12)
    t[:3] = translate
    if size == 3:
        return t[:3]
    t[3: 6] = np.rad2deg(angles)
    if size == 6:
        return t[:6]
    if size == 7:
        t[6] = np.mean(scale)
        return t[:7]
    if size == 9:
        t[6:9] = scale
        return t[:9]
    if size == 12:
        t[6: 9] = scale
        t[9: 12] = shear
        return t

    raise ValueError('Size can be 3, 6, 7, 9 or 12')
