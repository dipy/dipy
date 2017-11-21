# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

"""
Implementation of a probabilistic direction getter based on sampling from
discrete distribution (pmf) at each step of the tracking.
"""

from random import random

import numpy as np
cimport numpy as np

from dipy.direction.peaks import peak_directions, default_sphere
from dipy.direction.pmf cimport PmfGen, SimplePmfGen, SHCoeffPmfGen
from dipy.tracking.local.direction_getter cimport DirectionGetter
from dipy.utils.fast_numpy cimport cumsum, where_to_insert


cdef class PeakDirectionGetter(DirectionGetter):
    """An abstract class for DirectionGetters that use the peak_directions
    machinery."""

    cdef:
        object sphere
        dict _pf_kwargs

    def __init__(self, sphere=None, **kwargs):
        if sphere is None:
            self.sphere = default_sphere
        else:
            self.sphere = sphere
        self._pf_kwargs = kwargs

    def _peak_directions(self, blob):
        """Gets directions using parameters provided at init.

        Blob can be any function defined on ``self.sphere``, ie an ODF, PMF,
        FOD.
        """
        return peak_directions(blob, self.sphere, **self._pf_kwargs)[0]


cdef class ProbabilisticDirectionGetter(PeakDirectionGetter):
    """Randomly samples direction of a sphere based on probability mass
    function (pmf).

    The main constructors for this class are current from_pmf and from_shcoeff.
    The pmf gives the probability that each direction on the sphere should be
    chosen as the next direction. To get the true pmf from the "raw pmf"
    directions more than ``max_angle`` degrees from the incoming direction are
    set to 0 and the result is normalized.

    """

    cdef:
        PmfGen pmf_gen
        double pmf_threshold
        double[:, :] vertices
        dict _adj_matrix
        # double[:] pmf

    @classmethod
    def from_pmf(klass, pmf, max_angle, sphere, pmf_threshold=0.1, **kwargs):
        """Constructor for making a DirectionGetter from an array of Pmfs

        Parameters
        ----------
        pmf : array, 4d
            The pmf to be used for tracking at each voxel.
        max_angle : float, [0, 90]
            The maximum allowed angle between incoming direction and new
            direction.
        sphere : Sphere
            The set of directions to be used for tracking.
        pmf_threshold : float [0., 1.]
            Used to remove direction from the probability mass function for
            selecting the tracking direction.
        relative_peak_threshold : float in [0., 1.]
            Used for extracting initial tracking directions. Passed to
            peak_directions.
        min_separation_angle : float in [0, 90]
            Used for extracting initial tracking directions. Passed to
            peak_directions.

        See also
        --------
        dipy.direction.peaks.peak_directions


        """
        pmf = np.asarray(pmf, dtype=float)
        if pmf.ndim != 4:
            raise ValueError("pmf should be a 4d array.")
        if pmf.shape[3] != len(sphere.theta):
            msg = ("The last dimension of pmf should match the number of "
                   "points in sphere.")
            raise ValueError(msg)
        pmf_gen = SimplePmfGen(pmf)
        return klass(pmf_gen, max_angle, sphere, pmf_threshold, **kwargs)

    @classmethod
    def from_shcoeff(klass, shcoeff, max_angle, sphere, pmf_threshold=0.1,
                     basis_type=None, **kwargs):
        """Probabilistic direction getter from a distribution of directions
        on the sphere.

        Parameters
        ----------
        shcoeff : array
            The distribution of tracking directions at each voxel represented
            as a function on the sphere using the real spherical harmonic
            basis. For example the FOD of the Constrained Spherical
            Deconvolution model can be used this way. This distribution will
            be discretized using ``sphere`` and tracking directions will be
            chosen from the vertices of ``sphere`` based on the distribution.
        max_angle : float, [0, 90]
            The maximum allowed angle between incoming direction and new
            direction.
        sphere : Sphere
            The set of directions to be used for tracking.
        pmf_threshold : float [0., 1.]
            Used to remove direction from the probability mass function for
            selecting the tracking direction.
        basis_type : name of basis
            The basis that ``shcoeff`` are associated with.
            ``dipy.reconst.shm.real_sym_sh_basis`` is used by default.
        relative_peak_threshold : float in [0., 1.]
            Used for extracting initial tracking directions. Passed to
            peak_directions.
        min_separation_angle : float in [0, 90]
            Used for extracting initial tracking directions. Passed to
            peak_directions.

        See also
        --------
        dipy.direction.peaks.peak_directions

        """
        pmf_gen = SHCoeffPmfGen(shcoeff, sphere, basis_type)
        return klass(pmf_gen, max_angle, sphere, pmf_threshold, **kwargs)

    def __init__(self, pmf_gen, max_angle, sphere=None, pmf_threshold=0.1,
                 **kwargs):
        """Direction getter from a pmf generator.

        Parameters
        ----------
        pmf_gen : PmfGen
            Used to get probability mass function for selecting tracking
            directions.
        max_angle : float, [0, 90]
            The maximum allowed angle between incoming direction and new
            direction.
        sphere : Sphere
            The set of directions to be used for tracking.
        pmf_threshold : float [0., 1.]
            Used to remove direction from the probability mass function for
            selecting the tracking direction.
        relative_peak_threshold : float in [0., 1.]
            Used for extracting initial tracking directions. Passed to
            peak_directions.
        min_separation_angle : float in [0, 90]
            Used for extracting initial tracking directions. Passed to
            peak_directions.

        See also
        --------
        dipy.direction.peaks.peak_directions

        """
        PeakDirectionGetter.__init__(self, sphere, **kwargs)
        self.pmf_gen = pmf_gen
        self.pmf_threshold = pmf_threshold
        # The vertices need to be in a contiguous array
        self.vertices = self.sphere.vertices.copy()
        cos_similarity = np.cos(np.deg2rad(max_angle))
        self._set_adjacency_matrix(sphere, cos_similarity)

    def _set_adjacency_matrix(self, sphere, cos_similarity):
        """Creates a dictionary where each key is a direction from sphere and
        each value is a boolean array indicating which directions are less than
        max_angle degrees from the key"""
        matrix = np.dot(sphere.vertices, sphere.vertices.T)
        matrix = (abs(matrix) >= cos_similarity).astype('uint8')
        keys = [tuple(v) for v in sphere.vertices]
        adj_matrix = dict(zip(keys, matrix))
        keys = [tuple(-v) for v in sphere.vertices]
        adj_matrix.update(zip(keys, matrix))
        self._adj_matrix = adj_matrix

    cpdef np.ndarray[np.float_t, ndim=2] initial_direction(
            self, double[::1] point):
        """Returns best directions at seed location to start tracking.

        Parameters
        ----------
        point : ndarray, shape (3,)
            The point in an image at which to lookup tracking directions.

        Returns
        -------
        directions : ndarray, shape (N, 3)
            Possible tracking directions from point. ``N`` may be 0, all
            directions should be unique.

        """
        cdef double[:] pmf = self.pmf_gen.get_pmf(point)
        return self._peak_directions(pmf)

    cdef int get_direction_c(self, double* point, double* direction):
        """Samples a pmf to updates ``direction`` array with a new direction.

        Parameters
        ----------
        point : memory-view (or ndarray), shape (3,)
            The point in an image at which to lookup tracking directions.
        direction : memory-view (or ndarray), shape (3,)
            Previous tracking direction.

        Returns
        -------
        status : int
            Returns 0 `direction` was updated with a new tracking direction, or
            1 otherwise.

        """
        cdef:
            size_t i, idx, _len
            double[:] newdir, pmf, cdf
            double last_cdf, random_sample
            np.uint8_t[:] bool_array

        # point and direction are passed in as cython memory views
        pmf = self.pmf_gen.get_pmf_c(point)
        _len = pmf.shape[0]
        for i in range(_len):
            if pmf[i] < self.pmf_threshold:
                pmf[i] = 0.0

        bool_array = self._adj_matrix[
            (direction[0], direction[1], direction[2])]

        cdf = pmf
        for i in range(_len):
            if bool_array[i] == 0:
                cdf[i] = 0.0
        cumsum(&cdf[0], &cdf[0], _len)

        last_cdf = cdf[_len - 1]
        if last_cdf == 0:
            return 1

        random_sample = random() * last_cdf
        idx = where_to_insert(&cdf[0], random_sample, _len)

        newdir = self.vertices[idx, :]
        # Update direction and return 0 for error
        if direction[0] * newdir[0] \
         + direction[1] * newdir[1] \
         + direction[2] * newdir[2] > 0:
            direction[0] = newdir[0]
            direction[1] = newdir[1]
            direction[2] = newdir[2]
        else:
            direction[0] = -newdir[0]
            direction[1] = -newdir[1]
            direction[2] = -newdir[2]
        return 0


cdef class DeterministicMaximumDirectionGetter(ProbabilisticDirectionGetter):
    """Return direction of a sphere with the highest probability mass
    function (pmf).
    """
    cdef int get_direction_c(self, double* point, double* direction):
        """Find direction with the highest pmf to updates ``direction`` array
        with a new direction.
        Parameters
        ----------
        point : memory-view (or ndarray), shape (3,)
            The point in an image at which to lookup tracking directions.
        direction : memory-view (or ndarray), shape (3,)
            Previous tracking direction.
        Returns
        -------
        status : int
            Returns 0 `direction` was updated with a new tracking direction, or
            1 otherwise.
        """
        cdef:
            size_t _len, max_idx
            double[:] newdir, pmf, cdf
            double max_value

        # point and direction are passed in as cython memory views
        pmf = self.pmf_gen.get_pmf_c(point)
        _len = pmf.shape[0]
        for i in range(_len):
            if pmf[i] < self.pmf_threshold:
                pmf[i] = 0.0

        cdf = self._adj_matrix[
            (direction[0], direction[1], direction[2])] * pmf
        max_idx = 0
        max_value = 0.0
        for i in range(_len):
            if cdf[i] > max_value:
                max_idx = i
                max_value = cdf[i]

        if pmf[max_idx] == 0:
            return 1

        newdir = self.vertices[max_idx]
        # Update direction and return 0 for error
        if direction[0] * newdir[0] \
         + direction[1] * newdir[1] \
         + direction[2] * newdir[2] > 0:
            direction[0] = newdir[0]
            direction[1] = newdir[1]
            direction[2] = newdir[2]
        else:
            direction[0] = -newdir[0]
            direction[1] = -newdir[1]
            direction[2] = -newdir[2]
        return 0
