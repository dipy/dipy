# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

"""
Implementation of a probabilistic direction getter based on sampling from
discrete distribution (pmf) at each step of the tracking.
"""

from random import random

import numpy as np
cimport numpy as cnp

from dipy.direction.closest_peak_direction_getter cimport PmfGenDirectionGetter
from dipy.utils.fast_numpy cimport cumsum, where_to_insert


cdef class ProbabilisticDirectionGetter(PmfGenDirectionGetter):
    """Randomly samples direction of a sphere based on probability mass
    function (pmf).

    The main constructors for this class are current from_pmf and from_shcoeff.
    The pmf gives the probability that each direction on the sphere should be
    chosen as the next direction. To get the true pmf from the "raw pmf"
    directions more than ``max_angle`` degrees from the incoming direction are
    set to 0 and the result is normalized.
    """
    cdef:
        double[:, :] vertices
        dict _adj_matrix

    def __init__(self, pmf_gen, max_angle, sphere, pmf_threshold=.1, **kwargs):
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

        See Also
        --------
        dipy.direction.peaks.peak_directions

        """
        PmfGenDirectionGetter.__init__(self, pmf_gen, max_angle, sphere,
                                       pmf_threshold, **kwargs)
        # The vertices need to be in a contiguous array
        self.vertices = self.sphere.vertices.copy()
        self._set_adjacency_matrix(sphere, self.cos_similarity)

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
            cnp.npy_intp i, idx, _len
            double[:] newdir, pmf
            double last_cdf, random_sample
            cnp.uint8_t[:] bool_array

        pmf = self._get_pmf(point)
        _len = pmf.shape[0]

        bool_array = self._adj_matrix[
            (direction[0], direction[1], direction[2])]

        for i in range(_len):
            if bool_array[i] == 0:
                pmf[i] = 0.0
        cumsum(&pmf[0], &pmf[0], _len)
        last_cdf = pmf[_len - 1]

        if last_cdf == 0:
            return 1

        random_sample = random() * last_cdf
        idx = where_to_insert(&pmf[0], random_sample, _len)

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

    def __init__(self, pmf_gen, max_angle, sphere, pmf_threshold=.1, **kwargs):
        ProbabilisticDirectionGetter.__init__(self, pmf_gen, max_angle, sphere,
                                              pmf_threshold, **kwargs)

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
            cnp.npy_intp _len, max_idx
            double[:] newdir, pmf
            double max_value
            cnp.uint8_t[:] bool_array

        pmf = self._get_pmf(point)
        _len = pmf.shape[0]

        bool_array = self._adj_matrix[
            (direction[0], direction[1], direction[2])]

        max_idx = 0
        max_value = 0.0
        for i in range(_len):
            if bool_array[i] > 0 and pmf[i] > max_value:
                max_idx = i
                max_value = pmf[i]

        if max_value <= 0:
            return 1

        newdir = self.vertices[max_idx]
        # Update direction
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
