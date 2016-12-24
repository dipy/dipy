"""
Implementation of a probabilistic direction getter based on sampling from
discrete distribution (pmf) at each step of the tracking."""
import numpy as np
from dipy.direction.closest_peak import ClosestPeakDirectionGetter


def _asarray(cython_memview):
    # TODO: figure out the best way to get an array from a memory view.
    # `np.array(view)` works, but is quite slow. Views are also "array_like",
    # but using them as arrays seems to also be quite slow.
    return np.fromiter(cython_memview, float)


class ProbabilisticDirectionGetter(ClosestPeakDirectionGetter):
    """Randomly samples direction of a sphere based on probability mass
    function (pmf).

    The main constructors for this class are current from_pmf and from_shcoeff.
    The pmf gives the probability that each direction on the sphere should be
    chosen as the next direction. To get the true pmf from the "raw pmf"
    directions more than ``max_angle`` degrees from the incoming direction are
    set to 0 and the result is normalized.
    """

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
        ClosestPeakDirectionGetter.__init__(self, pmf_gen, max_angle, sphere,
                                            pmf_threshold, **kwargs)
        # The vertices need to be in a contiguous array
        self.vertices = self.sphere.vertices.copy()
        self._set_adjacency_matrix(sphere, self.cos_similarity)

    def _set_adjacency_matrix(self, sphere, cos_similarity):
        """Creates a dictionary where each key is a direction from sphere and
        each value is a boolean array indicating which directions are less than
        max_angle degrees from the key"""
        matrix = np.dot(sphere.vertices, sphere.vertices.T)
        matrix = abs(matrix) >= cos_similarity
        keys = [tuple(v) for v in sphere.vertices]
        adj_matrix = dict(zip(keys, matrix))
        keys = [tuple(-v) for v in sphere.vertices]
        adj_matrix.update(zip(keys, matrix))
        self._adj_matrix = adj_matrix

    def get_direction(self, point, direction):
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
        # point and direction are passed in as cython memory views
        pmf = self.pmf_gen.get_pmf(point)
        pmf[pmf < self.pmf_threshold] = 0
        cdf = (self._adj_matrix[tuple(direction)] * pmf).cumsum()
        if cdf[-1] == 0:
            return 1
        random_sample = np.random.random() * cdf[-1]
        idx = cdf.searchsorted(random_sample, 'right')

        newdir = self.vertices[idx]
        # Update direction and return 0 for error
        if np.dot(newdir, _asarray(direction)) > 0:
            direction[:] = newdir
        else:
            direction[:] = -newdir
        return 0


class DeterministicMaximumDirectionGetter(ProbabilisticDirectionGetter):
    """Return direction of a sphere with the highest probability mass
    function (pmf).
    """
    def get_direction(self, point, direction):
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
        # point and direction are passed in as cython memory views
        pmf = self.pmf_gen.get_pmf(point)
        pmf[pmf < self.pmf_threshold] = 0
        cdf = self._adj_matrix[tuple(direction)] * pmf
        idx = np.argmax(cdf)

        if pmf[idx] == 0:
            return 1

        newdir = self.vertices[idx]
        # Update direction and return 0 for error
        if np.dot(newdir, _asarray(direction)) > 0:
            direction[:] = newdir
        else:
            direction[:] = -newdir
        return 0

