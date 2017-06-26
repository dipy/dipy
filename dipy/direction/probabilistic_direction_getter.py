"""
Implementation of a probabilistic direction getter based on sampling from
discrete distribution (pmf) at each step of the tracking."""
import numpy as np
from dipy.direction.peaks import peak_directions, default_sphere
from dipy.reconst.shm import order_from_ncoef, sph_harm_lookup
from dipy.tracking.local.direction_getter import DirectionGetter
from dipy.tracking.local.interpolation import trilinear_interpolate4d


def _asarray(cython_memview):
    # TODO: figure out the best way to get an array from a memory view.
    # `np.array(view)` works, but is quite slow. Views are also "array_like",
    # but using them as arrays seems to also be quite slow.
    return np.fromiter(cython_memview, float)


class PmfGen(object):
    pass


class SimplePmfGen(PmfGen):

    def __init__(self, pmf_array):
        if pmf_array.min() < 0:
            raise ValueError("pmf should not have negative values")
        self.pmf_array = pmf_array

    def get_pmf(self, point):
        return trilinear_interpolate4d(self.pmf_array, point)


class SHCoeffPmfGen(PmfGen):

    def __init__(self, shcoeff, sphere, basis_type):
        self.shcoeff = shcoeff
        self.sphere = sphere
        sh_order = order_from_ncoef(shcoeff.shape[-1])
        try:
            basis = sph_harm_lookup[basis_type]
        except KeyError:
            raise ValueError("%s is not a known basis type." % basis_type)
        self._B, m, n = basis(sh_order, sphere.theta, sphere.phi)

    def get_pmf(self, point):
        coeff = trilinear_interpolate4d(self.shcoeff, point)
        pmf = np.dot(self._B, coeff)
        pmf.clip(0, out=pmf)
        return pmf


class PeakDirectionGetter(DirectionGetter):
    """An abstract class for DirectionGetters that use the peak_directions
    machinery."""

    sphere = default_sphere

    def __init__(self, sphere=None, **kwargs):
        if sphere is not None:
            self.sphere = sphere
        self._pf_kwargs = kwargs

    def _peak_directions(self, blob):
        """Gets directions using parameters provided at init.

        Blob can be any function defined on ``self.sphere``, ie an ODF, PMF,
        FOD.
        """
        return peak_directions(blob, self.sphere, **self._pf_kwargs)[0]


class ProbabilisticDirectionGetter(PeakDirectionGetter):
    """Randomly samples direction of a sphere based on probability mass
    function (pmf).

    The main constructors for this class are current from_pmf and from_shcoeff.
    The pmf gives the probability that each direction on the sphere should be
    chosen as the next direction. To get the true pmf from the "raw pmf"
    directions more than ``max_angle`` degrees from the incoming direction are
    set to 0 and the result is normalized.

    """
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
        matrix = abs(matrix) >= cos_similarity
        keys = [tuple(v) for v in sphere.vertices]
        adj_matrix = dict(zip(keys, matrix))
        keys = [tuple(-v) for v in sphere.vertices]
        adj_matrix.update(zip(keys, matrix))
        self._adj_matrix = adj_matrix

    def initial_direction(self, point):
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
        pmf = self.pmf_gen.get_pmf(point)
        return self._peak_directions(pmf)

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
