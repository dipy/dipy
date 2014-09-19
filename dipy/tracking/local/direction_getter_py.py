import numpy as np
from dipy.reconst.peaks import peak_directions, default_sphere
from dipy.reconst.shm import (bootstrap_data_voxel, cart2sphere, hat,
                              lazy_index, lcr_matrix, normalize_data,
                              real_sym_sh_basis)
from .direction_getter import DirectionGetter
from .interpolation import trilinear_interpolate4d


def _asarray(cython_memview):
    # TODO: figure out the best way to get an array from a memory view.
    # `np.array(view)` works, but is quite slow. Views are also "array_like",
    # but using them as arrays seems to also be quite slow.
    return np.fromiter(cython_memview, float)


class PmfGen(object):
    pass


class SimplePmfGen(PmfGen):

    def __init__(self, pmf_array):
        if pmf_array.ndim != 4:
            raise ValueError("expecting 4d array")
        self.pmf_array = pmf_array

    def get_pmf(self, point):
        return trilinear_interpolate4d(self.pmf_array, point)


class ShmFitPmfGen(PmfGen):

    def __init__(self, shmfit, sphere):
        self.fit = shmfit
        self.sphere = sphere
        self._B = shmfit.model.sampling_matrix(sphere)
        self._coeff = shmfit.shm_coeff

    def get_pmf(self, point):
        coeff = trilinear_interpolate4d(self._coeff, point)
        odf = np.dot(self._B, coeff)
        pmf = odf.clip(0)
        return pmf


class ProbabilisticDirectionGetter(DirectionGetter):
    """Randomly samples direction of a sphere based on probability mass
    function (pmf).

    The main constructors for this class are current fromPmf and fromShmFit.
    The pmf gives the probability that each direction on the sphere should be
    chosen as the next direction. To get the true pmf from the "raw pmf"
    directions more than ``max_angle`` degrees from the incoming direction are
    set to 0 and the result is normalized.

    """
    @classmethod
    def fromPmf(klass, pmf, sphere, max_angle):
        """Constructor for making a DirectionGetter from an array of Pmfs

        Parameters
        ----------
        pmf : array, 4d
            The pmf to be used for tracking at each voxel.
        sphere : Sphere
            The set of directions to be used for tracking.
        max_angle : float, [0, 90]
            The maximum allowed angle between incoming direction and new
            direction.

        """
        pmf = np.asarray(pmf, dtype=float)
        if pmf.shape[3] != len(sphere.theta):
            raise ValueError("pmf and sphere do not match")
        if pmf.min() < 0:
            raise ValueError("pmf should not have negative values")
        pmf_gen = SimplePmfGen(pmf)
        return klass(pmf_gen, sphere, max_angle)

    @classmethod
    def fromShmFit(klass, shmFit, sphere, max_angle):
        """Use the ODF (or FOD) of a SphHarmFit object as the pmf

        Parameters
        ----------
        shmFit : SphHarmFit
            Fit object to be used for tracking.
        sphere : Sphere
            The set of directions to be used for tracking.
        max_angle : float, [0, 90]
            The maximum allowed angle between incoming direction and new
            direction.

        """
        pmf_gen = ShmFitPmfGen(shmFit, sphere)
        return klass(pmf_gen, sphere, max_angle)

    def __init__(self, pmf_gen, sphere, max_angle):
        self.pmf_gen = pmf_gen
        self.sphere = sphere
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
        """Returns best directions at seed location to start tracking."""
        pmf = self.pmf_gen.get_pmf(point)
        return peak_directions(pmf, self.sphere)[0]

    def get_direction(self, point, direction):
        """Samples a pmf to updates ``direction`` array with a new direction"""
        # point and direction are passed in as cython memory views
        pmf = self.pmf_gen.get_pmf(point)
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

