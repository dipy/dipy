import numpy as np
from dipy.reconst.peaks import peak_directions, default_sphere
from dipy.reconst.shm import (bootstrap_data_voxel, cart2sphere, hat,
                              lazy_index, lcr_matrix, normalize_data,
                              real_sym_sh_basis)
from ..markov import _closest_peak
from .direction_getter import DirectionGetter
from .interpolation import trilinear_interpolate4d


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

    def __init__(self, shmFit, sphere):
        self.fit = shmFit
        self.sphere = sphere
        self._B = fit.sampling_matrix(sphere)
        self._ceoff = fit.shm_coeff

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
        pmfGen = ShmPmfGen(shmFit, sphere)
        return klass(pmf_gen, sphere, max_angle)

    def __init__(self, pmf_gen, sphere, max_angle):
        self.pmf_gen = pmf_gen
        self.sphere = sphere
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
        # get direction gets memory views as inputs
        point = np.array(point, copy=False)
        direction = np.array(direction, copy=False)

        pmf = self.pmf_gen.get_pmf(point)
        cdf = (self._adj_matrix[tuple(direction)] * pmf).cumsum()
        if cdf[-1] == 0:
            return 1
        random_sample = np.random.random() * cdf[-1]
        idx = cdf.searchsorted(random_sample, 'right')

        newdir = self.sphere.vertices[idx]
        # Update direction and return 0 for error
        if np.dot(newdir, direction) > 0:
            direction[:] = newdir
        else:
            direction[:] = -newdir
        return 0

