import numpy as np
from dipy.reconst.peaks import peak_directions, default_sphere
from dipy.reconst.shm import (bootstrap_data_voxel, cart2sphere, hat,
                              lazy_index, lcr_matrix, normalize_data,
                              real_sym_sh_basis)
from ..markov import _closest_peak
from .direction_getter import DirectionGetter
from .interpolation import trilinear_interpolate4d

class ProbabilisticOdfWightedDirectionGetter(DirectionGetter):
    """Returns a random direction from a sphere weighted by odf values

    """
    def __init__(self, shmFit, sphere, max_angle):
        self.shmFit = shmFit
        self._shm_coeff = self.shmFit.shm_coeff
        self.sphere = sphere
        B = real_sym_sh_basis(shmFit.model.sh_order, sphere.theta, sphere.phi)
        self._B = B[0]
        self._set_adjacency_matrix(sphere, np.cos(np.deg2rad(max_angle)))

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

    def _odf_at(self, point):
        coeff = trilinear_interpolate4d(self._shm_coeff, point)
        odf = np.dot(self._B, coeff)
        odf.clip(0, out=odf)
        return odf

    def initial_direction(self, point):
        odf = self._odf_at(point)
        return peak_directions(odf, self.sphere)[0]

    def get_direction(self, point, direction):
        # get direction gets memory views as inputs
        point = np.array(point, copy=False)
        direction = np.array(direction, copy=False)

        odf = self._odf_at(point)
        cdf = (self._adj_matrix[tuple(direction)] * odf).cumsum()
        if cdf[-1] == 0:
            return 1
        random_sample = np.random.random() * cdf[-1]
        idx = cdf.searchsorted(random_sample, 'right')

        newdir = self.sphere.vertices[idx]
        tmp = float(direction[0])
        # Update direction and return 0 for error
        if np.dot(newdir, direction) > 0:
            direction[:] = newdir
        else:
            direction[:] = -newdir
        return 0

