import numpy as np
from .shm import real_sym_sh_basis
from .peaks import peak_directions
from ..tracking.localtrack import PythonDirectionGetter


sign = np.array([-1., 1.]).reshape((2, 1))
edge = np.array([1., 0.]).reshape((2, 1))
def trilinear_weights(point):
    w = edge + sign * (point % 1.)
    w = np.ix_(*w.T)
    return w[0] * w[1] * w[2]

class ProbabilisticOdfWightedDirectionGetter(PythonDirectionGetter):
    """Returns a random direction from a sphere weighted by odf values

    """
    def __init__(self, shmFit, sphere, max_angle):
        self.shmFit = shmFit
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

    def _interpolate_shm_coeff(self, point):
        index = []
        for p in np.floor(point):
            index.append(slice(p, p+2))
        coeff = self.shmFit.shm_coeff[index]
        weights = trilinear_weights(point)
        coeff *= weights[..., None]
        coeff = coeff.reshape((8, -1))
        return coeff.sum(0)

    def _odf_at(self, point):
        coeff = self._interpolate_shm_coeff(point)
        odf = np.dot(self._B, coeff)
        odf.clip(0, out=odf)
        return odf

    def initial_direction(self, point):
        odf = self._odf_at(point)
        return peak_directions(odf, self.sphere)[0]

    def _get_direction(self, point, prev_dir):
        odf = self._odf_at(point)
        cdf = (self._adj_matrix[tuple(prev_dir)] * odf).cumsum()
        if cdf[-1] == 0:
            return None
        random_sample = np.random.random() * cdf[-1]
        idx = cdf.searchsorted(random_sample, 'right')
        direction = self.sphere.vertices[idx]
        if np.dot(direction, prev_dir) > 0:
            return direction
        else:
            return -direction

