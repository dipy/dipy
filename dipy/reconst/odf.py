from __future__ import division
import numpy as np
from .recspeed import local_maxima, _filter_peaks

#Classes OdfModel and OdfFit are using API ReconstModel and ReconstFit from .base 

class OdfFit(object):
    pass

class OdfModel(object):
    """An abstract class to be sub-classed by specific odf models

    Attributes
    ----------
    relative_peak_threshold : float
        The minimum peak value, relative to the largest peak.
    angular_distance_threshold : float between 0 and 90
        The minimum angular distance between two peaks.
    odf_vertices : array (n, 3)
        The x, y, z coordinates of n points on a unit sphere.
    odf_edges : array, shape == (m, 2), dtype == int16
        A list of neighboring vertices.
    """
    relative_peak_threshold = .25
    _cos_distance_threshold = np.sqrt(2) / 2
    _distance_matrix = None
    _odf_vertices = None
    _odf_edges = None

    @property
    def odf_vertices(self):
        return self._odf_vertices
    @property
    def odf_edges(self):
        return self._odf_edges
    @property
    def angular_distance_threshold(self):
        return 180/np.pi * np.arccos(self._cos_distance_threshold)
    @angular_distance_threshold.setter
    def angular_distance_threshold(self, angle):
        if angle < 0 or angle > 90:
            raise ValueError("angle must be between 0 and 90")
        self._cos_distance_threshold = np.cos(np.pi/180 * angle)

    def set_odf_vertices(self, vertices, edges=None, faces=None):
        """Sets the vertices used to evaluate the odf or get odf peaks

        Parameters
        ----------
        vertices : ndarray (n, 3), dtype=float
            The x, y, z coordinates of n points on a unit sphere.
        edges : ndarray (m, 2), dtype=int16, optional
            A list neighboring vertices, for example if (1, 2) is in edges
            then vertex 1 and 2 are treated as neighbors.
        """
        on_unit_sphere = np.allclose((vertices*vertices).sum(-1), 1)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not on_unit_sphere:
            raise ValueError("vertices should be an (n, 3) array of points on "
                             "the unit sphere")
        if (edges is not None) and (edges.ndim != 2 or edges.shape[1] != 2):
            raise ValueError("Edges should be an (m, 2) array with a list of "
                             "neighboring vertices")
        self._odf_vertices = vertices
        self._odf_edges = edges
        self._odf_faces = faces
        self._distance_matrix = abs(vertices.dot(vertices.T))


    def evaluate_odf():
        """To be implemented by subclasses"""
        raise NotImplementedError()

    def get_directions(self, sig):
        """Estimate of fiber directions using model"""
        odf = self.evaluate_odf(sig)
        pk, ind = local_maxima(odf, self._odf_edges)
        pk, ind = _filter_peaks(pk, ind, self._distance_matrix,
                                self.relative_peak_threshold,
                                self._cos_distance_threshold)
        return self._odf_vertices[ind]


    def fit(self, data, mask=None, return_odf=False, normalize_peaks=False):
        """Fits the model to data and returns an OdfFit"""

        data_flat = data.reshape((-1, data.shape[-1]))
        size = len(data_flat)
        if mask is None:
            mask = np.ones(size, dtype='bool')
        else:
            mask = mask.ravel()
            if len(mask) != size:
                raise ValueError("mask is not the same size as data")

        npeaks = 5
        gfa_array = np.zeros(size)
        qa_array = np.zeros((size, npeaks))
        peak_values = np.zeros((size, npeaks))
        peak_indices = np.zeros((size, npeaks), dtype='int')
        peak_indices.fill(-1)

        if return_odf:
            odf_array = np.zeros((size, len(self.odf_vertices)))

        global_max = -np.inf
        for i, sig in enumerate(data_flat):
            if not mask[i]:
                continue
            odf = self.evaluate_odf(sig)
            if return_odf:
                odf_array[i] = odf

            gfa_array[i] = gfa(odf)
            pk, ind = local_maxima(odf, self.odf_edges)
            pk, ind = _filter_peaks(pk, ind,
                                    self._distance_matrix,
                                    self.relative_peak_threshold,
                                    self._cos_distance_threshold)

            global_max = max(global_max, pk[0])
            n = min(npeaks, len(pk))
            qa_array[i, :n] = pk[:n] - odf.min()
            if normalize_peaks:
                peak_values[i, :n] = pk[:n] / pk[0]
            else:
                peak_values[i, :n] = pk[:n]
            peak_indices[i, :n] = ind[:n]

        shape = data.shape[:-1]
        gfa_array = gfa_array.reshape(shape)
        qa_array = qa_array.reshape(shape + (npeaks,)) / global_max
        peak_values = peak_values.reshape(shape + (npeaks,))
        peak_indices = peak_indices.reshape(shape + (npeaks,))

        odffit = OdfFit()
        odffit.peak_values = peak_values
        odffit.peak_indices = peak_indices
        odffit.gfa = gfa_array
        odffit.qa = qa_array
        if return_odf:
            odffit.odf = odf_array

        return odffit


def gfa(samples):
    """The gfa of a function evaluated on the unit sphere"""
    diff = samples - samples.mean(-1)[..., None]
    n = samples.shape[-1]
    numer = n*(diff*diff).sum(-1)
    denom = (n-1)*(samples*samples).sum(-1)
    return np.sqrt(numer/denom)
