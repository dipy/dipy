""" Classes and functions for generalized q-sampling """
import numpy as np
from dipy.reconst.odf import OdfModel, OdfFit
from dipy.reconst.cache import Cache
import warnings
from dipy.reconst.multi_voxel import multi_voxel_fit


class GeneralizedQSamplingModel(OdfModel, Cache):
    def __init__(self,
                 gtab,
                 method='gqi2',
                 sampling_length=1.2,
                 normalize_peaks=False):
        r""" Generalized Q-Sampling Imaging [1]_

        This model has the same assumptions as the DSI method i.e. Cartesian
        grid sampling in q-space and fast gradient switching.

        Implements equations 2.14 from [2]_ for standard GQI and equation 2.16
        from [2]_ for GQI2. You can think of GQI2 as an analytical solution of
        the DSI ODF.

        Parameters
        ----------
        gtab : object,
            GradientTable
        method : str,
            'standard' or 'gqi2'
        sampling_length : float,
            diffusion sampling length (lambda in eq. 2.14 and 2.16)

        References
        ----------
        .. [1] Yeh F-C et al., "Generalized Q-Sampling Imaging", IEEE TMI, 2010

        .. [2] Garyfallidis E, "Towards an accurate brain tractography", PhD
        thesis, University of Cambridge, 2012.

        Notes
        -----
        As of version 0.9, range of the sampling length in GQI2 has changed
        to match the same scale used in the 'standard' method [1]_. This
        means that the value of `sampling_length` should be approximately
        1 - 1.3 (see [1]_, pg. 1628).

        Examples
        --------
        Here we create an example where we provide the data, a gradient table
        and a reconstruction sphere and calculate the ODF for the first
        voxel in the data.

        >>> from dipy.data import dsi_voxels
        >>> data, gtab = dsi_voxels()
        >>> from dipy.core.subdivide_octahedron import create_unit_sphere
        >>> sphere = create_unit_sphere(5)
        >>> from dipy.reconst.gqi import GeneralizedQSamplingModel
        >>> gq = GeneralizedQSamplingModel(gtab, 'gqi2', 1.1)
        >>> voxel_signal = data[0, 0, 0]
        >>> odf = gq.fit(voxel_signal).odf(sphere)

        See Also
        --------
        dipy.reconst.dsi.DiffusionSpectrumModel

        """
        OdfModel.__init__(self, gtab)
        self.method = method
        self.Lambda = sampling_length
        self.normalize_peaks = normalize_peaks
        # 0.01506 = 6*D where D is the free water diffusion coefficient
        # l_values sqrt(6 D tau) D free water diffusion coefficient and
        # tau included in the b-value
        scaling = np.sqrt(self.gtab.bvals * 0.01506)
        tmp = np.tile(scaling, (3, 1))
        gradsT = self.gtab.bvecs.T
        b_vector = gradsT * tmp  # element-wise product
        self.b_vector = b_vector.T

    @multi_voxel_fit
    def fit(self, data):
        return GeneralizedQSamplingFit(self, data)


class GeneralizedQSamplingFit(OdfFit):

    def __init__(self, model, data):
        """ Calculates PDF and ODF for a single voxel

        Parameters
        ----------
        model : object,
            DiffusionSpectrumModel
        data : 1d ndarray,
            signal values

        """
        OdfFit.__init__(self, model, data)
        self._gfa = None
        self.npeaks = 5
        self._peak_values = None
        self._peak_indices = None
        self._qa = None

    def odf(self, sphere):
        """ Calculates the discrete ODF for a given discrete sphere.
        """
        self.gqi_vector = self.model.cache_get('gqi_vector', key=sphere)
        if self.gqi_vector is None:
            if self.model.method == 'gqi2':
                H = squared_radial_component
                # print self.gqi_vector.shape
                self.gqi_vector = np.real(H(np.dot(
                    self.model.b_vector, sphere.vertices.T) *
                    self.model.Lambda))
            if self.model.method == 'standard':
                self.gqi_vector = np.real(np.sinc(np.dot(
                    self.model.b_vector, sphere.vertices.T) *
                    self.model.Lambda / np.pi))
            self.model.cache_set('gqi_vector', sphere, self.gqi_vector)

        return np.dot(self.data, self.gqi_vector)


def normalize_qa(qa, max_qa=None):
    """ Normalize quantitative anisotropy.

    Used mostly with GQI rather than GQI2.

    Parameters
    ----------
    qa : array, shape (X, Y, Z, N)
        where N is the maximum number of peaks stored
    max_qa : float,
        maximum qa value. Usually found in the CSF (corticospinal fluid).

    Returns
    -------
    nqa : array, shape (x, Y, Z, N)
        normalized quantitative anisotropy

    Notes
    -----
    Normalized quantitative anisotropy has the very useful property
    to be very small near gray matter and background areas. Therefore,
    it can be used to mask out white matter areas.

    """
    if max_qa is None:
        return qa / qa.max()
    return qa / max_qa


def squared_radial_component(x, tol=0.01):
    """ Part of the GQI2 integral

    Eq.8 in the referenced paper by Yeh et al. 2010
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = (2 * x * np.cos(x) + (x * x - 2) * np.sin(x)) / (x ** 3)
    x_near_zero = (x < tol) & (x > -tol)
    return np.where(x_near_zero, 1./3, result)


def npa(self, odf, width=5):
    """ non-parametric anisotropy

    Nimmo-Smith et al.  ISMRM 2011
    """
    # odf = self.odf(s)
    t0, t1, t2 = triple_odf_maxima(self.odf_vertices, odf, width)
    psi0 = t0[1] ** 2
    psi1 = t1[1] ** 2
    psi2 = t2[1] ** 2
    npa = (np.sqrt(
        (psi0 - psi1) ** 2 +
        (psi1 - psi2) ** 2 +
        (psi2 - psi0) ** 2) /
        np.sqrt(2 * (psi0 ** 2 + psi1 ** 2 + psi2 ** 2)))
    # print 'tom >>>> ',t0,t1,t2,npa

    return t0, t1, t2, npa


def equatorial_zone_vertices(vertices, pole, width=5):
    """
    finds the 'vertices' in the equatorial zone conjugate
    to 'pole' with width half 'width' degrees
    """
    return [i
            for i, v in enumerate(vertices)
            if np.abs(np.dot(v, pole)) < np.abs(np.sin(np.pi * width / 180))]


def polar_zone_vertices(vertices, pole, width=5):
    """
    finds the 'vertices' in the equatorial band around
    the 'pole' of radius 'width' degrees
    """
    return [i
            for i, v in enumerate(vertices)
            if np.abs(np.dot(v, pole)) > np.abs(np.cos(np.pi * width / 180))]


def upper_hemi_map(v):
    """
    maps a 3-vector into the z-upper hemisphere
    """
    return np.sign(v[2])*v


def equatorial_maximum(vertices, odf, pole, width):
    eqvert = equatorial_zone_vertices(vertices, pole, width)
    # need to test for whether eqvert is empty or not
    if len(eqvert) == 0:
        print('empty equatorial band at %s  pole with width %f' %
              (np.array_str(pole), width))
        return None, None
    eqvals = [odf[i] for i in eqvert]
    eqargmax = np.argmax(eqvals)
    eqvertmax = eqvert[eqargmax]
    eqvalmax = eqvals[eqargmax]

    return eqvertmax, eqvalmax


def patch_vertices(vertices, pole, width):
    """
    find 'vertices' within the cone of 'width' degrees around 'pole'
    """
    return [i
            for i, v in enumerate(vertices)
            if np.abs(np.dot(v, pole)) > np.abs(np.cos(np.pi * width / 180))]


def patch_maximum(vertices, odf, pole, width):
    eqvert = patch_vertices(vertices, pole, width)
    # need to test for whether eqvert is empty or not
    if len(eqvert) == 0:
        print('empty cone around pole %s with with width %f' %
              (np.array_str(pole), width))
        return np.Null, np.Null
    eqvals = [odf[i] for i in eqvert]
    eqargmax = np.argmax(eqvals)
    eqvertmax = eqvert[eqargmax]
    eqvalmax = eqvals[eqargmax]
    return eqvertmax, eqvalmax


def odf_sum(odf):
    return np.sum(odf)


def patch_sum(vertices, odf, pole, width):
    eqvert = patch_vertices(vertices, pole, width)
    # need to test for whether eqvert is empty or not
    if len(eqvert) == 0:
        print('empty cone around pole %s with with width %f' %
              (np.array_str(pole), width))
        return np.Null
    return np.sum([odf[i] for i in eqvert])


def triple_odf_maxima(vertices, odf, width):

    indmax1 = np.argmax([odf[i] for i, v in enumerate(vertices)])
    odfmax1 = odf[indmax1]
    pole = vertices[indmax1]
    eqvert = equatorial_zone_vertices(vertices, pole, width)
    indmax2, odfmax2 = equatorial_maximum(vertices, odf, pole, width)
    indmax3 = eqvert[np.argmin([np.abs(np.dot(vertices[indmax2], vertices[p]))
                                for p in eqvert])]
    odfmax3 = odf[indmax3]
    """
    cross12 = np.cross(vertices[indmax1],vertices[indmax2])
    cross12 = cross12/np.sqrt(np.sum(cross12**2))
    indmax3, odfmax3 = patch_maximum(vertices, odf, cross12, 2*width)
    """
    return [(indmax1, odfmax1), (indmax2, odfmax2), (indmax3, odfmax3)]
