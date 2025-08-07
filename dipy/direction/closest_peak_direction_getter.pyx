# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp

from dipy.direction.peaks import peak_directions, default_sphere
from dipy.direction.pmf cimport SimplePmfGen, SHCoeffPmfGen
from dipy.reconst import shm
from dipy.tracking.direction_getter cimport DirectionGetter
from dipy.utils.fast_numpy cimport copy_point, scalar_muliplication_point


cdef int closest_peak(cnp.ndarray[cnp.float_t, ndim=2] peak_dirs,
                      double[::1] direction, double cos_similarity):
    """Update direction with the closest direction from peak_dirs.

    All directions should be unit vectors. Antipodal symmetry is assumed, ie
    direction x is the same as -x.

    Parameters
    ----------
    peak_dirs : array (N, 3)
        N unit vectors.
    direction : array (3,) or None
        Previous direction. The new direction is saved here.
    cos_similarity : float
        `cos(max_angle)` where `max_angle` is the maximum allowed angle between
        prev_step and the returned direction.

    Returns
    -------
    0 : if ``direction`` is updated
    1 : if no new direction is founded
    """
    cdef:
        cnp.npy_intp _len=len(peak_dirs)
        cnp.npy_intp i
        int closest_peak_i=-1
        double _dot
        double closest_peak_dot=0

    for i in range(_len):
        _dot = (peak_dirs[i,0] * direction[0]
                + peak_dirs[i,1] * direction[1]
                + peak_dirs[i,2] * direction[2])

        if np.abs(_dot) > np.abs(closest_peak_dot):
            closest_peak_dot = _dot
            closest_peak_i = i

    if closest_peak_i >= 0:
        if closest_peak_dot >= cos_similarity:
            copy_point(&peak_dirs[closest_peak_i, 0], &direction[0])
            return 0
        if closest_peak_dot <= -cos_similarity:
            copy_point(&peak_dirs[closest_peak_i, 0], &direction[0])
            scalar_muliplication_point(&direction[0], -1)
            return 0
    return 1


cdef class BasePmfDirectionGetter(DirectionGetter):
    """A base class for dynamic direction getters"""

    def __init__(self, pmf_gen, max_angle, sphere, pmf_threshold=.1, **kwargs):
        self.sphere = sphere
        self._pf_kwargs = kwargs
        self.pmf_gen = pmf_gen
        if pmf_threshold < 0:
            raise ValueError("pmf threshold must be >= 0.")
        self.pmf_threshold = pmf_threshold
        self.cos_similarity = np.cos(np.deg2rad(max_angle))
        self.len_pmf = sphere.vertices.shape[0]

    def _get_peak_directions(self, blob):
        """Gets directions using parameters provided at init.

        Blob can be any function defined on ``self.sphere``, i.e. an ODF.
        """
        return peak_directions(blob, self.sphere, **self._pf_kwargs)[0]

    cpdef cnp.ndarray[cnp.float_t, ndim=2] initial_direction(self,
                                                             double[::1] point):
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
        cdef double* pmf = self._get_pmf(point)
        return self._get_peak_directions(<double[:self.len_pmf]>pmf)

    cdef double* _get_pmf(self, double[::1] point) nogil:
        cdef:
            cnp.npy_intp i
            cnp.npy_intp _len = self.len_pmf
            double* pmf = &self.pmf_gen.pmf[0]
            double pmf_threshold=self.pmf_threshold
            double absolute_pmf_threshold
            double max_pmf=0

        pmf = self.pmf_gen.get_pmf_c(&point[0], pmf)
        for i in range(_len):
            if pmf[i] > max_pmf:
                max_pmf = pmf[i]
        absolute_pmf_threshold = pmf_threshold * max_pmf

        for i in range(_len):
            if pmf[i] < absolute_pmf_threshold:
                pmf[i] = 0.0
        return pmf


cdef class PmfGenDirectionGetter(BasePmfDirectionGetter):
    """A base class for direction getter using a pmf"""

    @classmethod
    def from_pmf(cls, pmf, max_angle, sphere,
                 pmf_threshold=.1, **kwargs):
        """Constructor for making a DirectionGetter from an array of Pmfs

        Parameters
        ----------
        pmf : array, 4d
            The pmf to be used for tracking at each voxel.
        max_angle : float, [0, 90]
            The maximum allowed angle between incoming direction and new
            direction.
        sphere : Sphere
            The set of directions on which the pmf is sampled and to be used
            for tracking.
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
        if pmf.ndim != 4:
            raise ValueError("pmf should be a 4d array.")
        if pmf.shape[3] != len(sphere.theta):
            msg = ("The last dimension of pmf should match the number of "
                   "points in sphere.")
            raise ValueError(msg)

        pmf_gen = SimplePmfGen(np.asarray(pmf,dtype=float), sphere)
        return cls(pmf_gen, max_angle, sphere, pmf_threshold=pmf_threshold, **kwargs)

    @classmethod
    def from_shcoeff(cls, shcoeff, max_angle, sphere=default_sphere,
                    pmf_threshold=0.1, basis_type=None, legacy=True,
                     sh_to_pmf=False, **kwargs):
        """Probabilistic direction getter from a distribution of directions
        on the sphere

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
            ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
        relative_peak_threshold : float in [0., 1.]
            Used for extracting initial tracking directions. Passed to
            peak_directions.
        min_separation_angle : float in [0, 90]
            Used for extracting initial tracking directions. Passed to
            peak_directions.
        legacy: bool, optional
            True to use a legacy basis definition for backward compatibility
            with previous ``tournier07`` and ``descoteaux07`` implementations.
        sh_to_pmf: bool, optional
            If true, map sherical harmonics to spherical function (pmf) before
            tracking (faster, requires more memory).

        See Also
        --------
        dipy.direction.peaks.peak_directions

        """
        if sh_to_pmf:
            sh_order = shm.order_from_ncoef(shcoeff.shape[3])
            pmf = shm.sh_to_sf(shcoeff, sphere, sh_order_max=sh_order,
                               basis_type=basis_type, legacy=legacy)
            pmf[pmf<0] = 0
            pmf_gen = SimplePmfGen(np.asarray(pmf,dtype=float), sphere)
        else:
            pmf_gen = SHCoeffPmfGen(np.asarray(shcoeff,dtype=float), sphere,
                                    basis_type, legacy=legacy)
        return cls(pmf_gen, max_angle, sphere, pmf_threshold=pmf_threshold, **kwargs)


cdef class ClosestPeakDirectionGetter(PmfGenDirectionGetter):
    """A direction getter that returns the closest odf peak to previous tracking
    direction.
    """

    cdef int get_direction_c(self, double[::1] point, double[::1] direction):
        """
        Returns
        -------
        0 : if ``direction`` is updated
        1 : if no new direction is founded
        """
        cdef:
            cnp.npy_intp _len = self.len_pmf
            double* pmf
            cnp.ndarray[cnp.float_t, ndim=2] peaks

        pmf = self._get_pmf(point)

        peaks = self._get_peak_directions(<double[:_len]>pmf)
        if len(peaks) == 0:
            return 1

        return closest_peak(peaks, direction, self.cos_similarity)
