import numpy as np
cimport numpy as cnp
cimport cython

from warnings import warn

from dipy.direction.peaks import peak_directions, default_sphere
from dipy.direction.pmf cimport SimplePmfGen, SHCoeffPmfGen
from dipy.reconst.shm import order_from_ncoef, sph_harm_lookup
from dipy.tracking.direction_getter cimport DirectionGetter
from dipy.tracking.stopping_criterion cimport (StreamlineStatus,
                                               StoppingCriterion,
                                               TRACKPOINT,
                                               ENDPOINT,
                                               OUTSIDEIMAGE,
                                               INVALIDPOINT,
                                               PYERROR)
from dipy.utils.fast_numpy cimport copy_point, scalar_muliplication_point


cdef int closest_peak(cnp.ndarray[cnp.float_t, ndim=2] peak_dirs,
                      double* direction, double cos_similarity):
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
            copy_point(&peak_dirs[closest_peak_i, 0], direction)
            return 0
        if closest_peak_dot <= -cos_similarity:
            copy_point(&peak_dirs[closest_peak_i, 0], direction)
            scalar_muliplication_point(direction, -1)
            return 0
    return 1

cdef extern from "dpy_math.h" nogil:
    int dpy_signbit(double x)
    double dpy_rint(double x)
    double fabs(double)

@cython.cdivision(True)
cdef inline double stepsize(double point, double increment) nogil:
    """Compute the step size to the closest boundary in units of increment."""
    cdef:
        double dist
    dist = dpy_rint(point) + .5 - dpy_signbit(increment) - point
    if dist == 0:
        # Point is on an edge, return step size to next edge.  This is most
        # likely to come up if overstep is set to 0.
        return 1. / fabs(increment)
    else:
        return dist / increment

cdef void step_to_boundary(double * point, double * direction,
                           double overstep) nogil:
    """Takes a step from point in along direction just past a voxel boundary.

    Parameters
    ----------
    direction : c-pointer to double[3]
        The direction along which the step should be taken.
    point : c-pointer to double[3]
        The tracking point which will be updated by this function.
    overstep : double
        It's often useful to have the points of a streamline lie inside of a
        voxel instead of having them lie on the boundary. For this reason,
        each step will overshoot the boundary by ``overstep * direction``.
        This should not be negative.

    """
    cdef:
        double step_sizes[3]
        double smallest_step

    for i in range(3):
        step_sizes[i] = stepsize(point[i], direction[i])

    smallest_step = step_sizes[0]
    for i in range(1, 3):
        if step_sizes[i] < smallest_step:
            smallest_step = step_sizes[i]

    smallest_step += overstep
    for i in range(3):
        point[i] += smallest_step * direction[i]

cdef void fixed_step(double * point, double * direction, double step_size) nogil:
    """Updates point by stepping in direction.

    Parameters
    ----------
    direction : c-pointer to double[3]
        The direction along which the step should be taken.
    point : c-pointer to double[3]
        The tracking point which will be updated by this function.
    step_size : double
        The size of step in units of direction.

    """
    for i in range(3):
        point[i] += direction[i] * step_size


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
        cdef double[:] pmf = self._get_pmf(&point[0])
        return self._get_peak_directions(pmf)

    cdef _get_pmf(self, double* point):
        cdef:
            cnp.npy_intp _len, i
            double[:] pmf
            double absolute_pmf_threshold

        pmf = self.pmf_gen.get_pmf_c(point)
        _len = pmf.shape[0]

        absolute_pmf_threshold = self.pmf_threshold*np.max(pmf)
        for i in range(_len):
            if pmf[i] < absolute_pmf_threshold:
                pmf[i] = 0.0
        return pmf


cdef class PmfGenDirectionGetter(BasePmfDirectionGetter):
    """A base class for direction getter using a pmf"""

    @classmethod
    def from_pmf(klass, pmf, max_angle, sphere=default_sphere,
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
        if pmf.ndim != 4:
            raise ValueError("pmf should be a 4d array.")
        if pmf.shape[3] != len(sphere.theta):
            msg = ("The last dimension of pmf should match the number of "
                   "points in sphere.")
            raise ValueError(msg)

        pmf_gen = SimplePmfGen(np.asarray(pmf,dtype=float))
        return klass(pmf_gen, max_angle, sphere, pmf_threshold, **kwargs)

    @classmethod
    def from_shcoeff(klass, shcoeff, max_angle, sphere=default_sphere,
                     pmf_threshold=0.1, basis_type=None, **kwargs):
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

        See also
        --------
        dipy.direction.peaks.peak_directions

        """
        pmf_gen = SHCoeffPmfGen(np.asarray(shcoeff,dtype=float), sphere,
                                basis_type)
        return klass(pmf_gen, max_angle, sphere, pmf_threshold, **kwargs)


cdef class ClosestPeakDirectionGetter(PmfGenDirectionGetter):
    """A direction getter that returns the closest odf peak to previous tracking
    direction.
    """

    cdef int get_direction_c(self, double* point, double* direction):
        """
        Returns
        -------
        0 : if ``direction`` is updated
        1 : if no new direction is founded
        """
        cdef:
            double[:] pmf
            cnp.ndarray[cnp.float_t, ndim=2] peaks

        pmf = self._get_pmf(point)

        peaks = self._get_peak_directions(pmf)
        if len(peaks) == 0:
            return 1
        return closest_peak(peaks, direction, self.cos_similarity)
