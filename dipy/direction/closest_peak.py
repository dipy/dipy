import numpy as np
from dipy.direction.peaks import peak_directions, default_sphere
from dipy.reconst.shm import order_from_ncoef, sph_harm_lookup
from dipy.tracking.local.direction_getter import DirectionGetter
from dipy.tracking.local.interpolation import trilinear_interpolate4d


def _closest_peak(peak_directions, prev_step, cos_similarity):
    """Return the closest direction to prev_step from peak_directions.

    All directions should be unit vectors. Antipodal symmetry is assumed, ie
    direction x is the same as -x.

    Parameters
    ----------
    peak_directions : array (N, 3)
        N unit vectors.
    prev_step : array (3,) or None
        Previous direction.
    cos_similarity : float
        `cos(max_angle)` where `max_angle` is the maximum allowed angle between
        prev_step and the returned direction.

    Returns
    -------
    direction : array or None
        If prev_step is None, returns peak_directions. Otherwise returns the
        closest direction to prev_step. If no directions are close enough to
        prev_step, returns None
    """
    peak_dots = np.dot(peak_directions, prev_step)
    closest_peak = abs(peak_dots).argmax()
    dot_closest_peak = peak_dots[closest_peak]
    if dot_closest_peak >= cos_similarity:
        return peak_directions[closest_peak]
    elif dot_closest_peak <= -cos_similarity:
        return -peak_directions[closest_peak]
    else:
        return None


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


class BaseDirectionGetter(DirectionGetter):
    """A base class for dynamic direction getters"""

    def __init__(self, pmf_gen, max_angle, sphere, pmf_threshold=.1, **kwargs):
        self.sphere = sphere
        self._pf_kwargs = kwargs
        self.pmf_gen = pmf_gen
        self.pmf_threshold = pmf_threshold
        self.cos_similarity = np.cos(np.deg2rad(max_angle))

    def _peak_directions(self, blob):
        """Gets directions using parameters provided at init.

        Blob can be any function defined on ``self.sphere``, ie an ODF, PMF,
        FOD.
        """
        return peak_directions(blob, self.sphere, **self._pf_kwargs)[0]

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
        pmf = self.pmf_gen.get_pmf(point)
        pmf.clip(min=self.pmf_threshold, out=pmf) 
        peaks = self._peak_directions(pmf)
        if len(peaks) == 0:
            return 1
        new_dir = _closest_peak(peaks, direction, self.cos_similarity)
        if new_dir is None:
            return 1
        else:
            direction[:] = new_dir
            return 0


class ClosestPeakDirectionGetter(BaseDirectionGetter):
    """A direction getter that returns the closest odf peak to previous tracking
    direction.

    """
    @classmethod
    def from_pmf(klass, pmf, max_angle, sphere=default_sphere, pmf_threshold=0.1,
                 **kwargs):
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
    def from_shcoeff(klass, shcoeff, max_angle, sphere=default_sphere, pmf_threshold=0.1,
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

