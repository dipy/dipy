cimport numpy as cnp
import numpy as np

from dipy.data import default_sphere
from dipy.direction.closest_peak_direction_getter cimport (closest_peak,
                                                           BasePmfDirectionGetter)
from dipy.direction.pmf import BootPmfGen


cdef class BootDirectionGetter(BasePmfDirectionGetter):

    cdef:
        int max_attempts

    def __init__(self, pmfgen, maxangle, sphere=default_sphere,
                 max_attempts=5, **kwargs):
        if max_attempts < 1:
             raise ValueError("max_attempts must be greater than 0.")
        self.max_attempts = max_attempts
        BasePmfDirectionGetter.__init__(self, pmfgen, maxangle, sphere, **kwargs)

    @classmethod
    def from_data(klass, data, model, max_angle, sphere=default_sphere,
                  sh_order=0, max_attempts=5, **kwargs):
        """Create a BootDirectionGetter using HARDI data and an ODF type model

        Parameters
        ----------
        data : ndarray, float, (..., N)
            Diffusion MRI data with N volumes.
        model : dipy diffusion model
            Must provide fit with odf method.
        max_angle : float (0, 90)
            Maximum angle between tract segments. This angle can be more
            generous (larger) than values typically used with probabilistic
            direction getters.
        sphere : Sphere
            The sphere used to sample the diffusion ODF.
        sh_order : even int
            The order of the SH "model" used to estimate bootstrap residuals.
        max_attempts : int
            Max number of bootstrap samples used to find tracking direction
            before giving up.
        pmf_threshold : float
            Threshold for ODF functions.
        relative_peak_threshold : float in [0., 1.]
            Relative threshold for excluding ODF peaks.
        min_separation_angle : float in [0, 90]
            Angular threshold for excluding ODF peaks.

        """
        boot_gen = BootPmfGen(np.asarray(data, dtype=float), model, sphere,
                              sh_order=sh_order)
        return klass(boot_gen, max_angle, sphere, max_attempts, **kwargs)

    cdef int get_direction_c(self, double* point, double* direction):
        """Attempt direction getting on a few bootstrap samples.

        Returns
        -------
        status : int
            Returns 0 `direction` was updated with a new tracking direction, or
            1 otherwise.
        """
        cdef:
            double[:] pmf,
            cnp.ndarray[cnp.float_t, ndim=2] peaks

        for _ in range(self.max_attempts):
            pmf = self._get_pmf(point)
            peaks = self._get_peak_directions(pmf)
            if len(peaks) > 0:
                return closest_peak(peaks, direction, self.cos_similarity)
        return 1
