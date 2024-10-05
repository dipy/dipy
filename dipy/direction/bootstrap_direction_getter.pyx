# cython: wraparound=False, cdivision=True, boundscheck=False

cimport numpy as cnp
import numpy as np

from dipy.core.interpolation cimport trilinear_interpolate4d_c
from dipy.data import default_sphere
from dipy.direction.closest_peak_direction_getter cimport closest_peak
from dipy.direction.peaks import peak_directions
from dipy.reconst import shm
from dipy.tracking.direction_getter cimport DirectionGetter


cdef class BootDirectionGetter(DirectionGetter):

    cdef:
        cnp.ndarray dwi_mask
        double[:] vox_data
        dict _pf_kwargs
        double cos_similarity
        double min_separation_angle
        double relative_peak_threshold
        double[:] pmf
        double[:, :] R
        double[:, :, :, :] data
        int max_attempts
        int sh_order
        object H
        object model
        object sphere


    def __init__(self, data, model, max_angle, sphere=default_sphere,
                 max_attempts=5, sh_order=0, b_tol=20, **kwargs):
        cdef:
            cnp.ndarray x, y, z, r
            double[:] theta, phi
            double[:, :] B

        if max_attempts < 1:
             raise ValueError("max_attempts must be greater than 0.")

        if b_tol <= 0:
            raise ValueError("b_tol must be greater than 0.")

        self._pf_kwargs = kwargs
        self.data = np.asarray(data, dtype=float)
        self.model = model
        self.cos_similarity = np.cos(np.deg2rad(max_angle))
        self.sphere = sphere
        self.sh_order = sh_order
        self.max_attempts = max_attempts

        if self.sh_order == 0:
            if hasattr(model, "sh_order"):
                self.sh_order = model.sh_order
            else:
                self.sh_order = 4 #  DEFAULT Value

        self.dwi_mask = model.gtab.b0s_mask == 0
        x, y, z = model.gtab.gradients[self.dwi_mask].T
        r, theta, phi = shm.cart2sphere(x, y, z)
        if r.max() - r.min() >= b_tol:
            raise ValueError("BootDirectionGetter only supports single shell \
                              data.")
        B, _, _ = shm.real_sh_descoteaux(self.sh_order, theta, phi)
        self.H = shm.hat(B)
        self.R = shm.lcr_matrix(self.H)

        self.vox_data = np.empty(self.data.shape[3])
        self.pmf = np.empty(sphere.vertices.shape[0])


    @classmethod
    def from_data(cls, data, model, max_angle, sphere=default_sphere,
                  sh_order=0, max_attempts=5, b_tol=20, **kwargs):
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
        b_tol : float
            Maximum difference between b-values to be considered single shell.
        relative_peak_threshold : float in [0., 1.]
            Relative threshold for excluding ODF peaks.
        min_separation_angle : float in [0, 90]
            Angular threshold for excluding ODF peaks.

        """
        return cls(data, model, max_angle, sphere=sphere, max_attempts=max_attempts, sh_order=sh_order,
                   b_tol=b_tol, **kwargs)


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
        cdef:
            double[:] pmf = self.get_pmf_no_boot(point)

        return peak_directions(pmf, self.sphere, **self._pf_kwargs)[0]


    cpdef double[:] get_pmf(self, double[::1] point):
        """Produces an ODF from a SH bootstrap sample"""
        if trilinear_interpolate4d_c(self.data,
                                     &point[0],
                                     &self.vox_data[0]) != 0:
            self.__clear_pmf()
        else:
            np.asarray(self.vox_data)[self.dwi_mask] = shm.bootstrap_data_voxel(
                np.asarray(self.vox_data)[self.dwi_mask], self.H, self.R)
            self.pmf = self.model.fit(np.asarray(self.vox_data)).odf(self.sphere)
        return self.pmf


    cpdef double[:] get_pmf_no_boot(self, double[::1] point):

        if trilinear_interpolate4d_c(self.data,
                                     &point[0],
                                     &self.vox_data[0]) != 0:
            self.__clear_pmf()
        else:
            self.pmf = self.model.fit(np.asarray(self.vox_data)).odf(self.sphere)
        return self.pmf

    cdef void __clear_pmf(self) nogil:
        cdef:
            cnp.npy_intp len_pmf = self.pmf.shape[0]
            cnp.npy_intp i

        for i in range(len_pmf):
            self.pmf[i] = 0.0


    cdef int get_direction_c(self, double[::1] point, double[::1] direction):
        """Attempt direction getting on a few bootstrap samples.

        Returns
        -------
        status : int
            Returns 0 `direction` was updated with a new tracking direction, or
            1 otherwise.
        """
        cdef:
            double[:] pmf
            cnp.ndarray[cnp.float_t, ndim=2] peaks

        for _ in range(self.max_attempts):
            pmf = self.get_pmf(point)
            peaks = peak_directions(pmf, self.sphere, **self._pf_kwargs)[0]
            if len(peaks) > 0:
                return closest_peak(peaks, direction, self.cos_similarity)
        return 1
