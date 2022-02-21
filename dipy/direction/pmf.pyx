# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp

from dipy.reconst import shm

from dipy.core.interpolation cimport trilinear_interpolate4d_c


cdef class PmfGen:

    def __init__(self,
                 double[:, :, :, :] data,
                 object sphere):
        self.data = np.asarray(data, dtype=float)
        self.sphere = sphere

    cpdef double[:] get_pmf(self, double[::1] point):
        pass

    cpdef double get_pmf_value(self, double[::1] point, double[::1] xyz):
        """
        Return the pmf value corresponding to the closest vertex to the
        direction xyz.
        """
        cdef int idx = self.sphere.find_closest(xyz)
        return self.get_pmf(point)[idx]

    cdef void __clear_pmf(self):
        cdef:
            cnp.npy_intp len_pmf = self.pmf.shape[0]
            cnp.npy_intp i

        for i in range(len_pmf):
            self.pmf[i] = 0.0


cdef class SimplePmfGen(PmfGen):

    def __init__(self,
                 double[:, :, :, :] pmf_array,
                 object sphere):
        PmfGen.__init__(self, pmf_array, sphere)
        self.pmf = np.empty(pmf_array.shape[3])
        if np.min(pmf_array) < 0:
            raise ValueError("pmf should not have negative values.")
        if not pmf_array.shape[3] == sphere.vertices.shape[0]:
            raise ValueError("pmf should have the same number of values as the"
                             + " number of vertices of sphere.")

    cpdef double[:] get_pmf(self, double[::1] point):
        if trilinear_interpolate4d_c(self.data, &point[0], self.pmf) != 0:
            PmfGen.__clear_pmf(self)
        return self.pmf


cdef class SHCoeffPmfGen(PmfGen):

    def __init__(self,
                 double[:, :, :, :] shcoeff_array,
                 object sphere,
                 object basis_type):
        cdef:
            int sh_order

        PmfGen.__init__(self, shcoeff_array, sphere)

        sh_order = shm.order_from_ncoef(shcoeff_array.shape[3])
        try:
            basis = shm.sph_harm_lookup[basis_type]
        except KeyError:
            raise ValueError("%s is not a known basis type." % basis_type)
        self.B, _, _ = basis(sh_order, sphere.theta, sphere.phi)
        self.coeff = np.empty(shcoeff_array.shape[3])
        self.pmf = np.empty(self.B.shape[0])

    cpdef double[:] get_pmf(self, double[::1] point):
        cdef:
            cnp.npy_intp i, j
            cnp.npy_intp len_pmf = self.pmf.shape[0]
            cnp.npy_intp len_B = self.B.shape[1]
            double _sum

        if trilinear_interpolate4d_c(self.data, &point[0], self.coeff) != 0:
            PmfGen.__clear_pmf(self)
        else:
            for i in range(len_pmf):
                _sum = 0
                for j in range(len_B):
                    _sum += self.B[i, j] * self.coeff[j]
                self.pmf[i] = _sum
        return self.pmf


cdef class BootPmfGen(PmfGen):

    def __init__(self,
                 double[:, :, :, :] dwi_array,
                 object model,
                 object sphere,
                 int sh_order=0,
                 double tol=1e-2):
        cdef:
            double b_range
            np.ndarray x, y, z, r
            double[:] theta, phi
            double[:, :] B

        PmfGen.__init__(self, dwi_array, sphere)
        self.sh_order = sh_order
        if self.sh_order == 0:
            if hasattr(model, "sh_order"):
                self.sh_order = model.sh_order
            else:
                self.sh_order = 4 #  DEFAULT Value

        self.dwi_mask = model.gtab.b0s_mask == 0
        x, y, z = model.gtab.gradients[self.dwi_mask].T
        r, theta, phi = shm.cart2sphere(x, y, z)
        b_range = (r.max() - r.min()) / r.min()
        if b_range > tol:
            raise ValueError("BootPmfGen only supports single shell data.")
        B, _, _ = shm.real_sh_descoteaux(self.sh_order, theta, phi)
        self.H = shm.hat(B)
        self.R = shm.lcr_matrix(self.H)
        self.vox_data = np.empty(dwi_array.shape[3])

        self.model = model
        self.pmf = np.empty(len(sphere.theta))

    cpdef double[:] get_pmf(self, double[::1] point):
        """Produces an ODF from a SH bootstrap sample"""
        if trilinear_interpolate4d_c(self.data, &point[0], self.vox_data) != 0:
            self.__clear_pmf()
        else:
            self.vox_data[self.dwi_mask] = shm.bootstrap_data_voxel(
                self.vox_data[self.dwi_mask], self.H, self.R)
            self.pmf = self.model.fit(self.vox_data).odf(self.sphere)
        return self.pmf

    cpdef double[:] get_pmf_no_boot(self, double[::1] point):
        if trilinear_interpolate4d_c(self.data, &point[0], self.vox_data) != 0:
            self.__clear_pmf()
        else:
            self.pmf = self.model.fit(self.vox_data).odf(self.sphere)
        return self.pmf
