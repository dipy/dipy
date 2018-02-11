# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

from warnings import warn

import numpy as np
cimport numpy as np

from dipy.core.geometry import cart2sphere
from dipy.reconst import shm

from dipy.tracking.local.interpolation cimport trilinear_interpolate4d_c


cdef class PmfGen:

    cpdef double[:] get_pmf(self, double[::1] point):
        return self.get_pmf_c(&point[0])

    cdef double[:] get_pmf_c(self, double* point):
        pass


cdef class SimplePmfGen(PmfGen):

    def __init__(self, double[:, :, :, :] pmf_array):
        PmfGen.__init__(self)

        if np.min(pmf_array) < 0:
            raise ValueError("pmf should not have negative values")
        self.pmf_array = pmf_array

        self.out = np.empty(pmf_array.shape[3])

    cdef double[:] get_pmf_c(self, double* point):
        if trilinear_interpolate4d_c(self.pmf_array, point, self.out) == 0:
            return self.out
        return None


cdef class SHCoeffPmfGen(PmfGen):

    def __init__(self,
                 double[:, :, :, :] shcoeff,
                 object sphere,
                 object basis_type):
        cdef:
            int sh_order

        PmfGen.__init__(self)

        self.shcoeff = shcoeff
        self.sphere = sphere
        sh_order = shm.order_from_ncoef(shcoeff.shape[3])
        try:
            basis = shm.sph_harm_lookup[basis_type]
        except KeyError:
            raise ValueError("%s is not a known basis type." % basis_type)
        self.B, m, n = basis(sh_order, sphere.theta, sphere.phi)
        self.coeff = np.empty(self.shcoeff.shape[3])
        self.pmf = np.empty(self.B.shape[0])

    cdef double[:] get_pmf_c(self, double* point):
        cdef:
            size_t i, j
            size_t len_pmf = self.pmf.shape[0]
            size_t len_B = self.B.shape[1]
            double _sum

        if trilinear_interpolate4d_c(self.shcoeff, point, self.coeff) == 0:
            for i in range(len_pmf):
                _sum = 0
                for j in range(len_B):
                    _sum += self.B[i, j] * self.coeff[j]
                self.pmf[i] = _sum
                if self.pmf[i] < 0.0:
                    self.pmf[i] = 0.0
            return self.pmf
        return None


cdef class BootPmfGen(SHCoeffPmfGen):

    def __init__(self,
                 np.ndarray data,
                 object model,
                 object sphere,
                 int sh_order=0,
                 double tol=1e-2):
        cdef:
            double b_range
            np.ndarray x, y, z, r
            double[:] theta, phi
            double[:, :] B

        self.sh_order = sh_order
        if self.sh_order == 0:
            if hasattr(model, "sh_order"):
                self.sh_order = model.sh_order
            else:
                self.sh_order = 4 #  DEFAULT Value

        x, y, z = model.gtab.gradients[model.gtab.b0s_mask == 0].T
        r, theta, phi = shm.cart2sphere(x, y, z)
        b_range = (r.max() - r.min()) / r.min()
        if b_range > tol:
            raise ValueError("BootPmfGen only supports single shell data")
        B, m, n = shm.real_sym_sh_basis(self.sh_order, theta, phi)

        self.nbr_b0s = np.sum(model.gtab.b0s_mask == 1)
        self.nbr_dwi = np.sum(model.gtab.b0s_mask == 0)
        self.nbr_data = data.shape[3]
        self.vox_b0s = np.empty(self.nbr_b0s)
        self.vox_dwi = np.empty(self.nbr_dwi)
        self.vox_data = np.empty(self.nbr_data)
        self.model = model
        self.sphere = sphere
        self.H = shm.hat(B)
        self.R = shm.lcr_matrix(self.H)
        self.pmf = np.empty(self.B.shape[0])
        self.b0s = np.asarray(data[:,:,:,model.gtab.b0s_mask == 1], "float64")
        self.dwi = np.asarray(data[:,:,:,model.gtab.b0s_mask == 0], "float64")


    cdef double[:] get_pmf_c(self, double* point):
        """Produces an ODF from a SH bootstrap sample"""
        if (trilinear_interpolate4d_c(self.dwi, point, self.vox_dwi) == 0 and
                trilinear_interpolate4d_c(self.b0s, point, self.vox_b0s) == 0):
            self.vox_dwi = shm.bootstrap_data_voxel(self.vox_dwi, self.H,
                                                    self.R)
            self._set_vox_data()
            self.pmf = self.model.fit(self.vox_data).odf(self.sphere)
            return self.pmf
        return None


    cpdef double[:] get_pmf_no_boot(self, double[::1] point):
        return self.get_pmf_no_boot_c(&point[0])


    cdef double[:] get_pmf_no_boot_c(self, double* point):
        if (trilinear_interpolate4d_c(self.dwi, point, self.vox_dwi) == 0 and
                trilinear_interpolate4d_c(self.b0s, point, self.vox_b0s) == 0):
            self._set_vox_data()
            self.pmf = self.model.fit(self.vox_data).odf(self.sphere)
            return self.pmf
        return None


    cdef int _set_vox_data(self):
        cdef:
            size_t i, j
        j = 0
        for i in range(self.nbr_data):
            if self.model.gtab.b0s_mask[i] == 1:
                self.vox_data[i] = self.vox_b0s[j]
                j = j + 1
            else:
                self.vox_data[i] = self.vox_dwi[i-j]
        return 0
