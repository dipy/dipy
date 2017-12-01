# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

import numpy as np

from dipy.reconst.shm import order_from_ncoef, sph_harm_lookup
from dipy.tracking.local.interpolation cimport trilinear_interpolate4d_c


cdef class PmfGen:

    cpdef double[:] get_pmf(self, double[::1] point):
        return self.get_pmf_c(&point[0])

    cdef double[:] get_pmf_c(self, double* point) nogil:
        pass


cdef class SimplePmfGen(PmfGen):

    def __init__(self, double[:, :, :, :] pmf_array):
        PmfGen.__init__(self)

        if np.min(pmf_array) < 0:
            raise ValueError("pmf should not have negative values")
        self.pmf_array = pmf_array

        self.out = np.empty(pmf_array.shape[3])

    cdef double[:] get_pmf_c(self, double* point) nogil:
        trilinear_interpolate4d_c(self.pmf_array, point, self.out)
        return self.out


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
        sh_order = order_from_ncoef(shcoeff.shape[3])
        try:
            basis = sph_harm_lookup[basis_type]
        except KeyError:
            raise ValueError("%s is not a known basis type." % basis_type)
        self.B, m, n = basis(sh_order, sphere.theta, sphere.phi)
        self.coeff = np.empty(self.shcoeff.shape[3])
        self.pmf = np.empty(self.B.shape[0])

    cdef double[:] get_pmf_c(self, double* point) nogil:
        cdef:
            size_t i, j
            size_t len_pmf = self.pmf.shape[0]
            size_t len_B = self.B.shape[1]
            double _sum

        trilinear_interpolate4d_c(self.shcoeff, point, self.coeff)
        for i in range(len_pmf):
            _sum = 0
            for j in range(len_B):
                _sum += self.B[i, j] * self.coeff[j]
            self.pmf[i] = _sum
            if self.pmf[i] < 0.0:
                self.pmf[i] = 0.0

        return self.pmf
