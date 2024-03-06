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
        self.data = np.asarray(data, dtype=float, order='C')
        self.sphere = sphere
        self.vertices = np.asarray(sphere.vertices, dtype=float)

    def get_pmf(self, double[::1] point):
        return self.get_pmf_c(point)

    cdef double[:] get_pmf_c(self, double[::1] point) noexcept nogil:
        pass

    cdef int find_closest(self, double* xyz) noexcept nogil:
        cdef:
            cnp.npy_intp idx = 0
            cnp.npy_intp i
            cnp.npy_intp len_pmf = self.pmf.shape[0]
            double cos_max = 0
            double cos_sim

        for i in range(len_pmf):
            cos_sim = self.vertices[i][0] * xyz[0] \
                    + self.vertices[i][1] * xyz[1] \
                    + self.vertices[i][2] * xyz[2]
            if cos_sim < 0:
                cos_sim = cos_sim * -1
            if cos_sim > cos_max:
                cos_max = cos_sim
                idx = i
        return idx

    def get_pmf_value(self, double[::1] point, double[::1] xyz):
        return self.get_pmf_value_c(point, xyz)

    cdef double get_pmf_value_c(self, double[::1] point, double[::1] xyz) noexcept nogil:
        """
        Return the pmf value corresponding to the closest vertex to the
        direction xyz.
        """
        cdef int idx = self.find_closest(&xyz[0])
        return self.get_pmf_c(point)[idx]

    cdef void __clear_pmf(self) noexcept nogil:
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

    cdef double[:] get_pmf_c(self, double[::1] point) noexcept nogil:
        if trilinear_interpolate4d_c(self.data, &point[0], &self.pmf[0]) != 0:
            PmfGen.__clear_pmf(self)
        return self.pmf

    cdef double get_pmf_value_c(self, double[::1] point, double[::1] xyz) noexcept nogil:
        """
        Return the pmf value corresponding to the closest vertex to the
        direction xyz.
        """
        cdef:
            int idx

        idx = self.find_closest(&xyz[0])

        if trilinear_interpolate4d_c(self.data[:,:,:,idx:idx+1],
                                     &point[0],
                                     &self.pmf[0]) != 0:
            PmfGen.__clear_pmf(self)
        return self.pmf[0]


cdef class SHCoeffPmfGen(PmfGen):

    def __init__(self,
                 double[:, :, :, :] shcoeff_array,
                 object sphere,
                 object basis_type,
                 legacy=True):
        cdef:
            int sh_order

        PmfGen.__init__(self, shcoeff_array, sphere)

        sh_order = shm.order_from_ncoef(shcoeff_array.shape[3])
        try:
            basis = shm.sph_harm_lookup[basis_type]
        except KeyError:
            raise ValueError("%s is not a known basis type." % basis_type)
        self.B, _, _ = basis(sh_order, sphere.theta, sphere.phi, legacy=legacy)
        self.coeff = np.empty(shcoeff_array.shape[3])
        self.pmf = np.empty(self.B.shape[0])

    cdef double[:] get_pmf_c(self, double[::1] point) noexcept nogil:
        cdef:
            cnp.npy_intp i, j
            cnp.npy_intp len_pmf = self.pmf.shape[0]
            cnp.npy_intp len_B = self.B.shape[1]
            double _sum

        if trilinear_interpolate4d_c(self.data,
                                     &point[0],
                                     &self.coeff[0]) != 0:
            PmfGen.__clear_pmf(self)
        else:
            for i in range(len_pmf):
                _sum = 0
                for j in range(len_B):
                    _sum = _sum + (self.B[i, j] * self.coeff[j])
                self.pmf[i] = _sum
        return self.pmf
