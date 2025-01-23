# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp

from dipy.reconst import shm

from dipy.core.interpolation cimport trilinear_interpolate4d_c
from libc.stdlib cimport malloc, free

cdef extern from "stdlib.h" nogil:
    void *memset(void *ptr, int value, size_t num)


cdef class PmfGen:

    def __init__(self,
                 double[:, :, :, :] data,
                 object sphere):
        self.data = np.asarray(data, dtype=float, order='C')
        self.vertices = np.asarray(sphere.vertices, dtype=float)
        self.pmf = np.zeros(self.vertices.shape[0])
        self.sphere = sphere

    def get_pmf(self, double[::1] point, double[:] out=None):
        if out is None:
            out = self.pmf
        return <double[:len(self.vertices)]>self.get_pmf_c(&point[0], &out[0])

    def get_sphere(self):
        return self.sphere

    cdef double* get_pmf_c(self, double* point, double* out) noexcept nogil:
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
        return self.get_pmf_value_c(&point[0], &xyz[0])

    cdef double get_pmf_value_c(self,
                                double* point,
                                double* xyz) noexcept nogil:
        pass


cdef class SimplePmfGen(PmfGen):

    def __init__(self,
                 double[:, :, :, :] pmf_array,
                 object sphere):
        PmfGen.__init__(self, pmf_array, sphere)
        if not pmf_array.shape[3] == sphere.vertices.shape[0]:
            raise ValueError("pmf should have the same number of values as the"
                             + " number of vertices of sphere.")

    cdef double* get_pmf_c(self, double* point, double* out) noexcept nogil:
        if trilinear_interpolate4d_c(self.data, point, out) != 0:
            memset(out, 0, self.pmf.shape[0] * sizeof(double))
        return out

    cdef double get_pmf_value_c(self,
                                double* point,
                                double* xyz) noexcept nogil:
        """
        Return the pmf value corresponding to the closest vertex to the
        direction xyz.
        """
        cdef:
            int idx
            double pmf_value = 0

        idx = self.find_closest(xyz)
        trilinear_interpolate4d_c(self.data[:,:,:,idx:idx+1],
                                  point,
                                  &pmf_value)
        return pmf_value


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

    cdef double* get_pmf_c(self, double* point, double* out) noexcept nogil:
        cdef:
            cnp.npy_intp i, j
            cnp.npy_intp len_pmf = self.pmf.shape[0]
            cnp.npy_intp len_B = self.B.shape[1]
            double _sum
            double *coeff = <double*> malloc(len_B * sizeof(double))

        if trilinear_interpolate4d_c(self.data, point, coeff) != 0:
            memset(out, 0, len_pmf * sizeof(double))
        else:
            for i in range(len_pmf):
                _sum = 0
                for j in range(len_B):
                    _sum = _sum + (self.B[i, j] * coeff[j])
                out[i] = _sum
        free(coeff)
        return out

    cdef double get_pmf_value_c(self,
                                double* point,
                                double* xyz) noexcept nogil:
        """
        Return the pmf value corresponding to the closest vertex to the
        direction xyz.
        """
        cdef:
            int idx = self.find_closest(xyz)
            cnp.npy_intp j
            cnp.npy_intp len_B = self.B.shape[1]
            double *coeff = <double*> malloc(len_B * sizeof(double))
            double pmf_value = 0

        if trilinear_interpolate4d_c(self.data, point, coeff) == 0:
            for j in range(len_B):
                pmf_value = pmf_value + (self.B[idx, j] * coeff[j])

        free(coeff)
        return pmf_value

