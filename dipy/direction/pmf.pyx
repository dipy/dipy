# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp

from dipy.core.sphere import Sphere
from dipy.reconst import shm

from dipy.core.interpolation cimport (
    _trilinear_interpolation_iso,
    offset,
    trilinear_interpolate4d_c,
)
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

    cdef cnp.npy_intp get_peaks_c(self,
                                  double* point,
                                  double* out_values,
                                  double* out_indices,
                                  double* out_weights,
                                  cnp.npy_intp max_peaks,
                                  cnp.npy_intp* out_valid) noexcept nogil:
        return 0


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
            raise ValueError(f"{basis_type} is not a known basis type.")
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


cdef class SimplePeakGen(PmfGen):
    """PmfGen subclass for sphere-based peak data (EUDX-style tracking).

    This class stores peak indices and values for EUDX-style tracking,
    providing the PmfGen interface required by the tractogram generator.

    Parameters
    ----------
    peak_indices : ndarray, shape (X, Y, Z, npeaks)
        Indices into odf_vertices for each peak at each voxel.
    peak_values : ndarray, shape (X, Y, Z, npeaks)
        Peak strength values (QA, GFA, etc.) at each voxel.
    odf_vertices : ndarray, shape (N_vertices, 3)
        Sphere vertices representing possible peak directions.
    sphere : Sphere
        Sphere object (used for interface compatibility).

    Notes
    -----
    This class enables EUDX tracking to use the parallel generic_tracking
    infrastructure while maintaining backward compatibility with sphere-based
    peak representation.
    """

    def __init__(self,
                 double[:, :, :, :] peak_indices,
                 double[:, :, :, :] peak_values,
                 double[:, :] odf_vertices,
                 object sphere):
        """Initialize SimplePeakGen with peak data.

        Parameters
        ----------
        peak_indices : memoryview, shape (X, Y, Z, npeaks)
            Indices into odf_vertices.
        peak_values : memoryview, shape (X, Y, Z, npeaks)
            Peak strength values.
        odf_vertices : memoryview, shape (N_vertices, 3)
            Sphere vertices.
        sphere : Sphere
            Sphere object.
        """
        cdef int i
        cdef cnp.ndarray dummy_data
        cdef cnp.ndarray odf_vertices_arr
        cdef cnp.ndarray sphere_vertices_arr

        if (peak_indices.shape[0] != peak_values.shape[0] or
            peak_indices.shape[1] != peak_values.shape[1] or
            peak_indices.shape[2] != peak_values.shape[2]):
            raise ValueError(
                "peak_indices and peak_values must have matching spatial dimensions"
            )
        if peak_indices.shape[3] != peak_values.shape[3]:
            raise ValueError(
                "peak_indices and peak_values must have same number of peaks"
            )

        dummy_data = np.zeros((1, 1, 1, odf_vertices.shape[0]))
        odf_vertices_arr = np.asarray(odf_vertices, dtype=float, order="C")
        PmfGen.__init__(self, dummy_data, sphere)
        self.vertices = odf_vertices_arr
        self.pmf = np.zeros(self.vertices.shape[0])
        sphere_vertices_arr = np.asarray(sphere.vertices, dtype=float, order="C")
        if np.array_equal(sphere_vertices_arr, odf_vertices_arr):
            self.sphere = sphere
        else:
            self.sphere = Sphere(xyz=odf_vertices_arr)

        self.peak_indices = peak_indices
        self.peak_values = peak_values
        self.odf_vertices = odf_vertices
        self.max_peaks = peak_indices.shape[3]

        self.peak_indices_ptr = &peak_indices[0, 0, 0, 0]
        self.peak_values_ptr = &peak_values[0, 0, 0, 0]
        # Kept for Cython ABI compatibility across extension rebuild boundaries.
        self.odf_vertices_ptr = &odf_vertices[0, 0]

        self.peak_shape[0] = peak_indices.shape[0]
        self.peak_shape[1] = peak_indices.shape[1]
        self.peak_shape[2] = peak_indices.shape[2]
        self.peak_shape[3] = self.max_peaks

        cdef cnp.ndarray indices_arr = np.asarray(peak_indices)
        cdef cnp.npy_intp[:] arr_strides = <cnp.npy_intp[:4]>(<cnp.npy_intp*>indices_arr.strides)
        self.peak_strides[0] = arr_strides[0]
        self.peak_strides[1] = arr_strides[1]
        self.peak_strides[2] = arr_strides[2]
        self.peak_strides[3] = arr_strides[3]

    cdef void _compute_trilinear(self,
                                 double* point,
                                 double* out_weights,
                                 cnp.npy_intp* out_index) noexcept nogil:
        """Compute trilinear weights and neighboring indices."""
        _trilinear_interpolation_iso(point, out_weights, out_index)

    cdef cnp.npy_intp _inside_global_bounds(self,
                                            cnp.npy_intp* index) noexcept nogil:
        """Return 1 when trilinear neighborhood is fully inside data bounds."""
        cdef cnp.npy_intp i
        for i in range(3):
            if index[i] < 0 or index[7 * 3 + i] >= self.peak_shape[i]:
                return 0
        return 1

    cdef double* get_pmf_c(self, double* point, double* out) noexcept nogil:
        """Get PMF at a point.

        Parameters
        ----------
        point : double*, shape (3,)
            Query position in voxel coordinates.
        out : double*, shape (N_vertices,)
            Output PMF values.

        Returns
        -------
        out : double*
            Pointer to output array.
        """
        cdef:
            cnp.npy_intp i, j, m
            cnp.npy_intp off
            cnp.npy_intp peak_index
            cnp.npy_intp len_pmf = self.pmf.shape[0]
            cnp.npy_intp index[24]
            cnp.npy_intp xyz[4]
            double peak_value
            double weights[8]

        memset(out, 0, len_pmf * sizeof(double))
        self._compute_trilinear(point, weights, index)
        if self._inside_global_bounds(index) == 0:
            return out

        for m in range(8):
            for i in range(3):
                xyz[i] = index[m * 3 + i]

            for j in range(self.max_peaks):
                xyz[3] = j
                off = offset(xyz, <cnp.npy_intp*>self.peak_strides, 4, 8)
                peak_value = self.peak_values_ptr[off]
                if peak_value <= 0:
                    continue

                peak_index = <cnp.npy_intp>self.peak_indices_ptr[off]
                if peak_index < 0 or peak_index >= len_pmf:
                    continue

                out[peak_index] += weights[m] * peak_value
        return out

    cdef double get_pmf_value_c(self, double* point, double* xyz) noexcept nogil:
        """Get PMF value in a direction.

        Returns
        -------
        value : double
            PMF value.
        """
        cdef:
            cnp.npy_intp closest_idx = self.find_closest(xyz)
            cnp.npy_intp i, j, m
            cnp.npy_intp off
            cnp.npy_intp peak_index
            cnp.npy_intp index[24]
            cnp.npy_intp xyz_idx[4]
            double peak_value
            double weights[8]
            double pmf_value = 0

        self._compute_trilinear(point, weights, index)
        if self._inside_global_bounds(index) == 0:
            return 0.0

        for m in range(8):
            for i in range(3):
                xyz_idx[i] = index[m * 3 + i]

            for j in range(self.max_peaks):
                xyz_idx[3] = j
                off = offset(xyz_idx, <cnp.npy_intp*>self.peak_strides, 4, 8)
                peak_value = self.peak_values_ptr[off]
                if peak_value <= 0:
                    continue

                peak_index = <cnp.npy_intp>self.peak_indices_ptr[off]
                if peak_index == closest_idx:
                    pmf_value += weights[m] * peak_value

        return pmf_value

    cdef cnp.npy_intp get_peaks_c(self,
                                  double* point,
                                  double* out_values,
                                  double* out_indices,
                                  double* out_weights,
                                  cnp.npy_intp max_peaks,
                                  cnp.npy_intp* out_valid) noexcept nogil:
        """Get neighboring peak lists and trilinear weights around ``point``.

        Parameters
        ----------
        point : double*, shape (3,)
            Query position in voxel coordinates.
        out_values : double*, shape (8 * max_peaks,)
            Output peak values, grouped by neighbor voxel.
        out_indices : double*, shape (8 * max_peaks,)
            Output peak indices, grouped by neighbor voxel.
        out_weights : double*, shape (8,)
            Trilinear weights for each neighboring voxel.
        max_peaks : int
            Maximum number of peaks to write per neighbor.
        out_valid : int*, shape (8,)
            Validity flag per neighbor voxel.

        Returns
        -------
        peaks_used : int
            Number of peaks written per neighbor (<= max_peaks).
        """
        cdef:
            cnp.npy_intp i, j, m
            cnp.npy_intp off
            cnp.npy_intp idx_base
            cnp.npy_intp index[24]
            cnp.npy_intp xyz[4]
            cnp.npy_intp peaks_used = max_peaks

        if peaks_used > self.max_peaks:
            peaks_used = self.max_peaks
        if peaks_used <= 0:
            return 0

        self._compute_trilinear(point, out_weights, index)

        for m in range(8):
            idx_base = m * max_peaks
            out_valid[m] = 0
            for j in range(max_peaks):
                out_values[idx_base + j] = 0.0
                out_indices[idx_base + j] = -1.0

            for i in range(3):
                xyz[i] = index[m * 3 + i]

            if (xyz[0] < 0 or xyz[0] >= self.peak_shape[0] or
                xyz[1] < 0 or xyz[1] >= self.peak_shape[1] or
                xyz[2] < 0 or xyz[2] >= self.peak_shape[2]):
                continue

            out_valid[m] = 1
            for j in range(peaks_used):
                xyz[3] = j
                off = offset(xyz, <cnp.npy_intp*>self.peak_strides, 4, 8)
                out_values[idx_base + j] = self.peak_values_ptr[off]
                out_indices[idx_base + j] = self.peak_indices_ptr[off]

        return peaks_used
