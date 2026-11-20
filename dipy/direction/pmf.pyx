# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

from cython.parallel import prange, threadid
import numpy as np
cimport numpy as cnp

from dipy.core.sphere import Sphere
from dipy.reconst import shm
from dipy.utils.omp import determine_num_threads

from dipy.core.interpolation cimport (
    _trilinear_interpolation_iso,
    offset,
    trilinear_interpolate3d_c,
    trilinear_interpolate4d_c,
)
from libc.stdlib cimport malloc, free
from dipy.reconst.dirspeed cimport peak_directions_c
from scipy.linalg.cython_blas cimport dgemv

cdef extern from "stdlib.h" nogil:
    void *memset(void *ptr, int value, size_t num)

# stack buffer size for interpolated SH coefficients
# if larger than this, switch to malloc
cdef enum:
    _SH_STACK_COEFF = 256


cdef class PmfGen:

    def __init__(self,
                 double[:, :, :, :] data,
                 object sphere):
        self.data = np.asarray(data, dtype=float, order="C")
        self.vertices = np.asarray(sphere.vertices, dtype=float, order="C")
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
        trilinear_interpolate3d_c[double](self.data[:, :, :, idx], point,
                                          &pmf_value)
        return pmf_value


def _sh_order_from_ncoef(ncoef, full_basis=None):
    """Infer ``(sh_order_max, full_basis)`` from a number of SH coefficients.

    A symmetric basis (even ``l`` only) has ``(l + 1)(l + 2) / 2``
    coefficients, a full basis has ``(l + 1)**2``.

    Parameters
    ----------
    ncoef : int
        Number of SH coefficients.
    full_basis : bool or None, optional
        Basis of the coefficients. If None, the symmetric basis is tried
        first, then the full one.

    Returns
    -------
    sh_order_max : int
        Maximum SH order.
    full_basis : bool
        True for a full basis.
    """
    candidates = (False, True) if full_basis is None else (bool(full_basis),)
    for full in candidates:
        sh_order = shm.order_from_ncoef(ncoef, full_basis=full)
        if full:
            expected = (sh_order + 1) ** 2
        else:
            expected = (sh_order + 1) * (sh_order + 2) // 2
        if expected == ncoef and (full or sh_order % 2 == 0):
            return sh_order, full
    kind = {None: "a symmetric or a full", True: "a full", False: "a symmetric"}
    raise ValueError(
        f"{ncoef} coefficients do not match {kind[full_basis]} SH basis."
    )


cdef class SHCoeffPmfGen(PmfGen):

    def __init__(self,
                 double[:, :, :, :] shcoeff_array,
                 object sphere,
                 object basis_type,
                 legacy=True,
                 full_basis=None):
        """
        Parameters
        ----------
        shcoeff_array : ndarray, shape (x, y, z, ncoef)
            Spherical harmonic coefficients.
        sphere : Sphere
            Sphere on which the pmf is evaluated.
        basis_type : str or None
            One of ``dipy.reconst.shm.sph_harm_lookup``.
        legacy : bool, optional
            Use the legacy basis definition.
        full_basis : bool or None, optional
            True if ``shcoeff_array`` uses a full (odd and even order) SH
            basis, as produced by asymmetric ODF models. If None, it is
            inferred from the number of coefficients.
        """
        cdef:
            int sh_order

        PmfGen.__init__(self, shcoeff_array, sphere)

        sh_order, full_basis = _sh_order_from_ncoef(
            shcoeff_array.shape[3], full_basis
        )
        try:
            basis = shm.sph_harm_lookup[basis_type]
        except KeyError:
            raise ValueError(f"{basis_type} is not a known basis type.")
        B, _, _ = basis(sh_order, sphere.theta, sphere.phi,
                        full_basis=full_basis, legacy=legacy)
        if B.shape[1] != shcoeff_array.shape[3]:
            raise ValueError(
                f"SH basis has {B.shape[1]} functions but "
                f"{shcoeff_array.shape[3]} coefficients were given."
            )
        # C-contiguous so get_pmf_c can hand it to BLAS
        self.B = np.asarray(B, dtype=float, order="C")
        self.nb_coeff = self.B.shape[1]

    cdef double* get_pmf_c(self, double* point, double* out) noexcept nogil:
        cdef:
            int m = self.nb_coeff, n = self.pmf.shape[0], one = 1
            double alpha = 1.0, beta = 0.0
            double stack_buf[_SH_STACK_COEFF]
            double* coeff = stack_buf

        if m > _SH_STACK_COEFF:
            coeff = <double*> malloc(m * sizeof(double))

        if trilinear_interpolate4d_c(self.data, point, coeff) != 0:
            memset(out, 0, n * sizeof(double))
        else:
            # C-order (n, m) B is column-major (m, n) to BLAS, hence "T"
            dgemv("T", &m, &n, &alpha, &self.B[0, 0], &m, coeff, &one,
                  &beta, out, &one)

        if coeff != stack_buf:
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
            int m = self.nb_coeff
            double stack_buf[_SH_STACK_COEFF]
            double* coeff = stack_buf
            const double* brow = &self.B[idx, 0]
            double pmf_value = 0

        if m > _SH_STACK_COEFF:
            coeff = <double*> malloc(m * sizeof(double))

        if trilinear_interpolate4d_c(self.data, point, coeff) == 0:
            for j in range(m):
                pmf_value = pmf_value + brow[j] * coeff[j]

        if coeff != stack_buf:
            free(coeff)
        return pmf_value


cdef class SimplePeakGen(PmfGen):
    """PmfGen subclass for sphere-based peak data.

    This class stores peak indices and values, providing the PmfGen
    interface required by the tractogram generator.

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
                 peak_indices,
                 peak_values,
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
        cdef cnp.ndarray empty_data
        cdef cnp.ndarray indices_arr
        cdef cnp.ndarray values_arr
        cdef cnp.ndarray odf_vertices_arr
        cdef cnp.ndarray sphere_vertices_arr
        cdef cnp.npy_intp[:] indices_strides
        cdef cnp.npy_intp[:] values_strides

        # Keep indices as int32 to avoid float->int casts in tight loops.
        indices_arr = np.asarray(peak_indices, dtype=np.int32, order="C")
        values_arr = np.asarray(peak_values, dtype=float, order="C")

        if indices_arr.ndim != 4 or values_arr.ndim != 4:
            raise ValueError(
                "peak_indices and peak_values must be 4D arrays"
            )

        if (
            indices_arr.shape[0] != values_arr.shape[0] or
            indices_arr.shape[1] != values_arr.shape[1] or
            indices_arr.shape[2] != values_arr.shape[2]
        ):
            raise ValueError(
                "peak_indices and peak_values must have matching spatial dimensions"
            )
        if indices_arr.shape[3] != values_arr.shape[3]:
            raise ValueError(
                "peak_indices and peak_values must have same number of peaks"
            )

        # SimplePeakGen is peak-native; keep an empty placeholder for `data`
        # to satisfy the shared PmfGen layout without allocating dummy PMF data.
        empty_data = np.zeros((0, 0, 0, 0), dtype=float, order="C")
        odf_vertices_arr = np.asarray(odf_vertices, dtype=float, order="C")
        self.data = empty_data
        # Required shared PmfGen fields used by generic tracking interfaces.
        self.vertices = odf_vertices_arr
        self.pmf = np.zeros(self.vertices.shape[0], dtype=float)
        sphere_vertices_arr = np.asarray(sphere.vertices, dtype=float, order="C")
        if np.array_equal(sphere_vertices_arr, odf_vertices_arr):
            self.sphere = sphere
        else:
            self.sphere = Sphere(xyz=odf_vertices_arr)

        # Keep owning references alive; raw pointers below borrow this memory.
        self.peak_indices = indices_arr
        self.peak_values = values_arr
        self.max_peaks = indices_arr.shape[3]

        # Fast-path pointer access for tight nogil loops.
        self.peak_indices_ptr = &self.peak_indices[0, 0, 0, 0]
        self.peak_values_ptr = &self.peak_values[0, 0, 0, 0]

        # Cache shape metadata once to avoid repeated lookups in inner loops.
        self.peak_shape[0] = indices_arr.shape[0]
        self.peak_shape[1] = indices_arr.shape[1]
        self.peak_shape[2] = indices_arr.shape[2]
        self.peak_shape[3] = self.max_peaks

        # Cache strides once; indices/values have different item sizes.
        indices_strides = <cnp.npy_intp[:4]>(<cnp.npy_intp*>indices_arr.strides)
        self.peak_indices_strides[0] = indices_strides[0]
        self.peak_indices_strides[1] = indices_strides[1]
        self.peak_indices_strides[2] = indices_strides[2]
        self.peak_indices_strides[3] = indices_strides[3]

        values_strides = <cnp.npy_intp[:4]>(<cnp.npy_intp*>values_arr.strides)
        self.peak_values_strides[0] = values_strides[0]
        self.peak_values_strides[1] = values_strides[1]
        self.peak_values_strides[2] = values_strides[2]
        self.peak_values_strides[3] = values_strides[3]

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
            cnp.npy_intp off_indices
            cnp.npy_intp off_values
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
                off_values = offset(
                    xyz,
                    <cnp.npy_intp*>self.peak_values_strides,
                    4,
                    sizeof(double),
                )
                peak_value = self.peak_values_ptr[off_values]
                if peak_value <= 0:
                    continue

                off_indices = offset(
                    xyz,
                    <cnp.npy_intp*>self.peak_indices_strides,
                    4,
                    sizeof(cnp.int32_t),
                )
                peak_index = self.peak_indices_ptr[off_indices]
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
            cnp.npy_intp off_indices
            cnp.npy_intp off_values
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
                off_values = offset(
                    xyz_idx,
                    <cnp.npy_intp*>self.peak_values_strides,
                    4,
                    sizeof(double),
                )
                peak_value = self.peak_values_ptr[off_values]
                if peak_value <= 0:
                    continue

                off_indices = offset(
                    xyz_idx,
                    <cnp.npy_intp*>self.peak_indices_strides,
                    4,
                    sizeof(cnp.int32_t),
                )
                peak_index = self.peak_indices_ptr[off_indices]
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
            cnp.npy_intp off_indices
            cnp.npy_intp off_values
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

            if (
                xyz[0] < 0 or xyz[0] >= self.peak_shape[0]
                or xyz[1] < 0 or xyz[1] >= self.peak_shape[1]
                or xyz[2] < 0 or xyz[2] >= self.peak_shape[2]
            ):
                continue

            out_valid[m] = 1
            for j in range(peaks_used):
                xyz[3] = j
                off_values = offset(
                    xyz,
                    <cnp.npy_intp*>self.peak_values_strides,
                    4,
                    sizeof(double),
                )
                off_indices = offset(
                    xyz,
                    <cnp.npy_intp*>self.peak_indices_strides,
                    4,
                    sizeof(cnp.int32_t),
                )
                out_values[idx_base + j] = self.peak_values_ptr[off_values]
                out_indices[idx_base + j] = self.peak_indices_ptr[off_indices]

        return peaks_used


def pmf_peaks(double[:, ::1] points,
              PmfGen pmf_gen,
              *,
              bint is_symmetric=True,
              int nbr_threads=0,
              int max_peaks=-1,
              double relative_peak_threshold=0.5,
              double min_separation_angle=25):
    """Peaks of the pmf at each point (voxel coordinates), in parallel.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
        Points in voxel coordinates. Points outside the volume have no peak.
    pmf_gen : PmfGen
        Probability mass function generator.
    is_symmetric : bool, optional
        If True, v is considered equal to -v.
    nbr_threads : int, optional
        Number of threads. All available threads if <= 0.
    max_peaks : int, optional
        Maximum number of peaks per point. All peaks if <= 0.
    relative_peak_threshold : float, optional
        Only peaks greater than ``min + relative_peak_threshold * scale`` are
        kept (see :func:`dipy.direction.peaks.peak_directions`).
    min_separation_angle : float, optional
        Minimum angle, in degrees, between two peaks.

    Returns
    -------
    offsets : ndarray, shape (N + 1,)
        Exclusive prefix sum of the number of peaks per point.
    directions : ndarray, shape (offsets[-1], 3)
        Peak directions, sorted by decreasing peak value within each point.
    """
    cdef:
        cnp.npy_intp n = points.shape[0], dimt = pmf_gen.pmf.shape[0]
        cnp.npy_intp cap = dimt if max_peaks <= 0 else min(max_peaks, dimt)
        cnp.npy_intp i, j, k, t
        double[:, ::1] verts = pmf_gen.vertices
        cnp.uint16_t[:, ::1] edges = np.asarray(
            pmf_gen.sphere.edges, dtype=np.uint16, order="C")
        cnp.npy_intp[::1] counts, offsets
        cnp.int32_t[:, ::1] peak_idx
        double[:, ::1] dirs

    if nbr_threads <= 0:
        nbr_threads = determine_num_threads(None)

    # per-thread scratch for peak_directions_c
    cdef:
        double[:, ::1] pmf = np.empty((nbr_threads, dimt))
        double[:, ::1] values = np.empty((nbr_threads, dimt))
        cnp.npy_intp[:, ::1] indices = np.empty((nbr_threads, dimt), dtype=np.intp)
        double[:, :, ::1] out_dirs = np.empty((nbr_threads, dimt, 3))
        double[:, :, ::1] uniq = np.empty((nbr_threads, dimt, 3))
        cnp.uint16_t[:, ::1] mapping = np.empty((nbr_threads, dimt), dtype=np.uint16)
        cnp.uint16_t[:, ::1] index = np.empty((nbr_threads, dimt), dtype=np.uint16)

    counts = np.empty(n, dtype=np.intp)
    peak_idx = np.empty((n, cap), dtype=np.int32)
    for i in prange(n, nogil=True, num_threads=nbr_threads, schedule="dynamic",
                    chunksize=64):
        t = threadid()
        k = _peaks_at(&points[i, 0], pmf_gen, verts, edges, is_symmetric,
                      relative_peak_threshold, min_separation_angle, cap,
                      pmf[t], out_dirs[t], values[t], indices[t], uniq[t],
                      mapping[t], index[t])
        counts[i] = k
        for j in range(k):
            peak_idx[i, j] = <cnp.int32_t>indices[t, j]

    offsets = np.zeros(n + 1, dtype=np.intp)
    np.cumsum(counts, out=np.asarray(offsets)[1:])
    dirs = np.empty((offsets[n], 3))
    for i in range(n):
        for j in range(counts[i]):
            dirs[offsets[i] + j, 0] = verts[peak_idx[i, j], 0]
            dirs[offsets[i] + j, 1] = verts[peak_idx[i, j], 1]
            dirs[offsets[i] + j, 2] = verts[peak_idx[i, j], 2]
    return np.asarray(offsets), np.asarray(dirs)


cdef cnp.npy_intp _peaks_at(double* point,
                            PmfGen pmf_gen,
                            double[:, ::1] verts,
                            cnp.uint16_t[:, ::1] edges,
                            bint is_symmetric,
                            double relative_peak_threshold,
                            double min_separation_angle,
                            cnp.npy_intp cap,
                            double[::1] pmf,
                            double[:, ::1] out_dirs,
                            double[::1] values,
                            cnp.npy_intp[::1] indices,
                            double[:, ::1] uniq,
                            cnp.uint16_t[::1] mapping,
                            cnp.uint16_t[::1] index) noexcept nogil:
    cdef cnp.npy_intp k
    pmf_gen.get_pmf_c(point, &pmf[0])
    k = peak_directions_c(pmf, verts, edges, relative_peak_threshold,
                          min_separation_angle, is_symmetric, out_dirs,
                          values, indices, uniq, mapping, index)
    return k if k < cap else cap
