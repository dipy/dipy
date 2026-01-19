# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

"""Peak direction data container for EUDX tracking."""

import numpy as np
cimport numpy as cnp
cimport cython

from dipy.core.interpolation cimport _trilinear_interpolation_iso, offset

cdef class PeakDirectionGen:
    """Peak direction data container with trilinear interpolation.

    This class stores sphere-based peak representation (indices + odf_vertices)
    for EUDX-style tracking. It provides trilinear interpolation of peak data
    across neighboring voxels.

    Parameters
    ----------
    peak_indices : ndarray, shape (X, Y, Z, npeaks)
        Indices into odf_vertices for each peak at each voxel.
    peak_values : ndarray, shape (X, Y, Z, npeaks)
        Peak strength values (QA, GFA, etc.) at each voxel.
    odf_vertices : ndarray, shape (N_vertices, 3)
        Sphere vertices representing possible peak directions.

    Notes
    -----
    This class maintains backward compatibility with EUDX's sphere-based
    representation where peaks are stored as indices into a predefined sphere.
    """

    def __init__(self,
                 double[:, :, :, :] peak_indices,
                 double[:, :, :, :] peak_values,
                 double[:, :] odf_vertices):
        """Initialize PeakDirectionGen with peak data.

        Parameters
        ----------
        peak_indices : memoryview, shape (X, Y, Z, npeaks)
            Indices into odf_vertices.
        peak_values : memoryview, shape (X, Y, Z, npeaks)
            Peak strength values.
        odf_vertices : memoryview, shape (N_vertices, 3)
            Sphere vertices.
        """
        if (peak_indices.shape[0] != peak_values.shape[0] or
            peak_indices.shape[1] != peak_values.shape[1] or
            peak_indices.shape[2] != peak_values.shape[2]):
            raise ValueError("peak_indices and peak_values must have matching spatial dimensions")
        if peak_indices.shape[3] != peak_values.shape[3]:
            raise ValueError("peak_indices and peak_values must have same number of peaks")

        self.peak_indices = peak_indices
        self.peak_values = peak_values
        self.odf_vertices = odf_vertices
        self.max_peaks = peak_indices.shape[3]

        self.shape[0] = peak_indices.shape[0]
        self.shape[1] = peak_indices.shape[1]
        self.shape[2] = peak_indices.shape[2]
        self.shape[3] = self.max_peaks

        cdef cnp.ndarray indices_arr = np.asarray(peak_indices)
        cdef cnp.npy_intp[:] arr_strides = <cnp.npy_intp[:4]>(<cnp.npy_intp*>indices_arr.strides)
        self.strides[0] = arr_strides[0]
        self.strides[1] = arr_strides[1]
        self.strides[2] = arr_strides[2]
        self.strides[3] = arr_strides[3]

    cdef void get_peak_data_c(self,
                              double* point,
                              double* indices_out,
                              double* values_out,
                              double* weights_out) noexcept nogil:
        """Get interpolated peak data at a given point.

        This method computes trilinear interpolation weights for the 8 voxels
        surrounding the given point, then extracts peak indices and values
        from each neighbor.

        Parameters
        ----------
        point : double*, shape (3,)
            Query position in voxel coordinates.
        indices_out : double*, shape (8 * max_peaks,)
            Output: peak indices for each of 8 neighbors (flattened).
        values_out : double*, shape (8 * max_peaks,)
            Output: peak values for each of 8 neighbors (flattened).
        weights_out : double*, shape (8,)
            Output: trilinear interpolation weights for each neighbor.

        Notes
        -----
        The output arrays are organized as:
        - indices_out[neighbor * max_peaks + peak_idx]
        - values_out[neighbor * max_peaks + peak_idx]
        - weights_out[neighbor]

        where neighbor ranges from 0-7 and peak_idx ranges from 0 to max_peaks-1.
        """
        cdef:
            cnp.npy_intp index[24]  # 8 neighbors * 3 coordinates
            cnp.npy_intp xyz[4]
            cnp.npy_intp i, j, m, off
            int peaks = self.max_peaks
            double* indices_ptr
            double* values_ptr

        # Get raw pointers to data (safe in nogil context)
        indices_ptr = &self.peak_indices[0, 0, 0, 0]
        values_ptr = &self.peak_values[0, 0, 0, 0]

        # Get trilinear interpolation weights and neighbor indices
        _trilinear_interpolation_iso(point, weights_out, index)

        # For each of the 8 neighbors
        for m in range(8):
            # Extract x, y, z coordinates for this neighbor
            for i in range(3):
                xyz[i] = index[m * 3 + i]

            # Check if neighbor is within volume bounds
            if (xyz[0] < 0 or xyz[0] >= self.shape[0] or
                xyz[1] < 0 or xyz[1] >= self.shape[1] or
                xyz[2] < 0 or xyz[2] >= self.shape[2]):
                # Out of bounds - zero out this neighbor's data
                for j in range(peaks):
                    indices_out[m * peaks + j] = 0
                    values_out[m * peaks + j] = 0
                weights_out[m] = 0
                continue

            # Extract all peaks for this voxel
            for j in range(peaks):
                xyz[3] = j
                off = offset(xyz, self.strides, 4, 8)
                indices_out[m * peaks + j] = indices_ptr[off]
                values_out[m * peaks + j] = values_ptr[off]
