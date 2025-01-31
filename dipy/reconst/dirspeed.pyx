# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False


import numpy as np
cimport numpy as cnp

import cython
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.math cimport cos, M_PI, INFINITY

from dipy.core.math cimport f_max, f_array_min
from dipy.reconst.recspeed cimport local_maxima_c, search_descending_c, remove_similar_vertices_c


cdef cnp.uint16_t peak_directions_c(
    double[:] odf,
    double[:, ::1] sphere_vertices,
    cnp.uint16_t[:, ::1] sphere_edges,
    double relative_peak_threshold,
    double min_separation_angle,
    bint is_symmetric,
    double[:, ::1] out_directions,
    double[::1] out_values,
    cnp.npy_intp[::1] out_indices,
    double[:, :] unique_vertices,
    cnp.uint16_t[:] mapping,
    cnp.uint16_t[:] index
) noexcept nogil:
    """Get the directions of odf peaks.

    Peaks are defined as points on the odf that are greater than at least one
    neighbor and greater than or equal to all neighbors. Peaks are sorted in
    descending order by their values then filtered based on their relative size
    and spacing on the sphere. An odf may have 0 peaks, for example if the odf
    is perfectly isotropic.

    Parameters
    ----------
    odf : 1d ndarray
        The odf function evaluated on the vertices of `sphere`
    sphere_vertices : (N,3) ndarray
        The sphere vertices providing discrete directions for evaluation.
    sphere_edges : (N, 2) ndarray

    relative_peak_threshold : float in [0., 1.]
        Only peaks greater than ``min + relative_peak_threshold * scale`` are
        kept, where ``min = max(0, odf.min())`` and
        ``scale = odf.max() - min``.
    min_separation_angle : float in [0, 90]
        The minimum distance between directions. If two peaks are too close
        only the larger of the two is returned.
    is_symmetric : bool, optional
        If True, v is considered equal to -v.
    out_directions : (N, 3) ndarray
        The directions of the peaks, N vertices for sphere, one for each peak
    out_values : (N,) ndarray
        The peak values
    out_indices: (N,) ndarray
        The peak indices of the directions on the sphere
    unique_vertices : (N, 3) ndarray
        The unique vertices
    mapping  : (N,) ndarray
        For each element ``vertices[i]`` ($i \in 0..N-1$), the index $j$ to a
        vertex in `unique_vertices` that is less than `theta` degrees from
        ``vertices[i]``.
    index : (N,) ndarray
        `index` gives the reverse of `mapping`.  For each element
        ``unique_vertices[j]`` ($j \in 0..M-1$), the index $i$ to a vertex in
        `vertices` that is less than `theta` degrees from
        ``unique_vertices[j]``.  If there is more than one element of
        `vertices` that is less than theta degrees from `unique_vertices[j]`,
        return the first (lowest index) matching value.
    Returns
    -------
    n_unique : int
        The number of unique peaks

    Notes
    -----
    If the odf has any negative values, they will be clipped to zeros.

    """
    cdef cnp.npy_intp i, n, idx
    cdef long count
    cdef double odf_min
    cdef cnp.npy_intp num_vertices = sphere_vertices.shape[0]
    cdef double* tmp_buffer
    cdef cnp.uint16_t n_unique = 0

    count = local_maxima_c(odf, sphere_edges, out_values, out_indices)

    # If there is only one peak return
    if count == 0:
        return 0
    elif count == 1:
        return 1

    odf_min = f_array_min[double](&odf[0], num_vertices)
    odf_min = f_max[double](odf_min, 0.0)

    # because of the relative threshold this algorithm will give the same peaks
    # as if we divide (values - odf_min) with (odf_max - odf_min) or not so
    # here we skip the division to increase speed
    tmp_buffer = <double*>malloc(count * sizeof(double))
    memcpy(tmp_buffer, &out_values[0], count * sizeof(double))

    for i in range(count):
        tmp_buffer[i] -= odf_min

    # Remove small peaks
    n = search_descending_c[cython.double](tmp_buffer, <cnp.npy_intp>count, relative_peak_threshold)

    for i in range(n):
        idx = out_indices[i]
        out_directions[i, :] = sphere_vertices[idx, :]

    # Remove peaks too close together
    remove_similar_vertices_c(
        out_directions[:n, :],
        min_separation_angle,
        is_symmetric,
        0,  # return_mapping
        1,  # return_index
        unique_vertices[:n, :],
        mapping[:n],
        index[:n],
        &n_unique
    )

    # Update final results
    for i in range(n_unique):
        idx = index[i]
        out_directions[i, :] = unique_vertices[i, :]
        out_values[i] = out_values[idx]
        out_indices[i] = out_indices[idx]

    free(tmp_buffer)
    return n_unique


def peak_directions(
    double[:] odf,
    sphere,
    *,
    relative_peak_threshold=0.5,
    min_separation_angle=25,
    bint is_symmetric=True,
):
    """Get the directions of odf peaks.

    Peaks are defined as points on the odf that are greater than at least one
    neighbor and greater than or equal to all neighbors. Peaks are sorted in
    descending order by their values then filtered based on their relative size
    and spacing on the sphere. An odf may have 0 peaks, for example if the odf
    is perfectly isotropic.

    Parameters
    ----------
    odf : 1d ndarray
        The odf function evaluated on the vertices of `sphere`
    sphere : Sphere
        The Sphere providing discrete directions for evaluation.
    relative_peak_threshold : float in [0., 1.]
        Only peaks greater than ``min + relative_peak_threshold * scale`` are
        kept, where ``min = max(0, odf.min())`` and
        ``scale = odf.max() - min``.
    min_separation_angle : float in [0, 90]
        The minimum distance between directions. If two peaks are too close
        only the larger of the two is returned.
    is_symmetric : bool, optional
        If True, v is considered equal to -v.

    Returns
    -------
    directions : (N, 3) ndarray
        N vertices for sphere, one for each peak
    values : (N,) ndarray
        peak values
    indices : (N,) ndarray
        peak indices of the directions on the sphere

    Notes
    -----
    If the odf has any negative values, they will be clipped to zeros.

    """
    cdef double[:, ::1] vertices = sphere.vertices
    cdef cnp.uint16_t[:, ::1] edges = sphere.edges

    cdef cnp.npy_intp num_vertices = sphere.vertices.shape[0]
    cdef cnp.uint16_t n_unique = 0

    cdef double[:, ::1] directions_out = np.zeros((num_vertices, 3), dtype=np.float64)
    cdef double[::1] values_out = np.zeros(num_vertices, dtype=np.float64)
    cdef cnp.npy_intp[::1] indices_out = np.zeros(num_vertices, dtype=np.intp)
    cdef cnp.float64_t[:, ::1] unique_vertices = np.empty((num_vertices, 3), dtype=np.float64)
    cdef cnp.uint16_t[::1] mapping = None
    cdef cnp.uint16_t[::1] index = np.empty(num_vertices, dtype=np.uint16)

    n_unique = peak_directions_c(
        odf, vertices, edges, relative_peak_threshold, min_separation_angle, is_symmetric,
        directions_out, values_out, indices_out, unique_vertices, mapping, index
    )

    if n_unique == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=int)
    elif n_unique == 1:
        return (
            sphere.vertices[np.asarray(indices_out[:n_unique]).astype(int)],
            np.asarray(values_out[:n_unique]),
            np.asarray(indices_out[:n_unique]),
        )
    return (
        np.asarray(directions_out[:n_unique, :]),
        np.asarray(values_out[:n_unique]),
        np.asarray(indices_out[:n_unique]),
    )