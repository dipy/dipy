# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: language_level=3
"""Parallel ODF-fingerprint matching.

For each voxel, ODF-FP selects the dictionary fingerprint that maximizes the
penalized cosine similarity::

    argmax_j  log(s_vj) - 2 * penalty * c_j     if penalty > 0
    argmax_j  s_vj                              if penalty == 0

where ``s_vj`` is the cosine similarity between voxel ``v`` and fingerprint
``j`` and ``c_j`` is the model-complexity penalty coefficient (the number of
fibers in fingerprint ``j`` beyond the first). Because the penalty is constant
within a group of equal-complexity fingerprints, the best score per group is
simply the best similarity there.

The matching is fused with the similarity matmul and streamed over blocks of
the dictionary (:func:`accumulate_block`), so the full ``(n_voxels x n_dict)``
similarity matrix is never materialized: :func:`accumulate_block` keeps only a
running best similarity/index per (voxel, group), and :func:`finalize_match`
applies the penalty once at the end. The voxels are processed in parallel,
following the matching strategy of DIPY's FORCE reconstruction.
"""

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport log, INFINITY
from cython.parallel import parallel, prange

from dipy.utils.omp import determine_num_threads
from dipy.utils.omp cimport set_num_threads, restore_default_num_threads

cnp.import_array()


ctypedef fused floating:
    float
    double


def accumulate_block(floating[:, ::1] sim, cnp.npy_intp[::1] group,
                     double[:, ::1] group_best, cnp.npy_intp[:, ::1] group_idx,
                     cnp.npy_intp col_offset, num_threads=None):
    """Update the running best similarity/index from one dictionary block.

    Parameters
    ----------
    sim : ndarray (n_voxels, block), float32 or float64, C-contiguous
        Cosine similarity between each voxel and the columns of this block;
        ``sim[v, j]`` corresponds to dictionary column ``col_offset + j``.
    group : ndarray (block,), intp
        Penalty group of each column (``max(0, n_fibers - 1)``). Columns to
        ignore (e.g. the void element) are flagged with a negative value.
    group_best : ndarray (n_voxels, n_groups), float64, C-contiguous
        Best similarity seen so far per (voxel, group). Updated in place.
    group_idx : ndarray (n_voxels, n_groups), intp, C-contiguous
        Global column index attaining ``group_best``. Updated in place.
    col_offset : int
        Global column index of ``sim[:, 0]``.
    num_threads : int, optional
        Number of OpenMP threads. ``None`` uses all available threads.
    """
    cdef:
        cnp.npy_intp n_vox = sim.shape[0]
        cnp.npy_intp b = sim.shape[1]
        cnp.npy_intp v, j, g
        double s
        int threads_to_use

    threads_to_use = determine_num_threads(num_threads)
    set_num_threads(threads_to_use)

    with nogil, parallel():
        for v in prange(n_vox, schedule='static'):
            for j in range(b):
                g = group[j]
                if g < 0:
                    continue
                s = sim[v, j]
                if s > group_best[v, g]:
                    group_best[v, g] = s
                    group_idx[v, g] = col_offset + j

    if num_threads is not None:
        restore_default_num_threads()


def finalize_match(double[:, ::1] group_best, cnp.npy_intp[:, ::1] group_idx,
                   double penalty, num_threads=None):
    """Penalized arg-max across penalty groups for every voxel.

    Parameters
    ----------
    group_best : ndarray (n_voxels, n_groups), float64, C-contiguous
        Best similarity per (voxel, group), as filled by
        :func:`accumulate_block`.
    group_idx : ndarray (n_voxels, n_groups), intp, C-contiguous
        Global column index attaining ``group_best`` (negative if the group
        was never seen).
    penalty : float
        Model-complexity penalty. No penalty is applied when ``penalty == 0``.
    num_threads : int, optional
        Number of OpenMP threads. ``None`` uses all available threads.

    Returns
    -------
    best : ndarray (n_voxels,), intp
        Index of the best-matching fingerprint for each voxel.
    """
    cdef:
        cnp.npy_intp n_vox = group_best.shape[0]
        cnp.npy_intp n_groups = group_best.shape[1]
        cnp.npy_intp v, g, best_j
        double score, best_score
        int threads_to_use
        cnp.npy_intp[::1] best = np.empty(n_vox, dtype=np.intp)

    threads_to_use = determine_num_threads(num_threads)
    set_num_threads(threads_to_use)

    with nogil, parallel():
        for v in prange(n_vox, schedule='static'):
            best_score = -INFINITY
            best_j = -1
            for g in range(n_groups):
                if group_idx[v, g] < 0:
                    continue
                if penalty > 0:
                    score = log(group_best[v, g]) - 2.0 * penalty * g
                else:
                    score = group_best[v, g]
                if score > best_score:
                    best_score = score
                    best_j = group_idx[v, g]
            best[v] = best_j

    if num_threads is not None:
        restore_default_num_threads()

    return np.asarray(best)


def select_best_match(double[:, ::1] similarity, cnp.npy_intp[::1] group,
                      double penalty, int n_groups, num_threads=None):
    """Best-matching fingerprint per voxel from a full similarity matrix.

    Thin convenience wrapper over :func:`accumulate_block` (applied once to the
    whole matrix) and :func:`finalize_match`; useful when the full similarity
    is already available.

    Parameters
    ----------
    similarity : ndarray (n_voxels, n_dict), float64, C-contiguous
        Cosine similarity between each voxel and each dictionary fingerprint.
    group : ndarray (n_dict,), intp
        Penalty group of each fingerprint (negative -> ignored).
    penalty : float
        Model-complexity penalty.
    n_groups : int
        Number of penalty groups (``max(group) + 1``).
    num_threads : int, optional
        Number of OpenMP threads. ``None`` uses all available threads.

    Returns
    -------
    best : ndarray (n_voxels,), intp
        Index of the best-matching fingerprint for each voxel.
    """
    cdef cnp.npy_intp n_vox = similarity.shape[0]
    group_best = np.full((n_vox, n_groups), -np.inf, dtype=np.float64)
    group_idx = np.full((n_vox, n_groups), -1, dtype=np.intp)
    accumulate_block(
        similarity, group, group_best, group_idx, 0, num_threads=num_threads
    )
    return finalize_match(group_best, group_idx, penalty, num_threads=num_threads)
