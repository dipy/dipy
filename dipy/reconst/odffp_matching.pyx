# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: language_level=3
"""Parallel ODF-fingerprint matching.

For each voxel, :func:`select_best_match` selects the dictionary fingerprint
that maximizes the penalized cosine similarity used by ODF-FP::

    argmax_j  log(s_vj) - 2 * penalty * c_j     if penalty > 0
    argmax_j  s_vj                              if penalty == 0

where ``s_vj`` is the cosine similarity between voxel ``v`` and fingerprint
``j`` and ``c_j`` is the model-complexity penalty coefficient (the number of
fibers in fingerprint ``j`` beyond the first). The voxels are matched in
parallel, following the matching strategy of DIPY's FORCE reconstruction.
"""

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport log, INFINITY
from libc.stdlib cimport malloc, free
from cython.parallel import parallel, prange

from dipy.utils.omp import determine_num_threads
from dipy.utils.omp cimport set_num_threads, restore_default_num_threads

cnp.import_array()


def select_best_match(double[:, ::1] similarity, cnp.npy_intp[::1] group,
                      double penalty, int n_groups, num_threads=None):
    """Select the best-matching fingerprint for each voxel.

    Parameters
    ----------
    similarity : ndarray (n_voxels, n_dict), float64, C-contiguous
        Cosine similarity between each voxel and each dictionary fingerprint.
    group : ndarray (n_dict,), intp
        Penalty group of each fingerprint, i.e. ``max(0, n_fibers - 1)``.
        Fingerprints to ignore (e.g. the void element) must be flagged with a
        negative value.
    penalty : float
        Model-complexity penalty. No penalty is applied when ``penalty == 0``.
    n_groups : int
        Number of penalty groups (``max(group) + 1``).
    num_threads : int, optional
        Number of OpenMP threads. ``None`` uses all available threads.

    Returns
    -------
    best : ndarray (n_voxels,), intp
        Index of the best-matching fingerprint for each voxel.
    """
    cdef:
        cnp.npy_intp n_vox = similarity.shape[0]
        cnp.npy_intp n_dict = similarity.shape[1]
        cnp.npy_intp v, j, g, gj, best_j
        double s, score, best_score
        double* group_best
        cnp.npy_intp* group_idx
        int threads_to_use
        cnp.npy_intp[::1] best = np.empty(n_vox, dtype=np.intp)

    threads_to_use = determine_num_threads(num_threads)
    set_num_threads(threads_to_use)

    with nogil, parallel():
        # Per-thread scratch holding the best similarity and index per group.
        group_best = <double*> malloc(n_groups * sizeof(double))
        group_idx = <cnp.npy_intp*> malloc(n_groups * sizeof(cnp.npy_intp))

        for v in prange(n_vox, schedule='static'):
            for g in range(n_groups):
                group_best[g] = -INFINITY
                group_idx[g] = -1

            # Best similarity within each penalty group (penalty is constant
            # there, so the best score is the best similarity).
            for j in range(n_dict):
                gj = group[j]
                if gj < 0:
                    continue
                s = similarity[v, j]
                if s > group_best[gj]:
                    group_best[gj] = s
                    group_idx[gj] = j

            # Compare the groups once the penalty is applied.
            best_score = -INFINITY
            best_j = -1
            for g in range(n_groups):
                if group_idx[g] < 0:
                    continue
                if penalty > 0:
                    score = log(group_best[g]) - 2.0 * penalty * g
                else:
                    score = group_best[g]
                if score > best_score:
                    best_score = score
                    best_j = group_idx[g]
            best[v] = best_j

        free(group_best)
        free(group_idx)

    if num_threads is not None:
        restore_default_num_threads()

    return np.asarray(best)
