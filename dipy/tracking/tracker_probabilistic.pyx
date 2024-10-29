# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: Nonecheck=False

from libc.stdio cimport printf

import numpy as np
cimport numpy as cnp

from dipy.direction.pmf cimport PmfGen
from dipy.utils.fast_numpy cimport (copy_point, cumsum, norm, normalize,
                                    where_to_insert, random)
from dipy.tracking.fast_tracking cimport prepare_pmf
from dipy.tracking.tracker_parameters cimport (TrackerParameters, TrackerStatus, 
                                               SUCCESS, FAIL)
from libc.stdlib cimport malloc, free


cdef TrackerStatus probabilistic_tracker(double* point,
                                         double* direction,
                                         TrackerParameters params,
                                         double* stream_data,
                                         PmfGen pmf_gen) noexcept nogil:
    """
    Propagates the position by step_size amount. The propagation use randomly samples 
    direction of a sphere based on probability mass function (pmf).

    Parameters
    ----------
    point : double[3]
        Current tracking position.
    direction : double[3]
        Previous tracking direction.
    params : TrackerParameters
        Parallel Transport Tractography (PTT) parameters.
    stream_data : double*
        Streamline data persitant across tracking steps.
    pmf_gen : PmfGen
        Orientation data.

    Returns
    -------
    status : TrackerStatus
        Returns SUCCESS if the propagation was successful, or
        FAIL otherwise.
    """

    cdef:
        cnp.npy_intp i, idx
        double* newdir
        double* pmf
        double last_cdf, cos_sim
        cnp.npy_intp len_pmf=pmf_gen.pmf.shape[0]

    if norm(direction) == 0:
        return FAIL
    normalize(direction)

    pmf = <double*> malloc(len_pmf * sizeof(double))
    prepare_pmf(pmf, point, pmf_gen, params.sh.pmf_threshold, len_pmf)

    for i in range(len_pmf):
        cos_sim = pmf_gen.vertices[i][0] * direction[0] \
                + pmf_gen.vertices[i][1] * direction[1] \
                + pmf_gen.vertices[i][2] * direction[2]
        if cos_sim < 0:
            cos_sim = cos_sim * -1
        if cos_sim < params.cos_similarity:
            pmf[i] = 0

    cumsum(pmf, pmf, len_pmf)
    last_cdf = pmf[len_pmf - 1]
    if last_cdf == 0:
        free(pmf)
        return FAIL

    idx = where_to_insert(pmf, random() * last_cdf, len_pmf)
    newdir = &pmf_gen.vertices[idx][0]
    # Update direction
    if (direction[0] * newdir[0]
        + direction[1] * newdir[1]
        + direction[2] * newdir[2] > 0):
        copy_point(newdir, direction)
    else:
        copy_point(newdir, direction)
        direction[0] = direction[0] * -1
        direction[1] = direction[1] * -1
        direction[2] = direction[2] * -1
    free(pmf)
    return SUCCESS
    