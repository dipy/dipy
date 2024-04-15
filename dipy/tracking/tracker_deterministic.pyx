cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as cnp

from dipy.core.interpolation cimport trilinear_interpolate4d_c
from dipy.direction.pmf cimport PmfGen
from dipy.tracking.stopping_criterion cimport StoppingCriterion
from dipy.utils.fast_numpy cimport (copy_point, cumsum, norm, normalize,
                                    where_to_insert, random)
from dipy.tracking.fast_tracking cimport get_pmf, TrackingParameters

cdef int deterministic_tracker(double* point,
                                       double* direction,
                                       TrackingParameters params,
                                       PmfGen pmf_gen) noexcept nogil:
    cdef:
        cnp.npy_intp i, max_idx
        double max_value=0
        double* newdir
        double* pmf
        double cos_sim
        cnp.npy_intp len_pmf=pmf_gen.pmf.shape[0]

    pmf = <double*> malloc(len_pmf * sizeof(double))
    if get_pmf(pmf, point, pmf_gen, params.probabilistic.pmf_threshold, len_pmf):
        free(pmf)
        return 1
    if norm(direction) == 0:
        free(pmf)
        return 1
    normalize(direction)

    for i in range(len_pmf):
        cos_sim = pmf_gen.vertices[i][0] * direction[0] \
                + pmf_gen.vertices[i][1] * direction[1] \
                + pmf_gen.vertices[i][2] * direction[2]
        if cos_sim < 0:
            cos_sim = cos_sim * -1
        if cos_sim > params.probabilistic.cos_similarity and pmf[i] > max_value:
            max_idx = i
            max_value = pmf[i]

    if max_value <= 0:
        free(pmf)
        return 1

    newdir = &pmf_gen.vertices[max_idx][0]
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
    return 0