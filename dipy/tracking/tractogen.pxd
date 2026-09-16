cimport numpy as cnp

from dipy.tracking.stopping_criterion cimport StoppingCriterion, StreamlineStatus
from dipy.direction.pmf cimport PmfGen
from dipy.tracking.tracker_parameters cimport TrackerParameters


cdef void generate_tractogram_c(
    double[:, ::1] seeds,
    double[:, ::1] directions,
    cnp.npy_intp[::1] sl_seed,
    cnp.npy_intp sl_offset,
    cnp.npy_uint64 rng_seed,
    int nbr_threads,
    StoppingCriterion sc,
    TrackerParameters params,
    PmfGen pmf_gen,
    double[:, ::1] scratch,
    double[:, ::1] sline,
    int[:, ::1] stream_idx,
    int[::1] status,
)


cdef StreamlineStatus generate_local_streamline(
    double* seed,
    double* position,
    double* stream,
    int* stream_idx,
    double* stream_data,
    cnp.npy_uint64 rng_seed,
    StoppingCriterion sc,
    TrackerParameters params,
    PmfGen pmf_gen,
) noexcept nogil
