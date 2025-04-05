cimport numpy as cnp

from dipy.tracking.stopping_criterion cimport StoppingCriterion, StreamlineStatus
from dipy.direction.pmf cimport PmfGen
from dipy.tracking.tracker_parameters cimport TrackerParameters


cdef void generate_tractogram_c(double[:,::1] seed_positions,
                                double[:,::1] seed_directions,
                                int nbr_threads,
                                StoppingCriterion sc,
                                TrackerParameters params,
                                PmfGen pmf_gen,
                                double** streamlines,
                                int* length,
                                StreamlineStatus* status)


cdef StreamlineStatus generate_local_streamline(double* seed,
                                   double* position,
                                   double* stream,
                                   int* stream_idx,
                                   StoppingCriterion sc,
                                   TrackerParameters params,
                                   PmfGen pmf_gen) noexcept nogil


cdef void prepare_pmf(double* pmf,
                      double* point,
                      PmfGen pmf_gen,
                      double pmf_threshold,
                      int pmf_len) noexcept nogil
