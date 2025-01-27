cimport numpy as cnp

from dipy.direction.pmf cimport PmfGen
from dipy.tracking.tracker_parameters cimport TrackerParameters

cdef cnp.npy_intp _propagation_direction(double *point, double* prev, double* qa,
                                double *ind, double *odf_vertices,
                                double qa_thr, double ang_thr,
                                cnp.npy_intp *qa_shape,cnp.npy_intp* strides,
                                double *direction,double total_weight) noexcept nogil


cdef int initialize_ptt(TrackerParameters params,
                        double* stream_data,
                        PmfGen pmf_gen,
                        double* seed_point,
                        double* seed_direction) noexcept nogil


cdef void prepare_ptt_propagator(TrackerParameters params,
                                 double* stream_data,
                                 double arclength) noexcept nogil


cdef double calculate_ptt_data_support(TrackerParameters params,
                                       double* stream_data,
                                       PmfGen pmf_gen) noexcept nogil