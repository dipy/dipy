cimport numpy as cnp

from dipy.direction.pmf cimport PmfGen
from dipy.tracking.tracker_parameters cimport TrackerParameters, TrackerStatus
from dipy.utils.fast_numpy cimport RNGState


cdef TrackerStatus deterministic_propagator(double* point,
                                            double* direction,
                                            TrackerParameters params,
                                            double* stream_data,
                                            PmfGen pmf_gen,
                                            RNGState* rng) noexcept nogil


cdef TrackerStatus probabilistic_propagator(double* point,
                                            double* direction,
                                            TrackerParameters params,
                                            double* stream_data,
                                            PmfGen pmf_gen,
                                            RNGState* rng) noexcept nogil

cdef TrackerStatus parallel_transport_propagator(double* point,
                                                 double* direction,
                                                 TrackerParameters params,
                                                 double* stream_data,
                                                 PmfGen pmf_gen,
                                                 RNGState* rng) noexcept nogil



cdef cnp.npy_intp _propagation_direction(double *point, double* prev, double* qa,
                                double *ind, double *odf_vertices,
                                double qa_thr, double ang_thr,
                                cnp.npy_intp *qa_shape,cnp.npy_intp* strides,
                                double *direction,double total_weight) noexcept nogil
