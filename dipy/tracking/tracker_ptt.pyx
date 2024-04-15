cdef class PttTrackingParameters:

    def __init__(self, double probe_length, double probe_radius,
                int probe_quality, int probe_count, double data_support_exponent):
        self.probe_length = probe_length
        self.probe_radius = probe_radius
        self.probe_quality = probe_quality
        self.probe_count = probe_count
        self.data_support_exponent = data_support_exponent


cdef int parallel_transport_tracker(double* point,
                                    double* direction,
                                    TrackingParameters params,
                                    PmfGen pmf_gen) noexcept nogil:
    # update point and dir with new position and direction



    # return 1 if the propagation failed.

    return 0