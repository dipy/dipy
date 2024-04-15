from dipy.tracking.tracker_deterministic cimport deterministic_tracker
from dipy.tracking.tracker_probabilistic cimport probabilistic_tracker
from dipy.tracking.tracker_ptt cimport parallel_transport_tracker

def generate_tracking_parameters(algo_name, *,
    int max_len=500, double step_size=0.5, double[:] voxel_size,
    double max_angle=30, double pmf_threshold=0.1, double probe_length=0.5,
    double probe_radius=0, int probe_quality=3, int probe_count=1,
    double data_support_exponent=1):

    algo_name = algo_name.lower()

    if algo_name in ['deterministic', 'det']:
        return TrackingParameters(tracker=&deterministic_maximum_tracker,
                                  max_len=max_len, step_size=step_size,
                                  voxel_size=voxel_size)
    elif algo_name in ['probabilistic', 'prob']:
        return TrackingParameters(tracker=&probabilistic_tracker,
                                  max_len=max_len, step_size=step_size,
                                  voxel_size=voxel_size,
                                  pmf_threshold=pmf_threshold,
                                  max_angle=max_angle)
    elif algo_name == 'ptt':
        return TrackingParameters(tracker=&probabilistic_tracker,
                                  max_len=max_len, step_size=step_size,
                                  voxel_size=voxel_size,
                                  probe_length=probe_length,
                                  probe_radius=probe_radius,
                                  probe_quality=probe_quality,
                                  probe_count=probe_count,
                                  data_support_exponent=data_support_exponent)
    #elif algo_name == 'eudx':
    #    return TrackingParameters(tracker=euDX_tracker,
    #                              max_len=max_len, step_size=step_size,
    #                              voxel_size=voxel_size)
    else:
        raise ValueError("Invalid algorithm name")



cdef class TrackingParameters:

    def __init__(self, tracker, max_len, step_size, voxel_size,
                 max_angle, pmf_threshold=None, probe_length=None,
                 probe_radius=None, probe_quality=None, probe_count=None,
                 data_support_exponent=None):
        cdef cnp.npy_intp i

        self.tracker = tracker
        self.max_len = max_len
        self.step_size = step_size
        self.average_voxel_size = 0
        for i in range(3):
            self.voxel_size[i] = voxel_size[i]
            self.inv_voxel_size[i] = 1. / voxel_size[i]
            self.average_voxel_size += voxel_size[i] / 3

        self.max_angle = np.deg2rad(self.max_angle)
        self.cos_similarity = np.cos(self.max_angle)
        self.max_curvature = 1 / min_radius_curvature_from_angle(
            self.max_angle,
            self.step_size / self.average_voxel_size)

        self.probabilistic = None
        self.ptt = None

        if pmf_threshold is not None:
            self.probabilistic = ShTrackingParameters(pmf_threshold)

        if probe_length is not None and probe_radius is not None and probe_quality is not None and probe_count is not None and data_support_exponent is not None:
            self.ptt = PttTrackingParameters(probe_length, probe_radius, probe_quality, probe_count, data_support_exponent)


cdef class ShTrackingParameters:

    def __init__(self, pmf_threshold):
        self.pmf_threshold = pmf_threshold