# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: Nonecheck=False

from dipy.tracking.propspeed cimport (
    deterministic_propagator,
    probabilistic_propagator,
    parallel_transport_propagator,
)
from dipy.tracking.utils import min_radius_curvature_from_angle

import numpy as np
cimport numpy as cnp


def generate_tracking_parameters(algo_name, *,
    int max_len=500, int min_len=2, double step_size=0.2, double[:] voxel_size,
    double max_angle=20, bint return_all=True, double pmf_threshold=0.1,
    double probe_length=0.5, double probe_radius=0, int probe_quality=3,
    int probe_count=1, double data_support_exponent=1, int random_seed=0):

    cdef TrackerParameters params

    algo_name = algo_name.lower()

    if algo_name in ['deterministic', 'det']:
        params = TrackerParameters(max_len=max_len,
                                   min_len=min_len,
                                   step_size=step_size,
                                   voxel_size=voxel_size,
                                   pmf_threshold=pmf_threshold,
                                   max_angle=max_angle,
                                   random_seed=random_seed,
                                   return_all=return_all)
        params.set_tracker_c(deterministic_propagator)
        return params
    elif algo_name in ['probabilistic', 'prob']:
        params = TrackerParameters(max_len=max_len,
                                   min_len=min_len,
                                   step_size=step_size,
                                   voxel_size=voxel_size,
                                   pmf_threshold=pmf_threshold,
                                   max_angle=max_angle,
                                   random_seed=random_seed,
                                   return_all=return_all)
        params.set_tracker_c(probabilistic_propagator)
        return params
    elif algo_name == 'ptt':
        params = TrackerParameters(max_len=max_len,
                                   min_len=min_len,
                                   step_size=step_size,
                                   voxel_size=voxel_size,
                                   pmf_threshold=pmf_threshold,
                                   max_angle=max_angle,
                                   probe_length=probe_length,
                                   probe_radius=probe_radius,
                                   probe_quality=probe_quality,
                                   probe_count=probe_count,
                                   data_support_exponent=data_support_exponent,
                                   random_seed=random_seed,
                                   return_all=return_all)
        params.set_tracker_c(parallel_transport_propagator)
        return params
    else:
        raise ValueError("Invalid algorithm name")


cdef class TrackerParameters:

    def __init__(self, max_len, min_len, step_size, voxel_size,
                 max_angle, return_all, pmf_threshold=None, probe_length=None,
                 probe_radius=None, probe_quality=None, probe_count=None,
                 data_support_exponent=None, random_seed=None):
        cdef cnp.npy_intp i

        self.max_nbr_pts = int(max_len/step_size)
        self.min_nbr_pts = int(min_len/step_size)
        self.return_all = return_all
        self.random_seed = random_seed
        self.step_size = step_size
        self.average_voxel_size = 0
        for i in range(3):
            self.voxel_size[i] = voxel_size[i]
            self.inv_voxel_size[i] = 1. / voxel_size[i]
            self.average_voxel_size += voxel_size[i] / 3

        self.max_angle = np.deg2rad(max_angle)
        self.cos_similarity = np.cos(self.max_angle)
        self.max_curvature = 1 / min_radius_curvature_from_angle(
            self.max_angle,
            self.step_size / self.average_voxel_size)

        self.sh = None
        self.ptt = None

        if pmf_threshold is not None:
            self.sh = ShTrackerParameters(pmf_threshold)

        if probe_length is not None and probe_radius is not None and probe_quality is not None and probe_count is not None and data_support_exponent is not None:
            self.ptt = ParallelTransportTrackerParameters(probe_length, probe_radius, probe_quality, probe_count, data_support_exponent)

    cdef void set_tracker_c(self, func_ptr tracker) noexcept nogil:
        self.tracker = tracker


cdef class ShTrackerParameters:

    def __init__(self, pmf_threshold):
        self.pmf_threshold = pmf_threshold

cdef class ParallelTransportTrackerParameters:

    def __init__(self, double probe_length, double probe_radius,
                int probe_quality, int probe_count, double data_support_exponent):
        self.probe_length = probe_length
        self.probe_radius = probe_radius
        self.probe_quality = probe_quality
        self.probe_count = probe_count
        self.data_support_exponent = data_support_exponent

        self.probe_step_size = self.probe_length / (self.probe_quality - 1)
        self.probe_normalizer = 1.0 / (self.probe_quality * self.probe_count)
        self.k_small = 0.0001

        # Adaptively set in Trekker
        self.rejection_sampling_nbr_sample = 10
        self.rejection_sampling_max_try = 100
