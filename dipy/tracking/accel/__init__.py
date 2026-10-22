"""Accelerated probabilistic tracking backends (CUDA, Metal, WebGPU).

See :func:`dipy.tracking.accel.tracker.\
prepare_gpu_tracker_data` for the assumptions these trackers make.
"""

from dipy.tracking.accel.tracker import (
    ACCEL_BACKENDS,
    accel_backend_available,
    gpu_streamline_generator,
    prepare_gpu_tracker_data,
)

__all__ = [
    "ACCEL_BACKENDS",
    "accel_backend_available",
    "gpu_streamline_generator",
    "prepare_gpu_tracker_data",
]
