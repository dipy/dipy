"""Simplified ("simplet") GPU/CPU probabilistic tracking backends.

Ported from GPUStreamlines. See :func:`dipy.tracking.simplet.tracker.\
prepare_simple_tracker_data` for the assumptions these trackers make.
"""

from dipy.tracking.simplet.tracker import (
    SIMPLE_BACKENDS,
    prepare_simple_tracker_data,
    simple_backend_available,
    simple_sl_generator,
)

__all__ = [
    "SIMPLE_BACKENDS",
    "prepare_simple_tracker_data",
    "simple_backend_available",
    "simple_sl_generator",
]
