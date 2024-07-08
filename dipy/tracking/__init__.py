# Init for tracking module
"""Tracking objects"""

import lazy_loader as lazy
from nibabel.streamlines import ArraySequence as Streamlines

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "direction_getter",
        "distances",
        "fbcmeasures",
        "learning",
        "life",
        "local_tracking",
        "localtrack",
        "mesh",
        "metrics",
        "propspeed",
        "stopping_criterion",
        "streamline",
        "streamlinespeed",
        "utils",
        "vox2track",
    ],
)

__all__ = [
    "direction_getter",
    "distances",
    "fbcmeasures",
    "learning",
    "life",
    "local_tracking",
    "localtrack",
    "mesh",
    "metrics",
    "propspeed",
    "stopping_criterion",
    "streamline",
    "Streamlines",
    "streamlinespeed",
    "utils",
    "vox2track",
]
