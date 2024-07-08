# Init for core dipy objects
"""Core objects"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "geometry",
        "gradients",
        "graph",
        "histeq",
        "interpolation",
        "ndindex",
        "onetime",
        "optimize",
        "profile",
        "rng",
        "sphere",
        "sphere_stats",
        "subdivide_octahedron",
        "wavelet",
    ],
)

__all__ += [
    "geometry",
    "gradients",
    "graph",
    "histeq",
    "interpolation",
    "ndindex",
    "onetime",
    "optimize",
    "profile",
    "rng",
    "sphere",
    "sphere_stats",
    "subdivide_octahedron",
    "wavelet",
]
