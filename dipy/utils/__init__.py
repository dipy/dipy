# code support utilities for dipy
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "arrfuncs",
        "convert",
        "deprecator",
        "fast_numpy",
        "multiproc",
        "omp",
        "optpkg",
        "parallel",
        "tractogram",
        "tripwire",
        "volume",
    ],
)

__all__ = [
    "arrfuncs",
    "convert",
    "deprecator",
    "fast_numpy",
    "multiproc",
    "omp",
    "optpkg",
    "parallel",
    "tractogram",
    "tripwire",
    "volume",
]
