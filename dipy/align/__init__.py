import warnings

from dipy.align._public import (
    affine,
    affine_registration,
    center_of_mass,
    motion_correction,
    read_mapping,
    register_dwi_series,
    register_dwi_to_template,
    register_series,
    resample,
    rigid,
    rigid_isoscaling,
    rigid_scaling,
    streamline_registration,
    syn_registration,
    translation,
    write_mapping,
)

__all__ = [
    "syn_registration",
    "register_dwi_to_template",
    "write_mapping",
    "read_mapping",
    "resample",
    "center_of_mass",
    "translation",
    "rigid_isoscaling",
    "rigid_scaling",
    "rigid",
    "affine",
    "motion_correction",
    "affine_registration",
    "register_series",
    "register_dwi_series",
    "streamline_registration",
]


def __getattr__(name):
    if name == "floating":
        warnings.warn(
            "Importing 'floating' from dipy.align is deprecated. "
            "Use np.float32 directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        import numpy as np

        return np.float32
    if name == "VerbosityLevels":
        warnings.warn(
            "Importing 'VerbosityLevels' from dipy.align is deprecated. "
            "Use 'from dipy.utils import VerbosityLevels'.",
            DeprecationWarning,
            stacklevel=2,
        )
        from dipy.utils import VerbosityLevels

        return VerbosityLevels
    if name == "Bunch":
        raise ImportError(
            "'Bunch' has been removed from dipy.align. Use enum.IntEnum instead."
        )
    raise AttributeError(f"module 'dipy.align' has no attribute {name!r}")
