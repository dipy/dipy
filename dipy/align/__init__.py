import numpy as np
floating = np.float32


class Bunch:
    def __init__(self, **kwds):
        r"""A 'bunch' of values (a replacement of Enum)

        This is a temporary replacement of Enum, which is not available
        on all versions of Python 2
        """
        self.__dict__.update(kwds)


VerbosityLevels = Bunch(NONE=0, STATUS=1, DIAGNOSE=2, DEBUG=3)
r""" VerbosityLevels
This enum defines the four levels of verbosity we use in the align
module.
NONE : do not print anything
STATUS : print information about the current status of the algorithm
DIAGNOSE : print high level information of the components involved in the
registration that can be used to detect a failing component.
DEBUG : print as much information as possible to isolate the cause of a bug.
"""

from dipy.align._public import (syn_registration, register_dwi_to_template, # noqa
                                write_mapping, read_mapping, resample,
                                center_of_mass, translation,
                                rigid_isoscaling, rigid_scaling,
                                rigid, affine, motion_correction,
                                affine_registration, register_series,
                                register_dwi_series, streamline_registration)

__all__ = ["syn_registration", "register_dwi_to_template",
           "write_mapping", "read_mapping", "resample",
           "center_of_mass", "translation",
           "rigid_isoscaling", "rigid_scaling",
           "rigid", "affine", "motion_correction",
           "affine_registration", "register_series",
           "register_dwi_series", "streamline_registration"]
