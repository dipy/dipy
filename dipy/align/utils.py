from enum import IntEnum

import numpy as np

FLOAT_DTYPE = np.float32


class RegistrationStages(IntEnum):
    """Callback stages for volumetric registration.

    The stage value is passed to the callback function so it can react
    accordingly.

    INIT_START : optimizer initialization starts
    INIT_END : optimizer initialization ends
    OPT_START : optimization starts
    OPT_END : optimization ends
    SCALE_START : optimization at a new scale space resolution starts
    SCALE_END : optimization at the current scale space resolution ends
    ITER_START : a new iteration starts
    ITER_END : the current iteration ends
    """

    INIT_START = 0
    INIT_END = 1
    OPT_START = 2
    OPT_END = 3
    SCALE_START = 4
    SCALE_END = 5
    ITER_START = 6
    ITER_END = 7
