import numpy as np
floating  = np.float32

def enum(**enums):
    return type('Enum', (), enums)

VerbosityLevels = enum(NONE=0, STATUS=1, DIAGNOSE=2, DEBUG=3)
r""" VerbosityLevels
This enum defines the four levels of verbosity we use in the align
module.
NONE : do not print anything
STATUS : print information about the current status of the algorithm
DIAGNOSE : print high level information of the components involved in the
		   registration that can be used to detect a failing component.
DEBUG : print as much information as possible to isolate the cause of a bug.
"""


