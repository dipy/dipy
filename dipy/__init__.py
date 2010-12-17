''' Load some modules
'''
import os

import dipy.core.track_metrics as track_metrics
import dipy.core.track_propagation as track_propagation
import dipy.core.track_performance as track_performance
import dipy.io.track_volumes as track_volumes

from dipy.core.generalized_q_sampling import GeneralizedQSampling
from dipy.core.dti import Tensor
from dipy.core.track_propagation import FACT_Delta
    
try:
    from nibabel import load, save
except ImportError:
    raise ImportError('nibabel is not installed')

try:
    from nibabel.nicom.dicomreaders import read_mosaic_dir as load_dcm_dir
except ImportError:
    pass
#    raise ImportError('nibabel.nicom.dicomreaders cannot be found')

# Test callable
from numpy.testing import Tester
test = Tester().test
del Tester

# Plumb in version etc info stuff
from .pkg_info import get_pkg_info as _get_pkg_info
get_info = lambda : _get_pkg_info(os.path.dirname(__file__))
