''' Load some modules
'''
import os

from .info import __version__, long_description as __doc__

import align
import reconst
import io
import tracking
import viz
import external
import core

'''
try:
    from nibabel.nicom.dicomreaders import read_mosaic_dir as load_dcm_dir
except ImportError:
    pass
'''

#    raise ImportError('nibabel.nicom.dicomreaders cannot be found')

# Test callable
from numpy.testing import Tester
test = Tester().test
del Tester

# Plumb in version etc info stuff
from .pkg_info import get_pkg_info as _get_pkg_info
get_info = lambda : _get_pkg_info(os.path.dirname(__file__))
