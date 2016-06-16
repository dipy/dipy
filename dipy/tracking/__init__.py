# Init for tracking module
""" Tracking objects """
from distutils.version import LooseVersion

# Test callable
from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
del Tester

import nibabel

NIBABEL_LESS_2_1 = LooseVersion(nibabel.__version__) < '2.1'

if not NIBABEL_LESS_2_1:
    from nibabel.streamlines import ArraySequence as Streamlines
