''' Utilities for testing '''
from os.path import dirname, abspath, join as pjoin

# set path to example data
IO_DATA_PATH = abspath(pjoin(dirname(__file__),
                             '..', 'io', 'tests', 'data'))

from .spherepoints import sphere_points

# Allow failed import of nose if not now running tests
try:
    import nose.tools as nt
except ImportError:
    pass
else:
    from lightunit import ParametricTestCase, parametric
    from nose.tools import (assert_equal, assert_not_equal,
                            assert_true, assert_false, assert_raises)
    

