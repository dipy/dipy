''' Utilities for testing '''
from os.path import dirname, abspath, join as pjoin
from dipy.testing.spherepoints import sphere_points
from dipy.testing.decorators import doctest_skip_parser
from numpy.testing import assert_array_equal

# set path to example data
IO_DATA_PATH = abspath(pjoin(dirname(__file__),
                             '..', 'io', 'tests', 'data'))

# Allow failed import of nose if not now running tests
try:
    import nose.tools as nt
except ImportError:
    pass
else:
    from nose.tools import (assert_equal, assert_not_equal,
                            assert_true, assert_false, assert_raises)


def assert_arrays_equal(arrays1, arrays2):
    for arr1, arr2 in zip(arrays1, arrays2):
        assert_array_equal(arr1, arr2)
