''' Utilities for testing '''
from os.path import dirname, abspath, join as pjoin
from dipy.testing.spherepoints import sphere_points
from dipy.testing.decorators import doctest_skip_parser
from numpy.testing import assert_array_equal
import numpy as np
import scipy
from distutils.version import LooseVersion

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


def setup_test():
    """ Set numpy print options to "legacy" for new versions of numpy

    If imported into a file, nosetest will run this before any doctests.

    References
    -----------
    https://github.com/numpy/numpy/commit/710e0327687b9f7653e5ac02d222ba62c657a718
    https://github.com/numpy/numpy/commit/734b907fc2f7af6e40ec989ca49ee6d87e21c495
    https://github.com/nipy/nibabel/pull/556
    """
    if LooseVersion(np.__version__) >= LooseVersion('1.14'):
        np.set_printoptions(legacy='1.13')

    # Temporary fix until scipy release in October 2018
    # must be removed after that
    # print the first occurrence of matching warnings for each location
    # (module + line number) where the warning is issued
    if LooseVersion(np.__version__) >= LooseVersion('1.15') and \
            LooseVersion(scipy.version.short_version) <= '1.1.0':
        import warnings
        warnings.simplefilter("default")
