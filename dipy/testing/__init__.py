"""Utilities for testing."""
import operator
from functools import partial
from os.path import dirname, abspath, join as pjoin
from numpy.testing import assert_array_equal
import numpy as np
import scipy
from distutils.version import LooseVersion
import warnings

# set path to example data
IO_DATA_PATH = abspath(pjoin(dirname(__file__),
                             '..', 'io', 'tests', 'data'))


def assert_operator(value1, value2, msg="", op=operator.eq):
    """Check Boolean statement."""
    try:
        if op == operator.is_:
                value1 = bool(value1)
        assert op(value1, value2)
    except AssertionError:
        raise AssertionError(msg.format(str(value2), str(value1)))


assert_greater_equal = partial(assert_operator, op=operator.ge,
                               msg="{0} >= {1}")
assert_greater = partial(assert_operator, op=operator.gt,
                         msg="{0} > {1}")
assert_less_equal = partial(assert_operator, op=operator.le,
                            msg="{0} =< {1}")
assert_less = partial(assert_operator, op=operator.lt,
                      msg="{0} < {1}")
assert_true = partial(assert_operator, value2=True, op=operator.is_,
                      msg="False is not true")
assert_false = partial(assert_operator, value2=False, op=operator.is_,
                       msg="True is not false")
assert_not_equal = partial(assert_operator, op=operator.ne)


def assert_arrays_equal(arrays1, arrays2):
    for arr1, arr2 in zip(arrays1, arrays2):
        assert_array_equal(arr1, arr2)


def setup_test():
    """ Set numpy print options to "legacy" for new versions of numpy

    If imported into a file, pytest will run this before any doctests.

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
        warnings.simplefilter(action="default", category=FutureWarning)

    warnings.simplefilter("always", category=UserWarning)
