""" Testing tripwire module.
"""

from ..tripwire import TripWire, is_tripwire, TripWireError

from nose import SkipTest
from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)


def test_is_tripwire():
    assert_false(is_tripwire(object()))
    assert_true(is_tripwire(TripWire('some message')))


def test_tripwire():
    # Test tripwire object
    silly_module_name = TripWire('We do not have silly_module_name')
    assert_raises(TripWireError,
                  getattr,
                  silly_module_name,
                  'do_silly_thing')
    # Check AttributeError can be checked too
    try:
        silly_module_name.__wrapped__
    except TripWireError as err:
        assert_true(isinstance(err, AttributeError))
    else:
        raise RuntimeError("No error raised, but expected")
