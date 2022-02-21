""" Testing tripwire module.
"""

from dipy.utils.tripwire import TripWire, is_tripwire, TripWireError

from dipy.testing import assert_true, assert_false
from numpy.testing import assert_raises


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
    assert_raises(TripWireError,
                  silly_module_name)
    # Check AttributeError can be checked too
    try:
        silly_module_name.__wrapped__
    except TripWireError as err:
        assert_true(isinstance(err, AttributeError))
    else:
        raise RuntimeError("No error raised, but expected")
