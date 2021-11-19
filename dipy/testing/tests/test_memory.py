from numpy.testing import assert_equal

from dipy.testing.memory import get_type_refcount


def test_get_type_refcount():
    list_ref_count = get_type_refcount("list")
    A = list()
    assert_equal(get_type_refcount("list")["list"], list_ref_count["list"]+1)
    del A
    assert_equal(get_type_refcount("list")["list"], list_ref_count["list"])
