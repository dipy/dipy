from numpy.testing import assert_equal
import pytest

from dipy.testing.memory import get_type_refcount


def test_get_type_refcount():
    with pytest.warns(DeprecationWarning, match="get_type_refcount is deprecated"):
        list_ref_count = get_type_refcount(pattern="list")
    A = []  # noqa: F841
    with pytest.warns(DeprecationWarning, match="get_type_refcount is deprecated"):
        with_a = get_type_refcount(pattern="list")
    assert_equal(with_a["list"], list_ref_count["list"] + 1)
    del A
    with pytest.warns(DeprecationWarning, match="get_type_refcount is deprecated"):
        without_a = get_type_refcount(pattern="list")
    assert_equal(without_a["list"], list_ref_count["list"])
