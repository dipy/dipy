from numpy.testing import assert_equal
import pytest

from dipy.testing.memory import _get_type_refcount, get_type_refcount


def test_get_type_refcount():
    list_ref_count = _get_type_refcount(pattern="list")
    A = []  # noqa: F841
    assert_equal(_get_type_refcount(pattern="list")["list"], list_ref_count["list"] + 1)
    del A
    assert_equal(_get_type_refcount(pattern="list")["list"], list_ref_count["list"])


def test_get_type_refcount_deprecated():
    with pytest.warns(DeprecationWarning, match="get_type_refcount is deprecated"):
        refcounts = get_type_refcount(pattern="list")
    assert refcounts["list"] > 0
