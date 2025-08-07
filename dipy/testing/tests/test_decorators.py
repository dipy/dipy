"""Testing decorators module."""

import warnings

from numpy.testing import assert_equal, assert_raises

import dipy
from dipy.testing import assert_true
from dipy.testing.decorators import doctest_skip_parser, warning_for_keywords


def test_skipper():
    def f():
        pass

    docstring = """ Header

        >>> something # skip if not HAVE_AMODULE
        >>> something + else
        >>> a = 1 # skip if not HAVE_BMODULE
        >>> something2   # skip if HAVE_AMODULE
        """
    f.__doc__ = docstring
    global HAVE_AMODULE, HAVE_BMODULE
    HAVE_AMODULE = False
    HAVE_BMODULE = True
    f2 = doctest_skip_parser(f)
    assert_true(f is f2)
    assert_equal(
        f2.__doc__,
        """ Header

        >>> something # doctest: +SKIP
        >>> something + else
        >>> a = 1
        >>> something2
        """,
    )
    HAVE_AMODULE = True
    HAVE_BMODULE = False
    f.__doc__ = docstring
    f2 = doctest_skip_parser(f)
    assert_true(f is f2)
    assert_equal(
        f2.__doc__,
        """ Header

        >>> something
        >>> something + else
        >>> a = 1 # doctest: +SKIP
        >>> something2   # doctest: +SKIP
        """,
    )
    del HAVE_AMODULE
    f.__doc__ = docstring
    assert_raises(NameError, doctest_skip_parser, f)


def test_warning_for_keywords():
    original_version = dipy.__version__

    @warning_for_keywords(from_version="1.0.0", until_version="10.0.0")
    def func_with_kwonly_args(a, b, *, c=3):
        return a + b + c

    # Case 1: Version is 0.9.0, no warnings expected
    dipy.__version__ = "0.9.0"
    assert func_with_kwonly_args(1, 2, c=3) == 6
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = func_with_kwonly_args(1, 2, 3)
        assert result == 6, "Expected result to be 6"
        assert len(w) == 0, "Expected no warnings for version 0.9.0"

    # Case 2: Version is 1.10.0, warnings expected
    dipy.__version__ = "1.10.0"
    assert func_with_kwonly_args(1, 2, c=3) == 6
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = func_with_kwonly_args(1, 2, 3)
        assert result == 6, "Expected result to be 6"
        assert len(w) == 1, "Expected warning for version 1.10.0"
        assert (
            "Pass ['c'] as keyword args. From version 10.0.0 passing these as "
            "positional arguments will result in an error" in str(w[-1].message)
        )

    # Case 3: Version is 1.15.0, warnings expected
    dipy.__version__ = "1.15.0"
    assert func_with_kwonly_args(1, 2, c=3) == 6
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = func_with_kwonly_args(1, 2, 3)
        assert result == 6, "Expected result to be 6"
        assert len(w) == 1, "Expected warning for version 1.15.0"
        assert (
            "Pass ['c'] as keyword args. From version 10.0.0 passing these as "
            "positional arguments will result in an error" in str(w[-1].message)
        )

    # Case 4: Version is 10.1.0, arguments should pass as they are,
    # expecting TypeError from system
    dipy.__version__ = "10.1.0"
    assert func_with_kwonly_args(1, 2, c=3) == 6
    try:
        result = func_with_kwonly_args(1, 2, 3)
    except TypeError:
        pass

    # Case 5: Version is a pre-release like '2.0.0rc1', warnings expected
    dipy.__version__ = "2.0.0rc1"
    assert func_with_kwonly_args(1, 2, c=3) == 6
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = func_with_kwonly_args(1, 2, 3)
        assert result == 6, "Expected result to be 6"
        assert len(w) == 1, "Expected warning for version 2.0.0rc1"
        assert (
            "Pass ['c'] as keyword args. From version 10.0.0 passing these as "
            "positional arguments will result in an error" in str(w[-1].message)
        )

    # Case 6: Version is a dev release like '1.10.0.dev1', warnings expected
    dipy.__version__ = "1.10.0.dev1"
    assert func_with_kwonly_args(1, 2, c=3) == 6
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = func_with_kwonly_args(1, 2, 3)
        assert result == 6, "Expected result to be 6"
        assert len(w) == 1, "Expected warning for version 1.10.0.dev1"
        assert (
            "Pass ['c'] as keyword args. From version 10.0.0 passing these as "
            "positional arguments will result in an error" in str(w[-1].message)
        )

    # Restore the original version
    dipy.__version__ = original_version
