"""Testing decorators module."""

import inspect
import warnings

import numpy as np
from numpy.testing import assert_equal, assert_raises

import dipy
from dipy.testing import assert_true
from dipy.testing.decorators import doctest_skip_parser, set_random_number_generator
from dipy.utils import deprecator
from dipy.utils.deprecator import warning_for_keywords


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


def test_warning_for_keywords(monkeypatch):
    @warning_for_keywords(from_version="1.0.0", until_version="10.0.0")
    def func_with_kwonly_args(a, b, *, c=3):
        return a + b + c

    # Case 1: Version is 0.9.0, no warnings expected
    monkeypatch.setattr(dipy, "__version__", "0.9.0")
    assert func_with_kwonly_args(1, 2, c=3) == 6
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = func_with_kwonly_args(1, 2, 3)
        assert result == 6, "Expected result to be 6"
        assert len(w) == 0, "Expected no warnings for version 0.9.0"

    # Case 2: Version is 1.10.0, warnings expected
    monkeypatch.setattr(dipy, "__version__", "1.10.0")
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
    monkeypatch.setattr(dipy, "__version__", "1.15.0")
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
    monkeypatch.setattr(dipy, "__version__", "10.1.0")
    assert func_with_kwonly_args(1, 2, c=3) == 6
    assert_raises(TypeError, func_with_kwonly_args, 1, 2, 3)

    # Case 5: Version is a pre-release like '2.0.0rc1', warnings expected
    monkeypatch.setattr(dipy, "__version__", "2.0.0rc1")
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
    monkeypatch.setattr(dipy, "__version__", "1.10.0.dev1")
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


def test_warning_for_keywords_leaves_varargs_alone():
    """``*args`` absorbs surplus positional arguments, so nothing is moved.

    Converting them would rename a genuine variadic argument after the first
    keyword-only parameter and silently drop the model's ``*args``. A call
    that leaves such a parameter at its default still has to be reported: a
    keyword-only parameter that quietly keeps its default is a wrong result,
    not an error the caller can see.
    """

    @warning_for_keywords(from_version="1.0.0", until_version="10.0.0")
    def func_with_varargs(a, *args, b=1, **kwargs):
        return a, args, b, kwargs

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert_equal(func_with_varargs(1, 2, 3), (1, (2, 3), 1, {}))
        assert_equal(len(w), 1)
        assert_true("absorbed by '*args'" in str(w[-1].message))
        assert_true("['b']" in str(w[-1].message))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert_equal(func_with_varargs(1, b=5), (1, (), 5, {}))
        assert_equal(len(w), 0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert_equal(func_with_varargs(1, 2, b=5), (1, (2,), 5, {}))
        assert_equal(len(w), 0)


def test_warning_for_keywords_varargs_reports_only_shadowed_parameters():
    """Only the parameters the surplus could have bound to are named.

    ``KurtosisMicrostructureModel`` forwards ``(gtab, *args)`` to
    ``DiffusionKurtosisModel.__init__`` with ``fit_method`` already named:
    reporting ``return_S0_hat`` there would blame a parameter no positional
    argument ever reached, on a call the user did not write.
    """

    @warning_for_keywords(from_version="1.0.0", until_version="10.0.0")
    def func_with_varargs(a, *args, b=1, c=2):
        return a, args, b, c

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert_equal(func_with_varargs(1, 2), (1, (2,), 1, 2))
        assert_equal(len(w), 1)
        assert_true("['b']" in str(w[-1].message))
        assert_true("c" not in str(w[-1].message))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert_equal(func_with_varargs(1, 2, 3), (1, (2, 3), 1, 2))
        assert_equal(len(w), 1)
        assert_true("['b', 'c']" in str(w[-1].message))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert_equal(func_with_varargs(1, 2, 3, b=9), (1, (2, 3), 9, 2))
        assert_equal(len(w), 1)
        assert_true("['c']" in str(w[-1].message))


def test_warning_for_keywords_varargs_report_expires(monkeypatch):
    """Past ``until_version`` the variadic call runs without a word."""

    @warning_for_keywords(from_version="1.0.0", until_version="10.0.0")
    def func_with_varargs(a, *args, b=1):
        return a, args, b

    monkeypatch.setattr(dipy, "__version__", "10.1.0")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert_equal(func_with_varargs(1, 2), (1, (2,), 1))
        assert_equal(len(w), 0)


def test_warning_for_keywords_varargs_without_keyword_only_is_silent():
    """No keyword-only parameter to miss, so a variadic call is ordinary."""

    @warning_for_keywords(from_version="1.0.0", until_version="10.0.0")
    def func_with_varargs(a, *args):
        return a, args

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert_equal(func_with_varargs(1, 2, 3), (1, (2, 3)))
        assert_equal(len(w), 0)


def test_warning_for_keywords_does_not_drop_surplus_arguments():
    """More positional arguments than keyword-only parameters is a ``TypeError``.

    Truncating them instead would swallow the extra argument without a word.
    """

    @warning_for_keywords(from_version="1.0.0", until_version="10.0.0")
    def func_with_kwonly_args(a, *, b=2):
        return a + b

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_raises(TypeError, func_with_kwonly_args, 1, 2, 3)


def test_warning_for_keywords_positional_versions(monkeypatch):
    """``from_version``/``until_version`` used to be positional-or-keyword."""
    # Pinned below ``_POSITIONAL_VERSIONS_REMOVED``, where the factory raises
    # instead of warning; otherwise this test expires with the spelling.
    monkeypatch.setattr(dipy, "__version__", "1.13.0")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        @warning_for_keywords("1.0.0", "10.0.0")
        def func_with_kwonly_args(a, *, b=2):
            return a + b

        assert_equal(len(w), 1)
        assert_true("Pass ['from_version', 'until_version']" in str(w[-1].message))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert_equal(func_with_kwonly_args(1, 5), 6)
        assert_equal(len(w), 1)
        assert_true("From version 10.0.0" in str(w[-1].message))

    assert_raises(TypeError, warning_for_keywords, "1.0.0", "10.0.0", "2.0.0")


def test_warning_for_keywords_does_not_shadow_an_explicit_keyword():
    """A name passed both ways must not lose the value the caller named.

    Overwriting it with the positional one would hand the function a different
    argument than the caller wrote, without a word.
    """

    @warning_for_keywords(from_version="1.0.0", until_version="10.0.0")
    def func_with_kwonly_args(a, *, b=2):
        return a + b

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_raises(TypeError, func_with_kwonly_args, 1, 5, b=7)


def test_warning_for_keywords_positional_versions_expire(monkeypatch):
    """Past the announced version, positional versions stop being accepted."""
    monkeypatch.setattr(dipy, "__version__", deprecator._POSITIONAL_VERSIONS_REMOVED)
    assert_raises(TypeError, warning_for_keywords, "1.0.0", "10.0.0")


def test_warning_for_keywords_rejects_bare_use():
    """``@warning_for_keywords`` without parentheses is a silent no-op trap."""

    def func_with_kwonly_args(a, *, b=2):
        return a + b

    assert_raises(TypeError, warning_for_keywords, func_with_kwonly_args)


@set_random_number_generator(1234)
def test_set_random_number_generator_seeds_rng(rng=None):
    assert_true(isinstance(rng, np.random.Generator))
    assert_equal(rng.random(), np.random.default_rng(1234).random())


@set_random_number_generator()
def test_set_random_number_generator_with_tmp_path_fixture(tmp_path, rng):
    # `rng` deliberately carries no default value here: the decorator supplies
    # it, so `pytest` must resolve `tmp_path` and leave `rng` alone.
    assert_true(isinstance(rng, np.random.Generator))
    assert_true(tmp_path.is_dir())


@set_random_number_generator()
def test_set_random_number_generator_with_pytestconfig_fixture(pytestconfig, rng):
    assert_true(isinstance(rng, np.random.Generator))
    assert_true(pytestconfig is not None)


def test_set_random_number_generator_hides_rng_from_signature():
    @set_random_number_generator()
    def func(tmp_path, rng):
        return tmp_path, rng

    parameters = inspect.signature(func).parameters
    assert_true("tmp_path" in parameters)
    assert_true("rng" not in parameters)


def test_warning_for_keywords_old_import_path_is_deprecated():
    """``dipy.testing.decorators`` re-exports it, but using it warns."""
    import dipy.testing.decorators as decorators

    aliased = decorators.warning_for_keywords
    assert_true(aliased is not warning_for_keywords)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        @aliased(from_version="1.0.0", until_version="10.0.0")
        def func_with_kwonly_args(a, *, b=2):
            return a + b

        assert_equal(len(w), 1)
        assert_true(issubclass(w[-1].category, DeprecationWarning))
        assert_true("dipy.utils.deprecator" in str(w[-1].message))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert_equal(func_with_kwonly_args(1, 5), 6)
        assert_equal(len(w), 1)
        assert_true("Pass ['b'] as keyword args" in str(w[-1].message))


def test_decorators_module_rejects_unknown_attributes():
    """The ``__getattr__`` shim must not swallow genuine typos."""
    import dipy.testing.decorators as decorators

    assert_raises(AttributeError, getattr, decorators, "no_such_decorator")
