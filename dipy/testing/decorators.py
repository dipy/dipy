"""
Decorators for dipy tests
"""

from functools import wraps
import importlib
import inspect
import os
import platform
import re

import numpy as np

from dipy.utils.deprecator import deprecate_with_version

_NEW_LOCATIONS = {"warning_for_keywords": "dipy.utils.deprecator"}

__all__ = [
    "doctest_skip_parser",
    "is_linux",
    "is_macOS",
    "is_windows",
    "set_random_number_generator",
    "use_xvfb",
    "xvfb_it",
] + list(_NEW_LOCATIONS)


def __getattr__(name):
    """Resolve a deprecated re-export from its new location.

    Parameters
    ----------
    name : str
        Name of the attribute being looked up on this module.

    Returns
    -------
    callable
        The re-exported callable, wrapped so that calling it warns.

    Raises
    ------
    AttributeError
        If `name` is not one of the deprecated re-exports.
    """
    if name not in _NEW_LOCATIONS:
        raise AttributeError(
            f"module 'dipy.testing.decorators' has no attribute {name!r}"
        )

    module = _NEW_LOCATIONS[name]
    resolved = deprecate_with_version(
        f"Using '{name}' from dipy.testing.decorators is deprecated. "
        f"Import it from {module} instead.",
        since="1.13.0",
        until="2.0.0",
    )(getattr(importlib.import_module(module), name))
    globals()[name] = resolved
    return resolved


SKIP_RE = re.compile(r"(\s*>>>.*?)(\s*)#\s*skip\s+if\s+(.*)$")


def doctest_skip_parser(func):
    """Decorator replaces custom skip test markup in doctests.

    Say a function has a docstring::

        >>> something # skip if not HAVE_AMODULE
        >>> something + else
        >>> something # skip if HAVE_BMODULE

    This decorator will evaluate the expression after ``skip if``.  If this
    evaluates to True, then the comment is replaced by ``# doctest: +SKIP``.
    If False, then the comment is just removed. The expression is evaluated in
    the ``globals`` scope of `func`.

    For example, if the module global ``HAVE_AMODULE`` is False, and module
    global ``HAVE_BMODULE`` is False, the returned function will have
    docstring::
        >>> something # doctest: +SKIP
        >>> something + else
        >>> something

    """
    lines = func.__doc__.split("\n")
    new_lines = []
    for line in lines:
        match = SKIP_RE.match(line)
        if match is None:
            new_lines.append(line)
            continue
        code, space, expr = match.groups()
        if eval(expr, func.__globals__):
            code = code + space + "# doctest: +SKIP"
        new_lines.append(code)
    func.__doc__ = "\n".join(new_lines)
    return func


###
# In some cases (e.g., on Travis), we want to use a virtual frame-buffer for
# testing. The following decorator runs the tests under xvfb (mediated by
# xvfbwrapper) conditioned on an environment variable (that we set in
# .travis.yml for these cases):
use_xvfb = os.environ.get("TEST_WITH_XVFB", False)
is_windows = platform.system().lower() == "windows"
is_macOS = platform.system().lower() == "darwin"
is_linux = platform.system().lower() == "linux"


def xvfb_it(my_test):
    """Run a test with xvfbwrapper."""
    # When we use verbose testing we want the name:
    fname = my_test.__name__

    def test_with_xvfb(*args, **kwargs):
        if use_xvfb:
            from xvfbwrapper import Xvfb

            display = Xvfb(width=1920, height=1080)
            display.start()
        my_test(*args, **kwargs)
        if use_xvfb:
            display.stop()

    # Plant it back in and return the new function:
    test_with_xvfb.__name__ = fname
    return test_with_xvfb if not is_windows else my_test


def set_random_number_generator(seed_v=1234):  # pep3102: ignore
    """Decorator to use a fixed value for the random generator seed.

    This will make the tests that use random functions reproducible.

    """

    def _set_random_number_generator(func):
        @wraps(func)
        def _set_random_number_generator_wrapper(*args, **kwargs):
            kwargs["rng"] = np.random.default_rng(seed_v)
            return func(*args, **kwargs)

        sig = inspect.signature(func)
        _set_random_number_generator_wrapper.__signature__ = sig.replace(
            parameters=[p for name, p in sig.parameters.items() if name != "rng"]
        )
        return _set_random_number_generator_wrapper

    return _set_random_number_generator
