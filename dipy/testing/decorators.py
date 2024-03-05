# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Decorators for dipy tests
"""

import re
import os
import platform
import inspect
import numpy as np

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
    lines = func.__doc__.split('\n')
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
use_xvfb = os.environ.get('TEST_WITH_XVFB', False)
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


def set_random_number_generator(seed_v=1234):
    """Decorator to use a fixed value for the random generator seed.

    This will make the tests that use random functions reproducible.

    """
    def _set_random_number_generator(func):
        def _set_random_number_generator_wrapper(pytestconfig, *args, **kwargs):
            rng = np.random.default_rng(seed_v)
            kwargs['rng'] = rng
            signature = inspect.signature(func)
            if pytestconfig and 'pytestconfig' in signature.parameters.keys():
                output = func(pytestconfig, *args, **kwargs)
            else:
                output = func(*args, **kwargs)
            return output
        return _set_random_number_generator_wrapper
    return _set_random_number_generator
