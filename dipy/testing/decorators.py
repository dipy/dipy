# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Decorators for dipy tests
"""

import re
import os


SKIP_RE = re.compile("(\s*>>>.*?)(\s*)#\s*skip\s+if\s+(.*)$")


def doctest_skip_parser(func):
    """Decorator replaces custom skip test markup in doctests.

    Say a function has a docstring::

        >>> something # skip if not HAVE_AMODULE
        >>> something + else
        >>> something # skip if HAVE_BMODULE

    This decorator will evaluate the expresssion after ``skip if``.  If this
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


def xvfb_it(my_test):
    """Run a test with xvfbwrapper."""
    # When we use verbose testing we want the name:
    fname = my_test.__name__

    def test_with_xvfb():
        if use_xvfb:
            from xvfbwrapper import Xvfb
            display = Xvfb(width=1920, height=1080)
            display.start()
        my_test()
        if use_xvfb:
            display.stop()
    # Plant it back in and return the new function:
    test_with_xvfb.__name__ = fname
    return test_with_xvfb
