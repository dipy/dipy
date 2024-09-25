"""
Decorators for dipy tests
"""

from functools import wraps
import inspect
from inspect import Parameter, signature
import os
import platform
import re
import warnings

import numpy as np
from packaging import version

import dipy

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


def set_random_number_generator(seed_v=1234):
    """Decorator to use a fixed value for the random generator seed.

    This will make the tests that use random functions reproducible.

    """

    def _set_random_number_generator(func):
        def _set_random_number_generator_wrapper(pytestconfig, *args, **kwargs):
            rng = np.random.default_rng(seed_v)
            kwargs["rng"] = rng
            signature = inspect.signature(func)
            if pytestconfig and "pytestconfig" in signature.parameters.keys():
                output = func(pytestconfig, *args, **kwargs)
            else:
                output = func(*args, **kwargs)
            return output

        return _set_random_number_generator_wrapper

    return _set_random_number_generator


def warning_for_keywords(from_version="1.10.0", until_version="2.0.0"):
    """
    Decorator to warn about keyword arguments passed as positional arguments
    and handle version-based deprecation of functions.

    Parameters
    ----------
    from_version : str, optional
        The version from which the warning should start.
    until_version : str, optional
        The version until which the warning is applicable.

    Returns
    -------
    function
        The wrapped function that will issue a warning based
        on the version.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_version = dipy.__version__

            parsed_version = version.parse(current_version)
            current_version = parsed_version.base_version

            def convert_positional_to_keyword(func, args, kwargs):
                """
                Converts excess positional arguments to keyword arguments.

                Parameters
                ----------
                func : function
                    The original function to be called.
                args : tuple
                    The positional arguments passed to the function.
                kwargs : dict
                    The keyword arguments passed to the function.

                Returns
                -------
                result
                    The result of the function call with corrected arguments.
                """
                sig = signature(func)
                params = sig.parameters
                max_positional_args = sum(
                    1
                    for param in params.values()
                    if param.kind
                    in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
                )

                if len(args) > max_positional_args:
                    # Split positional arguments into valid positional and
                    # those needing conversion
                    positional_args = list(args[:max_positional_args])
                    corrected_kwargs = dict(kwargs)
                    positionally_passed_kwonly_args = []
                    for param, arg in zip(
                        list(params.values())[max_positional_args:],
                        args[max_positional_args:],
                    ):
                        corrected_kwargs[param.name] = arg
                        positionally_passed_kwonly_args.append(param.name)

                    if positionally_passed_kwonly_args and version.parse(
                        from_version
                    ) <= version.parse(current_version) <= version.parse(until_version):
                        warnings.warn(
                            f"Pass {positionally_passed_kwonly_args} as keyword args. "
                            f"From version {until_version} passing these as positional "
                            f"arguments will result in an error. ",
                            UserWarning,
                            stacklevel=3,
                        )

                    return func(*positional_args, **corrected_kwargs)

                return func(*args, **kwargs)

            # Check if the current version is within the warning range
            if (
                version.parse(from_version)
                <= version.parse(current_version)
                <= version.parse(until_version)
            ):
                # Convert positional to keyword arguments and issue a warning
                return convert_positional_to_keyword(func, args, kwargs)

            # If the version is greater than the until_version,
            # pass the arguments as they are
            elif version.parse(current_version) > version.parse(until_version):
                return func(*args, **kwargs)

            # Convert positional to keyword arguments if
            # current version is less than from_version without warning
            elif version.parse(current_version) < version.parse(from_version):
                return convert_positional_to_keyword(func, args, kwargs)

            # Default case: call the function with the original arguments
            return func(*args, **kwargs)

        return wrapper

    return decorator
