"""
Decorators for dipy tests
"""

from functools import cache, wraps
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


# Applied as @set_random_number_generator(42), so seed_v stays positional.
def set_random_number_generator(seed_v=1234):  # pep3102: ignore
    """Decorator to use a fixed value for the random generator seed.

    This will make the tests that use random functions reproducible.

    """

    def _set_random_number_generator(func):
        @wraps(func)
        def _set_random_number_generator_wrapper(*args, **kwargs):
            kwargs["rng"] = np.random.default_rng(seed_v)
            return func(*args, **kwargs)

        # `pytest` builds a test's fixture request from the signature it can
        # see, so the wrapper must advertise the parameters the test owns
        # (`tmp_path`, `pytestconfig`, ...) while hiding `rng`, which this
        # decorator supplies. Leaving `rng` visible makes `pytest` look for a
        # fixture of that name and fail at setup with `fixture 'rng' not
        # found`, unless the test happens to give `rng` a default value.
        sig = inspect.signature(func)
        _set_random_number_generator_wrapper.__signature__ = sig.replace(
            parameters=[p for name, p in sig.parameters.items() if name != "rng"]
        )
        return _set_random_number_generator_wrapper

    return _set_random_number_generator


@cache
def _parse_base_version(raw_version):
    """Parse ``raw_version``, dropping any pre/post/dev suffix.

    Memoized: :func:`warning_for_keywords` reads ``dipy.__version__`` on every
    decorated call, and ``packaging`` parsing is comparatively expensive.

    Parameters
    ----------
    raw_version : str
        Version string, e.g. ``"1.13.0.dev0"``.

    Returns
    -------
    packaging.version.Version
        The parsed base version, e.g. ``1.13.0``.
    """
    return version.parse(version.parse(raw_version).base_version)


def warning_for_keywords(*args, from_version="1.10.0", until_version="2.0.0"):
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
    if args:
        # Both versions used to be positional-or-keyword.
        if len(args) > 2:
            raise TypeError(
                "warning_for_keywords() takes at most 2 positional arguments "
                f"but {len(args)} were given"
            )
        warnings.warn(
            "Pass ['from_version', 'until_version'] as keyword args. From version "
            "2.0.0 passing these as positional arguments will result in an error. ",
            UserWarning,
            stacklevel=2,
        )
        from_version = args[0]
        if len(args) > 1:
            until_version = args[1]

    parsed_from = version.parse(from_version)
    parsed_until = version.parse(until_version)

    def decorator(func):
        # Resolved once, at decoration time: the signature of ``func`` cannot
        # change, so there is no reason to re-inspect it on every call.
        sig = signature(func)
        params = tuple(sig.parameters.values())
        max_positional_args = sum(
            1
            for param in params
            if param.kind
            in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        )
        extra_params = params[max_positional_args:]

        def convert_positional_to_keyword(args, kwargs, *, warn):
            """
            Converts excess positional arguments to keyword arguments.

            Parameters
            ----------
            args : tuple
                The positional arguments passed to the function.
            kwargs : dict
                The keyword arguments passed to the function.
            warn : bool
                Whether to report the arguments that had to be converted.

            Returns
            -------
            result
                The result of the function call with corrected arguments.
            """
            if len(args) <= max_positional_args:
                return func(*args, **kwargs)

            # Split positional arguments into valid positional and
            # those needing conversion
            corrected_kwargs = dict(kwargs)
            positionally_passed_kwonly_args = []
            for param, arg in zip(extra_params, args[max_positional_args:]):
                corrected_kwargs[param.name] = arg
                positionally_passed_kwonly_args.append(param.name)

            if warn and positionally_passed_kwonly_args:
                warnings.warn(
                    f"Pass {positionally_passed_kwonly_args} as keyword args. "
                    f"From version {until_version} passing these as positional "
                    f"arguments will result in an error. ",
                    UserWarning,
                    stacklevel=3,
                )

            return func(*args[:max_positional_args], **corrected_kwargs)

        @wraps(func)
        def wrapper(*args, **kwargs):
            current_version = _parse_base_version(dipy.__version__)

            # Check if the current version is within the warning range
            if parsed_from <= current_version <= parsed_until:
                # Convert positional to keyword arguments and issue a warning
                return convert_positional_to_keyword(args, kwargs, warn=True)

            # If the version is greater than the until_version,
            # pass the arguments as they are
            if current_version > parsed_until:
                return func(*args, **kwargs)

            # Convert positional to keyword arguments if
            # current version is less than from_version without warning
            return convert_positional_to_keyword(args, kwargs, warn=False)

        # Explicitly preserve the original function's signature and wrapped attribute
        # This is crucial for compatibility with libraries like Keras 3.x that use
        # inspect.signature() and getfullargspec() for introspection
        wrapper.__wrapped__ = func
        wrapper.__signature__ = sig

        return wrapper

    return decorator
