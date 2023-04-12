"""Function for recording and reporting deprecations.

Note
-----
this file is copied (with minor modifications) from the Nibabel.
https://github.com/nipy/nibabel. See COPYING file distributed along with
the Nibabel package for the copyright and license terms.

"""

import functools
import warnings
import re
from inspect import signature
from dipy import __version__
from packaging.version import parse as version_cmp

_LEADING_WHITE = re.compile(r'^(\s*)')


class ExpiredDeprecationError(RuntimeError):
    """Error for expired deprecation.

    Error raised when a called function or method has passed out of its
    deprecation period.

    """

    pass


class ArgsDeprecationWarning(DeprecationWarning):
    """Warning for args deprecation.

    Warning raised when a function or method argument has changed or removed.

    """

    pass


def _ensure_cr(text):
    """Remove trailing whitespace and add carriage return.

    Ensures that `text` always ends with a carriage return
    """
    return text.rstrip() + '\n'


def _add_dep_doc(old_doc, dep_doc):
    """Add deprecation message `dep_doc` to docstring in `old_doc`.

    Parameters
    ----------
    old_doc : str
        Docstring from some object.
    dep_doc : str
        Deprecation warning to add to top of docstring, after initial line.

    Returns
    -------
    new_doc : str
        `old_doc` with `dep_doc` inserted after any first lines of docstring.

    """
    dep_doc = _ensure_cr(dep_doc)
    if not old_doc:
        return dep_doc
    old_doc = _ensure_cr(old_doc)
    old_lines = old_doc.splitlines()
    new_lines = []
    for line_no, line in enumerate(old_lines):
        if line.strip():
            new_lines.append(line)
        else:
            break
    next_line = line_no + 1
    if next_line >= len(old_lines):
        # nothing following first paragraph, just append message
        return old_doc + '\n' + dep_doc
    indent = _LEADING_WHITE.match(old_lines[next_line]).group()
    dep_lines = [indent + L for L in [''] + dep_doc.splitlines() + ['']]
    return '\n'.join(new_lines + dep_lines + old_lines[next_line:]) + '\n'


def cmp_pkg_version(version_str, pkg_version_str=__version__):
    """Compare `version_str` to current package version.

    Parameters
    ----------
    version_str : str
        Version string to compare to current package version
    pkg_version_str : str, optional
        Version of our package.  Optional, set from ``__version__`` by default.

    Returns
    -------
    version_cmp : int
        1 if `version_str` is a later version than `pkg_version_str`, 0 if
        same, -1 if earlier.

    Examples
    --------
    >>> cmp_pkg_version('1.2.1', '1.2.0')
    1
    >>> cmp_pkg_version('1.2.0dev', '1.2.0')
    -1

    """
    if any([re.match(r'^[a-z, A-Z]', v)for v in [version_str,
                                                 pkg_version_str]]):
        msg = 'Invalid version {0} or {1}'.format(version_str, pkg_version_str)
        raise ValueError(msg)
    elif version_cmp(version_str) > version_cmp(pkg_version_str):
        return 1
    elif version_cmp(version_str) == version_cmp(pkg_version_str):
        return 0
    else:
        return -1


def is_bad_version(version_str, version_comparator=cmp_pkg_version):
    """Return True if `version_str` is too high."""
    return version_comparator(version_str) == -1


def deprecate_with_version(message, since='', until='',
                           version_comparator=cmp_pkg_version,
                           warn_class=DeprecationWarning,
                           error_class=ExpiredDeprecationError):
    """Return decorator function function for deprecation warning / error.

    The decorated function / method will:

    * Raise the given `warning_class` warning when the function / method gets
      called, up to (and including) version `until` (if specified);
    * Raise the given `error_class` error when the function / method gets
      called, when the package version is greater than version `until` (if
      specified).

    Parameters
    ----------
    message : str
        Message explaining deprecation, giving possible alternatives.
    since : str, optional
        Released version at which object was first deprecated.
    until : str, optional
        Last released version at which this function will still raise a
        deprecation warning.  Versions higher than this will raise an
        error.
    version_comparator : callable
        Callable accepting string as argument, and return 1 if string
        represents a higher version than encoded in the `version_comparator`, 0
        if the version is equal, and -1 if the version is lower.  For example,
        the `version_comparator` may compare the input version string to the
        current package version string.
    warn_class : class, optional
        Class of warning to generate for deprecation.
    error_class : class, optional
        Class of error to generate when `version_comparator` returns 1 for a
        given argument of ``until``.

    Returns
    -------
    deprecator : func
        Function returning a decorator.

    """
    messages = [message]
    if (since, until) != ('', ''):
        messages.append('')
    if since:
        messages.append('* deprecated from version: ' + since)
    if until:
        messages.append('* {0} {1} as of version: {2}'.format(
            "Raises" if is_bad_version(until) else "Will raise",
            error_class,
            until))
    message = '\n'.join(messages)

    def deprecator(func):

        @functools.wraps(func)
        def deprecated_func(*args, **kwargs):
            if until and is_bad_version(until, version_comparator):
                raise error_class(message)
            warnings.warn(message, warn_class, stacklevel=2)
            return func(*args, **kwargs)

        deprecated_func.__doc__ = _add_dep_doc(deprecated_func.__doc__,
                                               message)
        return deprecated_func

    return deprecator


def deprecated_params(old_name, new_name=None, since='', until='',
                      version_comparator=cmp_pkg_version,
                      arg_in_kwargs=False,
                      warn_class=ArgsDeprecationWarning,
                      error_class=ExpiredDeprecationError,
                      alternative=''):
    """Deprecate a *renamed* or *removed* function argument.

    The decorator assumes that the argument with the ``old_name`` was removed
    from the function signature and the ``new_name`` replaced it at the
    **same position** in the signature.  If the ``old_name`` argument is
    given when calling the decorated function the decorator will catch it and
    issue a deprecation warning and pass it on as ``new_name`` argument.

    Parameters
    ----------
    old_name : str or list/tuple thereof
        The old name of the argument.
    new_name : str or list/tuple thereof or ``None``, optional
        The new name of the argument. Set this to `None` to remove the
        argument ``old_name`` instead of renaming it.
    since : str or number or list/tuple thereof, optional
        The release at which the old argument became deprecated.
    until : str or number or list/tuple thereof, optional
        Last released version at which this function will still raise a
        deprecation warning.  Versions higher than this will raise an
        error.
    version_comparator : callable
        Callable accepting string as argument, and return 1 if string
        represents a higher version than encoded in the ``version_comparator``,
        0 if the version is equal, and -1 if the version is lower. For example,
        the ``version_comparator`` may compare the input version string to the
        current package version string.
    arg_in_kwargs : bool or list/tuple thereof, optional
        If the argument is not a named argument (for example it
        was meant to be consumed by ``**kwargs``) set this to
        ``True``.  Otherwise the decorator will throw an Exception
        if the ``new_name`` cannot be found in the signature of
        the decorated function.
        Default is ``False``.
    warn_class : warning, optional
        Warning to be issued.
    error_class : Exception, optional
        Error to be issued
    alternative : str, optional
        An alternative function or class name that the user may use in
        place of the deprecated object if ``new_name`` is None. The deprecation
        warning will tell the user about this alternative if provided.

    Raises
    ------
    TypeError
        If the new argument name cannot be found in the function
        signature and arg_in_kwargs was False or if it is used to
        deprecate the name of the ``*args``-, ``**kwargs``-like arguments.
        At runtime such an Error is raised if both the new_name
        and old_name were specified when calling the function and
        "relax=False".

    Notes
    -----
    This function is based on the Astropy (major modification).
    https://github.com/astropy/astropy. See COPYING file distributed along with
    the astropy package for the copyright and license terms.

    Examples
    --------
    The deprecation warnings are not shown in the following examples.
    To deprecate a positional or keyword argument::
    >>> from dipy.utils.deprecator import deprecated_params
    >>> @deprecated_params('sig', 'sigma', '0.3')
    ... def test(sigma):
    ...     return sigma
    >>> test(2)
    2
    >>> test(sigma=2)
    2
    >>> test(sig=2)  # doctest: +SKIP
    2

    It is also possible to replace multiple arguments. The ``old_name``,
    ``new_name`` and ``since`` have to be `tuple` or `list` and contain the
    same number of entries::
    >>> @deprecated_params(['a', 'b'], ['alpha', 'beta'],
    ...                    ['0.2', 0.4])
    ... def test(alpha, beta):
    ...     return alpha, beta
    >>> test(a=2, b=3)  # doctest: +SKIP
    (2, 3)

    """
    if isinstance(old_name, (list, tuple)):
        # Normalize input parameters
        if not isinstance(arg_in_kwargs, (list, tuple)):
            arg_in_kwargs = [arg_in_kwargs] * len(old_name)
        if not isinstance(since, (list, tuple)):
            since = [since] * len(old_name)
        if not isinstance(until, (list, tuple)):
            until = [until] * len(old_name)
        if not isinstance(new_name, (list, tuple)):
            new_name = [new_name] * len(old_name)

        if len({len(old_name), len(new_name), len(since), len(until),
                len(arg_in_kwargs)}) != 1:
            raise ValueError("All parameters should have the same length")
    else:
        # To allow a uniform approach later on, wrap all arguments in lists.
        old_name = [old_name]
        new_name = [new_name]
        since = [since]
        until = [until]
        arg_in_kwargs = [arg_in_kwargs]

    def deprecator(function):
        # The named arguments of the function.
        arguments = signature(function).parameters
        positions = [None] * len(old_name)

        for i, (o_name, n_name, in_keywords) in enumerate(zip(old_name,
                                                              new_name,
                                                              arg_in_kwargs)):
            # Determine the position of the argument.
            if in_keywords:
                continue

            if n_name is not None and n_name not in arguments:
                # In case the argument is not found in the list of arguments
                # the only remaining possibility is that it should be caught
                # by some kind of **kwargs argument.
                msg = '"{}" was not specified in the function '.format(n_name)
                msg += 'signature. If it was meant to be part of '
                msg += '"**kwargs" then set "arg_in_kwargs" to "True"'
                raise TypeError(msg)

            key = o_name if n_name is None else n_name
            param = arguments[key]

            if param.kind == param.POSITIONAL_OR_KEYWORD:
                key = o_name if n_name is None else n_name
                positions[i] = list(arguments.keys()).index(key)
            elif param.kind == param.KEYWORD_ONLY:
                # These cannot be specified by position.
                positions[i] = None
            else:
                # positional-only argument, varargs, varkwargs or some
                # unknown type:
                msg = 'cannot replace argument "{}" '.format(n_name)
                msg += 'of kind {}.'.format(repr(param.kind))
                raise TypeError(msg)

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            for i, (o_name, n_name) in enumerate(zip(old_name, new_name)):
                messages = ['"{}" was deprecated'.format(o_name), ]
                if (since[i], until[i]) != ('', ''):
                    messages.append('')
                if since[i]:
                    messages.append('* deprecated from version: ' +
                                    str(since[i]))
                if until[i]:
                    messages.append('* {0} {1} as of version: {2}'.format(
                        "Raises" if is_bad_version(until[i]) else "Will raise",
                        error_class,
                        until[i]))
                messages.append('')
                message = '\n'.join(messages)

                # The only way to have oldkeyword inside the function is
                # that it is passed as kwarg because the oldkeyword
                # parameter was renamed to newkeyword.
                if o_name in kwargs:
                    value = kwargs.pop(o_name)
                    # Check if the newkeyword was given as well.
                    newarg_in_args = (positions[i] is not None and
                                      len(args) > positions[i])
                    newarg_in_kwargs = n_name in kwargs

                    if newarg_in_args or newarg_in_kwargs:
                        msg = 'cannot specify both "{}"'.format(o_name)
                        msg += ' (deprecated parameter) and '
                        msg += '"{}" (new parameter name).'.format(n_name)
                        raise TypeError(msg)

                    # Pass the value of the old argument with the
                    # name of the new argument to the function
                    key = n_name or o_name
                    kwargs[key] = value

                    if n_name is not None:
                        message += '* Use argument "{}" instead.' \
                            .format(n_name)
                    elif alternative:
                        message += '* Use {} instead.'.format(alternative)

                    if until[i] and is_bad_version(until[i],
                                                   version_comparator):
                        raise error_class(message)
                    warnings.warn(message, warn_class, stacklevel=2)

                # Deprecated keyword without replacement is given as
                # positional argument.
                elif (not n_name and positions[i] and
                      len(args) > positions[i]):
                    if alternative:
                        message += '* Use {} instead.'.format(alternative)
                    if until[i] and is_bad_version(until[i],
                                                   version_comparator):
                        raise error_class(message)

                    warnings.warn(message, warn_class, stacklevel=2)

            return function(*args, **kwargs)

        return wrapper
    return deprecator
