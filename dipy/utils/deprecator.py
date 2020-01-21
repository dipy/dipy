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
from dipy import __version__
from packaging.version import parse as version_cmp

_LEADING_WHITE = re.compile(r'^(\s*)')


class ExpiredDeprecationError(RuntimeError):
    """Error for expired deprecation.

    Error raised when a called function or method has passed out of its
    deprecation period.

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
        Version of our package.  Optional, set fom ``__version__`` by default.
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
    def is_bad_version(version_str):
        """Return True if `version_str` is too high."""
        return version_comparator(version_str) == -1

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
            if until and is_bad_version(until):
                raise error_class(message)
            warnings.warn(message, warn_class, stacklevel=2)
            return func(*args, **kwargs)

        deprecated_func.__doc__ = _add_dep_doc(deprecated_func.__doc__,
                                               message)
        return deprecated_func

    return deprecator
