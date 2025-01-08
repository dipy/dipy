"""Utility functions for checking different stuffs."""

import operator

from packaging import version


def check_version(package_name, ver, *, operator=operator.ge):
    """Check if the package is installed and the version satisfies the operator.

    Parameters
    ----------
    package_name : str
        The name of the package to check.
    ver : str
        The version to check against.
    operator : callable, optional
        The operator to use for the comparison.

    Returns
    -------
    bool
        True if the package is installed and the version satisfies the operator.
    """
    try:
        pkg = __import__(package_name)
    except ImportError:
        return False
    if hasattr(pkg, "__version__"):
        return operator(version.parse(pkg.__version__), version.parse(ver))
    return False


def check_min_version(package_name, min_version, *, strict=False):
    """Check if the package is installed and the version is at least min_version.

    Parameters
    ----------
    package_name : str
        The name of the package to check.
    min_version : str
        The minimum version required.
    strict : bool, optional
        If True, the version must be strictly greater than min_version.

    Returns
    -------
    bool
        True if the package is installed and the version is at least min_version.
    """
    op = operator.gt if strict else operator.ge
    return check_version(package_name, min_version, operator=op)


def check_max_version(package_name, max_version, *, strict=False):
    """Check if the package is installed and the version is at most max_version.

    Parameters
    ----------
    package_name : str
        The name of the package to check.
    max_version : str
        The maximum version required.
    strict : bool, optional
        If True, the version must be strictly less than max_version.

    Returns
    -------
    bool
        True if the package is installed and the version is at most max_version.
    """
    op = operator.lt if strict else operator.le
    return check_version(package_name, max_version, operator=op)
