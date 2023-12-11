""" Routines to support optional packages """

import importlib
from packaging.version import Version

try:
    import pytest
except ImportError:
    have_pytest = False
else:
    have_pytest = True

from dipy.utils.tripwire import TripWire


def optional_package(name, *, trip_msg=None, min_version=None):
    """ Return package-like thing and module setup for package `name`

    Parameters
    ----------
    name : str
        package name
    trip_msg : None or str
        message to give when someone tries to use the return package, but we
        could not import it, and have returned a TripWire object instead.
        Default message if None.
    min_version : None or str
        If not None, require that the imported package be at least this
        version.  If the package has no ``__version__`` attribute, or if the
        version is not parseable, raise an error.

    Returns
    -------
    pkg_like : module or ``TripWire`` instance
        If we can import the package, return it.  Otherwise return an object
        raising an error when accessed
    have_pkg : bool
        True if import for package was successful, false otherwise
    module_setup : function
        callable usually set as ``setup_module`` in calling namespace, to allow
        skipping tests.

    Examples
    --------
    Typical use would be something like this at the top of a module using an
    optional package:

    >>> from dipy.utils.optpkg import optional_package
    >>> pkg, have_pkg, setup_module = optional_package('not_a_package')

    Of course in this case the package doesn't exist, and so, in the module:

    >>> have_pkg
    False

    and

    >>> pkg.some_function() #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TripWireError: We need package not_a_package for these functions, but
    ``import not_a_package`` raised an ImportError

    If the module does exist - we get the module

    >>> pkg, _, _ = optional_package('os')
    >>> hasattr(pkg, 'path')
    True

    Or a submodule if that's what we asked for

    >>> subpkg, _, _ = optional_package('os.path')
    >>> hasattr(subpkg, 'dirname')
    True
    """
    try:
        pkg = importlib.import_module(name)
    except ImportError:
        pass
    else:  # import worked
        # top level module
        if not min_version:
            return pkg, True, lambda: None

        current_version = getattr(pkg, '__version__', '0.0.0')
        if Version(current_version) >= Version(min_version):
            return pkg, True, lambda: None

        if trip_msg is None:
            trip_msg = (f'We need at least version {min_version} of '
                        f'package {name}, but ``import {name}`` '
                        f'found version {pkg.__version__}')

    if trip_msg is None:
        trip_msg = ('We need package %s for these functions, but '
                    '``import %s`` raised an ImportError'
                    % (name, name))
    pkg = TripWire(trip_msg)

    def setup_module():
        if have_pytest:
            pytest.mark.skip('No {0} for these tests'.format(name))

    return pkg, False, setup_module
