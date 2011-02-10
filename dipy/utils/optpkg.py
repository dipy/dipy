""" Routines to support optional packages """

try:
    import nose
except ImportError:
    have_nose = False
else:
    have_nose = True

from .tripwire import TripWire, is_tripwire

def optional_package(name, trip_msg=None):
    """ Return package-like thing and module setup for package `name`

    Parameters
    ----------
    name : str
        package name
    trip_msg : None or str
        message to give when someone tries to use the return package, but we
        could not import it, and have returned a TripWire object instead.
        Default message if None.

    Returns
    -------
    pkg_like : module or ``TripWire`` instance
        If we can import the package, return it.  Otherwise return an object
        raising an error when accessed
    have_pkg : bool
        True if import for package was succesful, false otherwise
    module_setup : function
        callable usually set as ``setup_module`` in calling namespace, to allow
        skipping tests.

    Example
    -------
    Typical use would be something like this at the top of a module using an
    optional package:

    >>> from dipy.utils.optpkg import optional_package
    >>> pkg, have_pkg, setup_module = optional_package('not_a_package')

    Of course in this case the package doesn't exist, and so, in the module:

    >>> have_pkg
    False

    and

    >>> pkg.some_function()
    Traceback (most recent call last):
        ...
    TripWireError: We need package not_a_package for these functions, but ``import not_a_package`` raised an ImportError
    """
    try:
        pkg = __import__(name)
    except ImportError:
        pass
    else: # import worked
        return pkg, True, lambda : None
    if trip_msg is None:
        trip_msg = ('We need package %s for these functions, but '
                    '``import %s`` raised an ImportError'
                    % (name, name))
    pkg = TripWire(trip_msg)
    def setup_module():
        if have_nose:
            raise nose.plugins.skip.SkipTest('No %s for these tests'
                                             % name)
    return pkg, False, setup_module

