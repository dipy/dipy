"""Utilities for testing."""
from functools import partial
import operator
from os.path import dirname, abspath, join as pjoin
import warnings

from numpy.testing import assert_array_equal
import numpy.testing as npt


# set path to example data
IO_DATA_PATH = abspath(pjoin(dirname(__file__),
                             '..', 'io', 'tests', 'data'))


def assert_operator(value1, value2, msg="", op=operator.eq):
    """Check Boolean statement."""
    try:
        if op == operator.is_:
            value1 = bool(value1)
        assert op(value1, value2)
    except AssertionError:
        raise AssertionError(msg.format(str(value2), str(value1)))


assert_greater_equal = partial(assert_operator, op=operator.ge,
                               msg="{0} >= {1}")
assert_greater = partial(assert_operator, op=operator.gt,
                         msg="{0} > {1}")
assert_less_equal = partial(assert_operator, op=operator.le,
                            msg="{0} =< {1}")
assert_less = partial(assert_operator, op=operator.lt,
                      msg="{0} < {1}")
assert_true = partial(assert_operator, value2=True, op=operator.is_,
                      msg="False is not true")
assert_false = partial(assert_operator, value2=False, op=operator.is_,
                       msg="True is not false")
assert_not_equal = partial(assert_operator, op=operator.ne)


def assert_arrays_equal(arrays1, arrays2):
    for arr1, arr2 in zip(arrays1, arrays2):
        assert_array_equal(arr1, arr2)


class clear_and_catch_warnings(warnings.catch_warnings):
    """Context manager that resets warning registry for catching warnings.

    Warnings can be slippery, because, whenever a warning is triggered, Python
    adds a ``__warningregistry__`` member to the *calling* module.  This makes
    it impossible to retrigger the warning in this module, whatever you put in
    the warnings filters.  This context manager accepts a sequence of `modules`
    as a keyword argument to its constructor and:
    * stores and removes any ``__warningregistry__`` entries in given `modules`
      on entry;
    * resets ``__warningregistry__`` to its previous state on exit.
    This makes it possible to trigger any warning afresh inside the context
    manager without disturbing the state of warnings outside.
    For compatibility with Python 3.0, please consider all arguments to be
    keyword-only.

    Parameters
    ----------
    record : bool, optional
        Specifies whether warnings should be captured by a custom
        implementation of ``warnings.showwarning()`` and be appended to a list
        returned by the context manager. Otherwise None is returned by the
        context manager. The objects appended to the list are arguments whose
        attributes mirror the arguments to ``showwarning()``.
        NOTE: nibabel difference from numpy: default is True
    modules : sequence, optional
        Sequence of modules for which to reset warnings registry on entry and
        restore on exit

    Examples
    --------
    >>> import warnings
    >>> import numpy as np
    >>> with clear_and_catch_warnings(modules=[np.core.fromnumeric]):
    ...     warnings.simplefilter('always')
    ...     # do something that raises a warning in np.core.fromnumeric

    Notes
    -----
    this class is copied (with minor modifications) from Nibabel.
    https://github.com/nipy/nibabel. See COPYING file distributed along with
    the Nibabel package for the copyright and license terms.

    """

    class_modules = ()

    def __init__(self, record=True, modules=()):
        self.modules = set(modules).union(self.class_modules)
        self._warnreg_copies = {}
        super(clear_and_catch_warnings, self).__init__(record=record)

    def __enter__(self):
        for mod in self.modules:
            if hasattr(mod, '__warningregistry__'):
                mod_reg = mod.__warningregistry__
                self._warnreg_copies[mod] = mod_reg.copy()
                mod_reg.clear()
        return super(clear_and_catch_warnings, self).__enter__()

    def __exit__(self, *exc_info):
        super(clear_and_catch_warnings, self).__exit__(*exc_info)
        for mod in self.modules:
            if hasattr(mod, '__warningregistry__'):
                mod.__warningregistry__.clear()
            if mod in self._warnreg_copies:
                mod.__warningregistry__.update(self._warnreg_copies[mod])


def check_for_warnings(warn_printed, w_msg):
    selected_w = [w for w in warn_printed if issubclass(w.category,
                                                        UserWarning)]
    assert len(selected_w) >= 1
    msg = [str(m.message) for m in selected_w]
    npt.assert_equal(w_msg in msg, True)
