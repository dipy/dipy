"""Module to test ``dipy.utils.deprecator`` module.

Notes
-----
this file is copied (with minor modifications) from Nibabel.
https://github.com/nipy/nibabel. See COPYING file distributed along with
the Nibabel package for the copyright and license terms.

"""

import sys
import warnings
import numpy.testing as npt
import pytest
import dipy
from dipy.testing import clear_and_catch_warnings, assert_true
from dipy.utils.deprecator import (cmp_pkg_version, _add_dep_doc,
                                   _ensure_cr, deprecate_with_version,
                                   ExpiredDeprecationError)


def test_cmp_pkg_version():
    # Test version comparator
    npt.assert_equal(cmp_pkg_version(dipy.__version__), 0)
    npt.assert_equal(cmp_pkg_version('0.0'), -1)
    npt.assert_equal(cmp_pkg_version('1000.1000.1'), 1)
    npt.assert_equal(cmp_pkg_version(dipy.__version__, dipy.__version__), 0)
    for test_ver, pkg_ver, exp_out in (('1.0', '1.0', 0),
                                       ('1.0.0', '1.0', 0),
                                       ('1.0', '1.0.0', 0),
                                       ('1.1', '1.1', 0),
                                       ('1.2', '1.1', 1),
                                       ('1.1', '1.2', -1),
                                       ('1.1.1', '1.1.1', 0),
                                       ('1.1.2', '1.1.1', 1),
                                       ('1.1.1', '1.1.2', -1),
                                       ('1.1', '1.1dev', 1),
                                       ('1.1dev', '1.1', -1),
                                       ('1.2.1', '1.2.1rc1', 1),
                                       ('1.2.1rc1', '1.2.1', -1),
                                       ('1.2.1rc1', '1.2.1rc', 1),
                                       ('1.2.1rc', '1.2.1rc1', -1),
                                       ('1.2.1rc1', '1.2.1rc', 1),
                                       ('1.2.1rc', '1.2.1rc1', -1),
                                       ('1.2.1b', '1.2.1a', 1),
                                       ('1.2.1a', '1.2.1b', -1),
                                       ):
        npt.assert_equal(cmp_pkg_version(test_ver, pkg_ver), exp_out)

    npt.assert_raises(ValueError, cmp_pkg_version, 'foo.2')
    npt.assert_raises(ValueError, cmp_pkg_version, 'foo.2', '1.0')
    npt.assert_raises(ValueError, cmp_pkg_version, '1.0', 'foo.2')
    npt.assert_raises(ValueError, cmp_pkg_version, 'foo')


def test__ensure_cr():
    # Make sure text ends with carriage return
    npt.assert_equal(_ensure_cr('  foo'), '  foo\n')
    npt.assert_equal(_ensure_cr('  foo\n'), '  foo\n')
    npt.assert_equal(_ensure_cr('  foo  '), '  foo\n')
    npt.assert_equal(_ensure_cr('foo  '), 'foo\n')
    npt.assert_equal(_ensure_cr('foo  \n bar'), 'foo  \n bar\n')
    npt.assert_equal(_ensure_cr('foo  \n\n'), 'foo\n')


def test__add_dep_doc():
    # Test utility function to add deprecation message to docstring
    npt.assert_equal(_add_dep_doc('', 'foo'), 'foo\n')
    npt.assert_equal(_add_dep_doc('bar', 'foo'), 'bar\n\nfoo\n')
    npt.assert_equal(_add_dep_doc('   bar', 'foo'), '   bar\n\nfoo\n')
    npt.assert_equal(_add_dep_doc('   bar', 'foo\n'), '   bar\n\nfoo\n')
    npt.assert_equal(_add_dep_doc('bar\n\n', 'foo'), 'bar\n\nfoo\n')
    npt.assert_equal(_add_dep_doc('bar\n    \n', 'foo'), 'bar\n\nfoo\n')
    npt.assert_equal(_add_dep_doc(' bar\n\nSome explanation', 'foo\nbaz'),
                     ' bar\n\nfoo\nbaz\n\nSome explanation\n')
    npt.assert_equal(_add_dep_doc(' bar\n\n  Some explanation', 'foo\nbaz'),
                     ' bar\n  \n  foo\n  baz\n  \n  Some explanation\n')


def test_deprecate_with_version():

    def func_no_doc():
        pass

    def func_doc(i):
        """Fake docstring."""

    def func_doc_long(i, j):
        """Fake docstring.\n\n   Some text."""

    class CustomError(Exception):
        """Custom error class for testing expired deprecation errors."""

    my_mod = sys.modules[__name__]
    dec = deprecate_with_version

    func = dec('foo')(func_no_doc)
    with clear_and_catch_warnings(modules=[my_mod]) as w:
        warnings.simplefilter('always')
        npt.assert_equal(func(), None)
        npt.assert_equal(len(w), 1)
        assert_true(w[0].category is DeprecationWarning)
    npt.assert_equal(func.__doc__, 'foo\n')
    func = dec('foo')(func_doc)
    with clear_and_catch_warnings(modules=[my_mod]) as w:
        warnings.simplefilter('always')
        npt.assert_equal(func(1), None)
        npt.assert_equal(len(w), 1)
    npt.assert_equal(func.__doc__, 'Fake docstring.\n\nfoo\n')
    func = dec('foo')(func_doc_long)
    with clear_and_catch_warnings(modules=[my_mod]) as w:
        warnings.simplefilter('always')
        npt.assert_equal(func(1, 2), None)
        npt.assert_equal(len(w), 1)
    npt.assert_equal(func.__doc__,
                     'Fake docstring.\n   \n   foo\n   \n   Some text.\n')

    # Try some since and until versions
    func = dec('foo', '0.2')(func_no_doc)
    npt.assert_equal(func.__doc__, 'foo\n\n* deprecated from version: 0.2\n')
    with clear_and_catch_warnings(modules=[my_mod]) as w:
        warnings.simplefilter('always')
        npt.assert_equal(func(), None)
        npt.assert_equal(len(w), 1)
    func = dec('foo', until='100.6')(func_no_doc)
    with clear_and_catch_warnings(modules=[my_mod]) as w:
        warnings.simplefilter('always')
        npt.assert_equal(func(), None)
        npt.assert_equal(len(w), 1)
    npt.assert_equal(func.__doc__,
                     'foo\n\n* Will raise {} as of version: 100.6\n'
                     .format(ExpiredDeprecationError))
    func = dec('foo', until='0.3')(func_no_doc)
    npt.assert_raises(ExpiredDeprecationError, func)
    npt.assert_equal(func.__doc__,
                     'foo\n\n* Raises {} as of version: 0.3\n'
                     .format(ExpiredDeprecationError))
    func = dec('foo', '0.2', '0.3')(func_no_doc)
    npt.assert_raises(ExpiredDeprecationError, func)
    npt.assert_equal(func.__doc__,
                     'foo\n\n* deprecated from version: 0.2\n'
                     '* Raises {} as of version: 0.3\n'
                     .format(ExpiredDeprecationError))
    func = dec('foo', '0.2', '0.3')(func_doc_long)
    npt.assert_equal(func.__doc__,
                     'Fake docstring.\n   \n   foo\n   \n'
                     '   * deprecated from version: 0.2\n'
                     '   * Raises {} as of version: 0.3\n   \n'
                     '   Some text.\n'
                     .format(ExpiredDeprecationError))
    npt.assert_raises(ExpiredDeprecationError, func)

    # Check different warnings and errors
    func = dec('foo', warn_class=UserWarning)(func_no_doc)
    with clear_and_catch_warnings(modules=[my_mod]) as w:
        warnings.simplefilter('always')
        npt.assert_equal(func(), None)
        npt.assert_equal(len(w), 1)
        assert_true(w[0].category is UserWarning)

    func = dec('foo', error_class=CustomError)(func_no_doc)
    with clear_and_catch_warnings(modules=[my_mod]) as w:
        warnings.simplefilter('always')
        npt.assert_equal(func(), None)
        npt.assert_equal(len(w), 1)
        assert_true(w[0].category is DeprecationWarning)

    func = dec('foo', until='0.3', error_class=CustomError)(func_no_doc)
    npt.assert_raises(CustomError, func)
