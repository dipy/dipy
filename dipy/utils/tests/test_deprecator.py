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
                                   deprecated_params, ArgsDeprecationWarning,
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


def test_deprecated_argument():
    # Tests the decorator with function, method, staticmethod and classmethod.
    class CustomActor:

        @classmethod
        @deprecated_params('height', 'scale', '0.3')
        def test1(cls, scale):
            return scale

        @staticmethod
        @deprecated_params('height', 'scale', '0.3')
        def test2(scale):
            return scale

        @deprecated_params('height', 'scale', '0.3')
        def test3(self, scale):
            return scale

        @deprecated_params('height', 'scale', '0.3', '0.5')
        def test4(self, scale):
            return scale

        @deprecated_params('height', 'scale', '0.3', '10.0.0')
        def test5(self, scale):
            return scale

    @deprecated_params('height', 'scale', '0.3')
    def custom_actor(scale):
        return scale

    @deprecated_params('height', 'scale', '0.3', '0.5')
    def custom_actor_2(scale):
        return scale

    for method in [CustomActor().test1, CustomActor().test2,
                   CustomActor().test3, CustomActor().test4, custom_actor,
                   custom_actor_2]:
        # As positional argument only
        npt.assert_equal(method(1), 1)
        # As new keyword argument
        npt.assert_equal(method(scale=1), 1)
        # As old keyword argument
        if method.__name__ not in ['test4', 'custom_actor_2']:
            res = npt.assert_warns(ArgsDeprecationWarning, method, height=1)
            npt.assert_equal(res, 1)
        else:
            npt.assert_raises(ExpiredDeprecationError, method, height=1)

        # Using both. Both keyword
        npt.assert_raises(TypeError, method, height=2, scale=1)
        # One positional, one keyword
        npt.assert_raises(TypeError, method, 1, scale=2)

    with warnings.catch_warnings(record=True) as w_record:
        res = CustomActor().test5(4)

        # Select only UserWarning
        selected_w = [w for w in w_record
                      if issubclass(w.category, UserWarning)]
        npt.assert_equal(len(selected_w), 0)
        npt.assert_equal(res, 4)


def test_deprecated_argument_in_kwargs():
    # To rename an argument that is consumed by "kwargs" the "arg_in_kwargs"
    # parameter is used.
    @deprecated_params('height', 'scale', '0.3', arg_in_kwargs=True)
    def test(**kwargs):
        return kwargs['scale']

    @deprecated_params('height', 'scale', '0.3', '0.5', arg_in_kwargs=True)
    def test2(**kwargs):
        return kwargs['scale']

    # As positional argument only
    npt.assert_raises(TypeError, test, 1)

    # As new keyword argument
    npt.assert_equal(test(scale=1), 1)

    # Using the deprecated name
    res = npt.assert_warns(ArgsDeprecationWarning, test, height=1)
    npt.assert_equal(res, 1)

    npt.assert_raises(ExpiredDeprecationError, test2, height=1)

    # Using both. Both keyword
    npt.assert_raises(TypeError, test, height=2, scale=1)
    # One positional, one keyword
    npt.assert_raises(TypeError, test, 1, scale=2)


def test_deprecated_argument_multi_deprecation():

    @deprecated_params(['x', 'y', 'z'], ['a', 'b', 'c'],
                       [0.3, 0.2, 0.4])
    def test(a, b, c):
        return a, b, c

    @deprecated_params(['x', 'y', 'z'], ['a', 'b', 'c'],
                       '0.3')
    def test2(a, b, c):
        return a, b, c

    with pytest.warns(ArgsDeprecationWarning) as w:
        npt.assert_equal(test(x=1, y=2, z=3), (1, 2, 3))
        npt.assert_equal(test2(x=1, y=2, z=3), (1, 2, 3))
        npt.assert_equal(len(w), 6)

        npt.assert_raises(TypeError, test, x=1, y=2, z=3, b=3)
        npt.assert_raises(TypeError, test, x=1, y=2, z=3, a=3)


def test_deprecated_argument_not_allowed_use():
    # If the argument is supposed to be inside the kwargs one needs to set the
    # arg_in_kwargs parameter. Without it it raises a TypeError.
    with pytest.raises(TypeError):
        @deprecated_params('height', 'scale', '0.3')
        def test1(**kwargs):
            return kwargs['scale']

    # Cannot replace "*args".
    with pytest.raises(TypeError):
        @deprecated_params('scale', 'args', '0.3')
        def test2(*args):
            return args

    # Cannot replace "**kwargs".
    with pytest.raises(TypeError):
        @deprecated_params('scale', 'kwargs', '0.3')
        def test3(**kwargs):
            return kwargs

    # wrong number of arguments
    with pytest.raises(ValueError):
        @deprecated_params(['a', 'b', 'c'], ['x', 'y'], '0.3')
        def test4(**kwargs):
            return kwargs


def test_deprecated_argument_remove():
    @deprecated_params('x', None, '0.3', alternative='test2.y')
    def test(dummy=11, x=3):
        return dummy, x

    @deprecated_params('x', None, '0.3', '0.5', alternative='test2.y')
    def test2(dummy=11, x=3):
        return dummy, x

    @deprecated_params(['dummy', 'x'], None, '0.3', alternative='test2.y')
    def test3(dummy=11, x=3):
        return dummy, x

    @deprecated_params(['dummy', 'x'], None, '0.3', '0.5',
                       alternative='test2.y')
    def test4(dummy=11, x=3):
        return dummy, x

    with pytest.warns(ArgsDeprecationWarning,
                      match=r'Use test2\.y instead') as w:
        npt.assert_equal(test(x=1), (11, 1))
    npt.assert_equal(len(w), 1)

    with pytest.warns(ArgsDeprecationWarning,
                      match=r'Use test2\.y instead') as w:
        npt.assert_equal(test(x=1, dummy=10), (10, 1))
    npt.assert_equal(len(w), 1)

    with pytest.warns(ArgsDeprecationWarning,
                      match=r'Use test2\.y instead'):
        npt.assert_equal(test(121, 1), (121, 1))

    with pytest.warns(ArgsDeprecationWarning,
                      match=r'Use test2\.y instead') as w:
        npt.assert_equal(test3(121, 1), (121, 1))

    npt.assert_raises(ExpiredDeprecationError, test4, 121, 1)
    npt.assert_raises(ExpiredDeprecationError, test4, dummy=121, x=1)
    npt.assert_raises(ExpiredDeprecationError, test4, 121, x=1)
    npt.assert_raises(ExpiredDeprecationError, test2, x=1)
    npt.assert_equal(test(), (11, 3))
    npt.assert_equal(test(121), (121, 3))
    npt.assert_equal(test(dummy=121), (121, 3))
