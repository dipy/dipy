"""Lightweight testing that remains unittest-compatible.

This module exposes decorators and classes to make lightweight testing possible
in a manner similar to what nose allows, where standalone functions can be
tests.  It also provides parametric test support that is vastly easier to use
than nose's for debugging, because if a test fails, the stack under inspection
is that of the test and not that of the test framework.

- An @as_unittest decorator can be used to tag any normal parameter-less
  function as a unittest TestCase.  Then, both nose and normal unittest will
  recognize it as such.

Authors
-------

- Fernando Perez <Fernando.Perez@berkeley.edu>
"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2009  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Stdlib
import sys
import unittest

# Our own
import nosepatch

if sys.version[0]=='2':
    from _paramtestpy2 import ParametricTestCase, parametric
else:
    from _paramtestpy3 import ParametricTestCase, parametric

#-----------------------------------------------------------------------------
# Classes and functions
#-----------------------------------------------------------------------------

# Simple example of the basic idea
def as_unittest(func):
    """Decorator to make a simple function into a normal test via unittest."""
    class Tester(unittest.TestCase):
        def test(self):
            func()

    Tester.__name__ = func.__name__

    return Tester
