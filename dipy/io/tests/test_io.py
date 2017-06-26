""" Tests for overall io sub-package
"""

from dipy import io

from nose.tools import assert_false


def test_imports():
    # Make sure io has not pulled in setup_module from dpy
    assert_false(hasattr(io, 'setup_module'))
