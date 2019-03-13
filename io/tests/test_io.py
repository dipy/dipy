"""Tests for overall io sub-package."""

from dipy import io


def test_imports():
    # Make sure io has not pulled in setup_module from dpy
    assert not hasattr(io, 'setup_module')
