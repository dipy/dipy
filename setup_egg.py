#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Wrapper to run setup.py using setuptools."""

import sys
from os.path import dirname, join as pjoin
# Deal with setuptools monkeypatching bug for Pyrex
sys.path.insert(0, pjoin(dirname(__file__), 'fake_pyrex'))

import setuptools

# Get version and release info, which is all stored in dipy/info.py
ver_file = pjoin('dipy', 'info.py')
execfile(ver_file)

extra_setuptools_args = dict(
    tests_require=['nose'],
    test_suite='nose.collector',
    zip_safe=False,
    extras_require = dict(
        doc=['Sphinx>=1.0'],
        test=['nose>=0.10.1']),
    install_requires = ['nibabel>=' + NIBABEL_MIN_VERSION])
# I removed numpy and scipy from install requires because easy_install seems
# to want to fetch these if they are already installed, meaning of course
# that there's a long fragile and unnecessary compile before the install
# finishes.

if __name__ == '__main__':
    execfile('setup.py', dict(__name__='__main__',
                              extra_setuptools_args=extra_setuptools_args))



