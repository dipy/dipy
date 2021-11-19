#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Wrapper to run setup.py using setuptools."""

if __name__ == '__main__':
    with open('setup.py') as f:
        exec(f.read(), dict(__name__='__main__',
                            __file__='setup.py',  # needed in setup.py
                            force_setuptools=True))
