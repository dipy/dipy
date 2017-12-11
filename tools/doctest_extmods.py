#!/usr/bin/env python
"""Run doctests in extension modules of <pkg_name>

Collect extension modules in <pkg_name>

Run doctests in each extension module

Examples
----------
    %prog dipy
"""

import sys
import os
from os.path import dirname, relpath, sep, join as pjoin, splitext, abspath

from distutils.sysconfig import get_config_vars

import doctest
from optparse import OptionParser

EXT_EXT = get_config_vars('SO')[0]


def get_ext_modules(pkg_name):
    pkg = __import__(pkg_name, fromlist=[''])
    pkg_dir = abspath(dirname(pkg.__file__))
    # pkg_root = __import__(pkg_name)
    ext_modules = []
    for dirpath, dirnames, filenames in os.walk(pkg_dir):
        reldir = relpath(dirpath, pkg_dir)
        if reldir == '.':
            reldir = ''
        for filename in filenames:
            froot, ext = splitext(filename)
            if ext == EXT_EXT:
                mod_path = pjoin(reldir, froot)
                mod_uri = pkg_name + '.' + mod_path.replace(sep, '.')
                # fromlist=[''] results in submodule being returned, rather than the
                # top level module.  See help(__import__)
                mod = __import__(mod_uri, fromlist=[''])
                ext_modules.append(mod)
    return ext_modules


def main():
    usage = "usage: %prog [options] <pkg_name>\n\n" + __doc__
    parser = OptionParser(usage=usage)
    opts, args = parser.parse_args()
    if len(args) == 0:
        parser.print_help()
        sys.exit(1)
    mod_name = args[0]
    mods = get_ext_modules(mod_name)
    for mod in mods:
        print("Testing module: " + mod.__name__)
        doctest.testmod(mod)


if __name__ == '__main__':
    main()
