#!/usr/bin/env python3
"""Run doctests in extension modules of <pkg_name>

Collect extension modules in <pkg_name>

Run doctests in each extension module

Examples
----------
    %prog dipy
"""

from distutils.sysconfig import get_config_vars
import doctest
from optparse import OptionParser
import os
from os.path import abspath, dirname, join as pjoin, relpath, sep
import sys

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
            if filename.endswith(EXT_EXT):
                froot = filename[:-len(EXT_EXT)]
                mod_path = pjoin(reldir, froot)
                mod_uri = pkg_name + '.' + mod_path.replace(sep, '.')
                # fromlist=[''] results in submodule being returned, rather
                # than the top level module.  See help(__import__)
                mod = __import__(mod_uri, fromlist=[''])
                ext_modules.append(mod)
    return ext_modules


def main():
    usage = f"usage: %prog [options] <pkg_name>\n\n{__doc__}"
    parser = OptionParser(usage=usage)
    opts, args = parser.parse_args()
    if len(args) == 0:
        parser.print_help()
        sys.exit(1)
    mod_name = args[0]
    mods = get_ext_modules(mod_name)
    for mod in mods:
        print(f"Testing module: {mod.__name__}")
        doctest.testmod(mod, optionflags=doctest.NORMALIZE_WHITESPACE)


if __name__ == '__main__':
    main()
