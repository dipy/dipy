#!/usr/bin/env python
"""Script to auto-generate our API docs.
"""
# stdlib imports
import sys
import re

# local imports
from apigen import ApiDocWriter

# version comparison
from distutils.version import LooseVersion as V

#*****************************************************************************

def abort(error):
    print('*WARNING* API documentation not generated: %s' % error)
    exit()

if __name__ == '__main__':
    package = 'dipy'

    # Check that the 'image' package is available. If not, the API
    # documentation is not (re)generated and existing API documentation
    # sources will be used.

    try:
        __import__(package)
    except ImportError, e:
        abort("Can not import dipy")

    module = sys.modules[package]

    # Check that the source version is equal to the installed
    # version. If the versions mismatch the API documentation sources
    # are not (re)generated. This avoids automatic generation of documentation
    # for older or newer versions if such versions are installed on the system.

    installed_version = V(module.__version__)

    info_lines = open('../dipy/info.py').readlines()
    source_version = '.'.join([v.split('=')[1].strip(" '\n.")
                               for v in info_lines if re.match(
                                       '^_version_(major|minor|micro|extra)', v
                                       )])
    print '***', source_version

    if source_version != installed_version:
        abort("Installed version does not match source version")

    outdir = 'reference'
    docwriter = ApiDocWriter(package, rst_extension='.rst')
    docwriter.package_skip_patterns += [r'\.fixes$',
                                        r'\.externals$',
                                        r'\.reconst.eit$',
                                        r'\.tracking\.interfaces.*$',
                                        r'\.tracking\.gui_tools.*$',
                                        r'.*test.*$',
                                        r'\.utils.*$',
                                        r'\.viz.*$',
                                        r'\.boots\.resampling.*$',
                                        r'\.fixes.*$',
                                        r'\.info.*$',
                                        r'\.pkg_info.*$',
                                        ]
    docwriter.write_api_docs(outdir)
    docwriter.write_index(outdir, 'index', relative_to='reference')
    print('%d files written' % len(docwriter.written_modules))
