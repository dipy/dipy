#!/usr/bin/env python
""" Make documentation for module

Depends on some guessed filepaths

Filepaths guessed by importing
"""

import sys
from os.path import join as pjoin, dirname, abspath
ROOT_DIR = abspath(pjoin(dirname(__file__), '..'))
DOC_SDIR = pjoin(ROOT_DIR, 'doc', 'reference')

TEMPLATE = \
""":mod:`%s`
=========================

.. automodule:: %s
    :members:
"""

def main():
    try:
        mod_name = sys.argv[1]
    except IndexError:
        raise OSError('Need module import as input')
    out_fname = pjoin(DOC_SDIR, mod_name + '.rst')
    open(out_fname, 'wt').write(TEMPLATE % (mod_name,
                                            mod_name))


if __name__ == '__main__':
    main()
