#!/usr/bin/env python
""" Script to pack built examples into suitably named archive

Usage %s output_dir [doc_dir]
"""

import os
from os.path import join as pjoin
import sys
import shutil
import tarfile

import dipy

__doc__ = __doc__ % sys.argv[0]

EG_BUILT_SDIR = 'examples_built'
dpv = 'dipy-' + dipy.__version__
archive_name = dpv + '-doc-examples.tar.gz'

try:
    out_root = sys.argv[1]
except IndexError:
    print(__doc__)
    sys.exit(1)
try:
    os.mkdir(out_root)
except OSError:
    pass
try:
    doc_dir = sys.argv[2]
except IndexError:
    doc_dir = os.getcwd()

archive_fname = os.path.join(out_root, archive_name)

eg_built_dir = pjoin(doc_dir, EG_BUILT_SDIR)
eg_out_base = pjoin(out_root, dpv, 'doc')
eg_out_dir = pjoin(eg_out_base, EG_BUILT_SDIR)
if os.path.isdir(eg_out_dir):
    shutil.rmtree(eg_out_dir)


def ignorandi(src, names):
    return [name for name in names if name == 'README' or name == '.gitignore']


shutil.copytree(eg_built_dir, eg_out_dir, ignore=ignorandi)
os.chdir(out_root)
tar = tarfile.open(archive_fname, 'w|gz')
tar.add(dpv)
tar.close()
shutil.rmtree(pjoin(out_root, dpv))
print("Written " + archive_fname)
