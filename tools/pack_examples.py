#!/usr/bin/env python
""" Script to pack built examples into suitably named archive
"""

import os
from os.path import join as pjoin
import sys
import glob
import tarfile

import dipy

dpv = 'dipy-' + dipy.__version__
archive_name = dpv + '-doc-examples.tar.gz'

out_root = sys.argv[1]
try:
    os.mkdir(out_root)
except OSError:
    pass
try:
    doc_dir = sys.argv[2]
except IndexError:
    doc_dir = os.getcwd()

archive_fname = os.path.join(out_root, archive_name)

tarobj = tarfile.open(archive_fname, 'w|gz')
eg_built_dir = pjoin(doc_dir, 'examples_built')
for globber in ('*.rst', '*.png', pjoin('figs', '*')):
    globs = glob.glob(pjoin(eg_built_dir, globber))
    for in_fname in globs:
        arc_fname = pjoin(dpv, 'doc', in_fname)
        fobj = tarobj.gettarinfo(in_fname)
        fobj.name = arc_fname
        tarobj.addfile(fobj)
tarobj.close()
