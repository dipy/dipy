#!/usr/bin/env python
"""Run the py->rst conversion and run all examples.

Steps are:
    analyze example index file for example py filenames
    check for any filenames in example directory not included
    do py to rst conversion, writing into build directory
    run
"""
#-----------------------------------------------------------------------------
# Library imports
#-----------------------------------------------------------------------------

# Stdlib imports
import os
from os.path import join as pjoin, abspath, splitext
import sys
import shutil
from subprocess import check_call
from glob import glob
import numpy as np

# Third-party imports

# We must configure the mpl backend before making any further mpl imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib._pylab_helpers import Gcf

#-----------------------------------------------------------------------------
# Function defintions
#-----------------------------------------------------------------------------

# These global variables let show() be called by the scripts in the usual
# manner, but when generating examples, we override it to write the figures to
# files with a known name (derived from the script name) plus a counter
figure_basename = None

# We must change the show command to save instead
def show():
    allfm = Gcf.get_all_fig_managers()
    for fcount, fm in enumerate(allfm):
        fm.canvas.figure.savefig('%s_%02i.png' %
                                 (figure_basename, fcount+1))

_mpl_show = plt.show
plt.show = show

#-----------------------------------------------------------------------------
# Main script
#-----------------------------------------------------------------------------

# Where things are
EG_INDEX_FNAME = abspath('examples_index.rst')
EG_SRC_DIR = abspath('examples')

# Work in examples directory
os.chdir('examples_built')

if not os.getcwd().endswith(pjoin('doc','examples_built')):
    raise OSError('This must be run from the doc directory')

# Copy the py files; check they are in the examples list and warn if not
eg_index_contents = open(EG_INDEX_FNAME, 'rt').read()
pyfilelist = [fname for fname in os.listdir(EG_SRC_DIR)
              if fname.endswith('.py')]

# Here I am adding an extra step. The list of examples to be executed need
# also to be added in the following file (valid_examples.txt). This helps
# with debugging the examples and the documentation only a few examples at
# the time.
flist_name = pjoin(os.path.dirname(os.getcwd()), 'examples', 'valid_examples.txt')
flist = open(flist_name, "r")
validated_examples = flist.readlines()
flist.close()

def check_enabled(froot, example):
    if froot in example:
        if example.startswith('#'):
            return False
        return True
    return False

for fname in pyfilelist:

    froot, _ = splitext(fname)
    if np.sum([1 for example in validated_examples if check_enabled(froot, example)]) > 0:

        shutil.copyfile(pjoin(EG_SRC_DIR, fname), fname)

        if froot not in eg_index_contents:
            print 'Example %s not in index file %s' % (EG_SRC_DIR, EG_INDEX_FNAME)

# Run the conversion from .py to rst file
check_call('python ../../tools/ex2rst --project dipy --outdir . .',
            shell=True)

#added the path so that scripts can import other scripts on the same directory
sys.path.insert(0, os.getcwd())

# Execute each python script in the directory.
if not os.path.isdir('fig'):
    os.mkdir('fig')

for script in glob('*.py'):
    figure_basename = os.path.join('fig', os.path.splitext(script)[0])
    print script

    execfile(script)
    plt.close('all')

# clean up stray images, pickles, npy files, etc
for globber in ('*.nii.gz', '*.dpy', '*.npy', '*.pkl', '*.mat', '*.img',
                '*.hdr'):
    for fname in glob(globber):
        os.unlink(fname)

