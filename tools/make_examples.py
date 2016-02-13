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

# Here I am adding an extra step. The list of examples to be executed need
# also to be added in the following file (valid_examples.txt). This helps
# with debugging the examples and the documentation only a few examples at
# the time.
flist_name = pjoin(os.path.dirname(os.getcwd()), 'examples',
                   'valid_examples.txt')
flist = open(flist_name, "r")
validated_examples = flist.readlines()
flist.close()

# Parse "#" in lines
validated_examples = [line.split("#", 1)[0] for line in validated_examples]
# Remove leading and trailing white space from example names
validated_examples = [line.strip() for line in validated_examples]
# Remove blank lines
validated_examples = filter(None, validated_examples)

for example in validated_examples:
    fullpath = pjoin(EG_SRC_DIR, example)
    if not example.endswith(".py"):
        print ("%s not a python file, skipping." % example)
        continue
    elif not os.path.isfile(fullpath):
        print ("Cannot find file, %s, skipping." % example)
        continue
    shutil.copyfile(fullpath, example)

    # Check that example file is included in the docs
    file_root = example[:-3]
    if file_root not in eg_index_contents:
        msg = "Example, %s, not in index file %s."
        msg = msg % (example, EG_INDEX_FNAME)
        print(msg)

# Run the conversion from .py to rst file
check_call('python ../../tools/ex2rst --project dipy --outdir . .', shell=True)

# added the path so that scripts can import other scripts on the same directory
sys.path.insert(0, os.getcwd())

# Execute each python script in the directory.
if not os.path.isdir('fig'):
    os.mkdir('fig')

use_xvfb = os.environ.get('TEST_WITH_XVFB', False)

if use_xvfb:
    from xvfbwrapper import Xvfb
    display = Xvfb(width=1920, height=1080)
    display.start()

for script in validated_examples:
    namespace = {}
    figure_basename = os.path.join('fig', os.path.splitext(script)[0])
    print(script)
    execfile(script, namespace)
    del namespace
    # plt.close('all')

if use_xvfb:
    display.stop()

# clean up stray images, pickles, npy files, etc
for globber in ('*.nii.gz', '*.dpy', '*.npy', '*.pkl', '*.mat', '*.img',
                '*.hdr'):
    for fname in glob(globber):
        os.unlink(fname)
