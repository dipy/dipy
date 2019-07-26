#!/usr/bin/env python
"""Run the py->rst conversion and run all examples.

Steps are:
    analyze example index file for example py filenames
    check for any filenames in example directory not included
    do py to rst conversion, writing into build directory
    run
"""
# -----------------------------------------------------------------------------
# Library imports
# -----------------------------------------------------------------------------

# Stdlib imports
import os
import os.path as op
import sys
import shutil
import io
from subprocess import check_call
from glob import glob
from time import time

# Third-party imports


# We must configure the mpl backend before making any further mpl imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib._pylab_helpers import Gcf

import dipy

# -----------------------------------------------------------------------------
# Function defintions
# -----------------------------------------------------------------------------

# These global variables let show() be called by the scripts in the usual
# manner, but when generating examples, we override it to write the figures to
# files with a known name (derived from the script name) plus a counter
figure_basename = None


# We must change the show command to save instead
def show():
    allfm = Gcf.get_all_fig_managers()
    for fcount, fm in enumerate(allfm):
        fm.canvas.figure.savefig('%s_%02i.png' %
                                 (figure_basename, fcount + 1))

_mpl_show = plt.show
plt.show = show

# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

# Where things are
DOC_PATH = op.abspath('..')
EG_INDEX_FNAME = op.join(DOC_PATH, 'examples_index.rst')
EG_SRC_DIR = op.join(DOC_PATH, 'examples')

# Work in examples directory
# os.chdir(op.join(DOC_PATH, 'examples_built'))

if not os.getcwd().endswith(op.join('doc', 'examples_built')):
    raise OSError('This must be run from the doc directory')

# Copy the py files; check they are in the examples list and warn if not
with io.open(EG_INDEX_FNAME, 'rt', encoding="utf8") as f:
    eg_index_contents = f.read()

# Here I am adding an extra step. The list of examples to be executed need
# also to be added in the following file (valid_examples.txt). This helps
# with debugging the examples and the documentation only a few examples at
# the time.
flist_name = op.join(op.dirname(os.getcwd()), 'examples',
                     'valid_examples.txt')

with io.open(flist_name, "r", encoding="utf8") as flist:
    validated_examples = flist.readlines()

# Parse "#" in lines
validated_examples = [line.split("#", 1)[0] for line in validated_examples]
# Remove leading and trailing white space from example names
validated_examples = [line.strip() for line in validated_examples]
# Remove blank lines
validated_examples = list(filter(None, validated_examples))

for example in validated_examples:
    fullpath = op.join(EG_SRC_DIR, example)
    if not example.endswith(".py"):
        print("%s not a python file, skipping." % example)
        continue
    elif not op.isfile(fullpath):
        print("Cannot find file, %s, skipping." % example)
        continue
    shutil.copyfile(fullpath, example)

    # Check that example file is included in the docs
    file_root = example[:-3]
    if file_root not in eg_index_contents:
        msg = "Example, %s, not in index file %s."
        msg = msg % (example, EG_INDEX_FNAME)
        print(msg)

# Run the conversion from .py to rst file
check_call('{} ../../tools/ex2rst --project dipy --outdir . .'.format(sys.executable), shell=True)

# added the path so that scripts can import other scripts on the same directory
sys.path.insert(0, os.getcwd())

if not op.isdir('fig'):
    os.mkdir('fig')

use_xvfb = os.environ.get('TEST_WITH_XVFB', False)
use_memprof = os.environ.get('TEST_WITH_MEMPROF', False)

if use_xvfb:
    try:
        from xvfbwrapper import Xvfb
    except ImportError:
        raise RuntimeError("You are trying to run a documentation build",
                           "with 'TEST_WITH_XVFB' set to True, but ",
                           "xvfbwrapper is not available. Please install",
                           "xvfbwrapper and try again")

    display = Xvfb(width=1920, height=1080)
    display.start()

if use_memprof:
    try:
        import memory_profiler
    except ImportError:
        raise RuntimeError("You are trying to run a documentation build",
                           "with 'TEST_WITH_MEMPROF' set to True, but ",
                           "memory_profiler is not available. Please install",
                           "memory_profiler and try again")

name = ''


def run_script():
    namespace = {}
    t1 = time()
    with io.open(script, encoding="utf8") as f:
        exec(f.read(), namespace)
    t2 = time()
    print("That took %.2f seconds to run" % (t2 - t1))
    plt.close('all')
    del namespace


# Execute each python script in the directory:
for script in validated_examples:
    figure_basename = op.join('fig', op.splitext(script)[0])
    if use_memprof:
        print("memory profiling ", script)
        memory_profiler.profile(run_script)()

    else:
        print('*************************************************************')
        print(script)
        print('*************************************************************')
        run_script()

if use_xvfb:
    display.stop()

# clean up stray images, pickles, npy files, etc
for globber in ('*.nii.gz', '*.dpy', '*.npy', '*.pkl', '*.mat', '*.img',
                '*.hdr'):
    for fname in glob(globber):
        os.unlink(fname)
