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
from os.path import join as pjoin, abspath
import shutil
from subprocess import check_call
from glob import glob

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

# Work in examples directory
os.chdir('examples_built')
if not os.getcwd().endswith('doc/examples_built'):
    raise OSError('This must be run from doc/examples_built directory')

# Copy the py files
# Finding which py files to copy could come from the examples_index
eg_src_dir = abspath(pjoin('..', 'examples'))
pyfilelist = [fname for fname in os.listdir(eg_src_dir)
              if fname.endswith('.py')]
for fname in pyfilelist:
    shutil.copyfile(pjoin(eg_src_dir, fname), fname)

# Run the conversion from .py to rst file
check_call('../../tools/ex2rst --project dipy --outdir . .',
           shell=True)

# Execute each python script in the directory.
if not os.path.isdir('fig'):
    os.mkdir('fig')

for script in glob('*.py'):
    figure_basename = os.path.join('fig', os.path.splitext(script)[0])
    execfile(script)
    plt.close('all')

