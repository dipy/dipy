#!/usr/bin/env python
"""Run the py->rst conversion and run all examples.

This also creates the index.rst file appropriately, makes figures, etc.
"""
#-----------------------------------------------------------------------------
# Library imports
#-----------------------------------------------------------------------------

# Stdlib imports
import os
from subprocess import check_call
from glob import glob

# Third-party imports

# We must configure the mpl backend before making any further mpl imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib._pylab_helpers import Gcf

def sh(cmd):
    """Execute command in a subshell, return status code."""
    return check_call(cmd, shell=True)
#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------

examples_header = """

.. _examples:

========
Examples
========

.. toctree::
   :maxdepth: 2
   
   note_about_examples
"""
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
os.chdir('examples')
if not os.getcwd().endswith('doc/examples'):
    raise OSError('This must be run from doc/examples directory')

# Run the conversion from .py to rst file
sh('../../tools/ex2rst --project dipy --outdir . .')

# Make the index.rst file
with open('index.rst', 'w') as index:
    index.write(examples_header)
    for name in [os.path.splitext(f)[0] for f in glob('*.rst')]:
        #Don't add the index in there to avoid sphinx errors and don't add the
        #note_about examples again (because it was added at the top):
        if name not in(['index','note_about_examples']):
            index.write('   %s\n' % name)

# Execute each python script in the directory.
if not os.path.isdir('fig'):
    os.mkdir('fig')
    
for script in glob('*.py'):
    figure_basename = os.path.join('fig', os.path.splitext(script)[0])
    execfile(script)
    plt.close('all')


