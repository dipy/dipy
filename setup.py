#!/usr/bin/env python

''' Installation script for dipy package '''

from os.path import join as pjoin
from glob import glob
from distutils.core import setup
from distutils.extension import Extension
import numpy as np

from build_helpers import make_cython_ext

# we use cython to compile the module if we have it
try:
    import Cython
except ImportError:
    has_cython = False
else:
    has_cython = True
    
per_ext, cmdclass = make_cython_ext(
    'dipy.core.track_performance',
    has_cython,
    include_dirs = [np.get_include()])


tvol_ext, cmdclass = make_cython_ext(
    'dipy.io.track_volumes',
    has_cython,
    include_dirs = [np.get_include()])


setup(name='dipy',
      version='0.1a',
      description='Diffusion utilities in Python',
      author='DIPY Team',
      author_email='nipy-devel@neuroimaging.scipy.org',
      url='http://github.com/Garyfallidis/dipy',
      packages=['dipy', 'dipy.io', 'dipy.core','dipy.viz'],
      package_data={'dipy.io': ['tests/data/*', 'tests/*.py']},
      ext_modules = [per_ext,tvol_ext],
      cmdclass    = cmdclass,      
      scripts=glob('scripts/*.py')
      )

