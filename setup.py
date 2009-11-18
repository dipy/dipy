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
    'dipy.core.performance',
    has_cython,
    include_dirs = [np.get_include()])

setup(name='dipy',
      version='0.1a',
      description='Diffusion utilities in Python',
      author='DIPY python team',
      author_email='matthew.brett@gmail.com',
      url='http://github.com/matthew-brett/dipy',
      packages=['dipy', 'dipy.io', 'dipy.core','dipy.viz'],
      package_data={'dipy.io': ['tests/data/*', 'tests/*.py']},
	  ext_modules = [per_ext],
      cmdclass    = cmdclass,      
      scripts=glob('scripts/*.py')
      )

