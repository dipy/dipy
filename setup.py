#!/usr/bin/env python
''' Installation script for dipy package '''

from os.path import join as pjoin
from glob import glob
from distutils.core import setup
from distutils.extension import Extension
from distutils.version import LooseVersion

import numpy as np

from build_helpers import make_cython_ext

# we use cython to compile the modules
try:
    from Cython.Compiler.Version import version
except ImportError:
    raise RuntimeError('You need cython to build dipy')
if LooseVersion(version) < LooseVersion('0.12.1'):
    raise RuntimeError('Need cython >= 0.12.1 to build dipy')


per_ext, cmdclass = make_cython_ext(
    'dipy.core.track_performance',
    include_dirs = [np.get_include()])

tvol_ext, cmdclass = make_cython_ext(
    'dipy.io.track_volumes',
    include_dirs = [np.get_include()])

rec_ext, cmdclass = make_cython_ext(
    'dipy.core.reconstruction_performance',
    include_dirs = [np.get_include()])

tpp_ext, cmdclass = make_cython_ext(
    'dipy.core.track_propagation_performance',
    include_dirs = [np.get_include()])



setup(name='dipy',
      version='0.11a',
      description='Diffusion utilities in Python',
      author='DIPY Team',
      author_email='nipy-devel@neuroimaging.scipy.org',
      url='http://github.com/Garyfallidis/dipy',
      packages=['dipy', 'dipy.io', 'dipy.core', 'dipy.viz', 'dipy.testing'],
      package_data={'dipy.io': ['tests/data/*', 'tests/*.py']},
      ext_modules = [per_ext,tvol_ext, rec_ext,tpp_ext],
      cmdclass    = cmdclass,
      scripts=glob('scripts/*.py')
      )

