#!/usr/bin/env python
''' Installation script for dipy package '''

import os
import sys
from os.path import join as pjoin
from glob import glob

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

import numpy as np

# For some commands, use setuptools
if len(set(('develop', 'bdist_egg', 'bdist_rpm', 'bdist', 'bdist_dumb',
            'bdist_wininst', 'install_egg_info', 'egg_info', 'easy_install',
            )).intersection(sys.argv)) > 0:
    # setup_egg imports setuptools setup, thus monkeypatching distutils. 
    from setup_egg import extra_setuptools_args

# Import distutils _after_ potential setuptools import above, and after removing
# MANIFEST
from distutils.core import setup
from distutils.extension import Extension

# extra_setuptools_args can be defined from the line above, but it can
# also be defined here because setup.py has been exec'ed from
# setup_egg.py.
if not 'extra_setuptools_args' in globals():
    extra_setuptools_args = dict()

from nisext.sexts import get_comrec_build, package_check
cmdclass = {'build_py': get_comrec_build('dipy')}

# Get version and release info, which is all stored in dipy/info.py
ver_file = os.path.join('dipy', 'info.py')
execfile(ver_file)

# Do dependency checking
package_check('numpy', NUMPY_MIN_VERSION)
package_check('scipy', SCIPY_MIN_VERSION)
package_check('nibabel', NIBABEL_MIN_VERSION)
# Cython can be a build dependency
def _cython_version(pkg_name):
    from Cython.Compiler.Version import version
    return version
package_check('cython',
              CYTHON_MIN_VERSION,
              version_getter=_cython_version)

if 'setuptools' in sys.modules:
    extra_setuptools_args['extras_require'] = dict(
        doc='Sphinx>=1.0',
        test='nose>=0.10.1')

# we use cython to compile the modules
from Cython.Distutils import build_ext
cmdclass['build_ext'] = build_ext
EXTS = []
for modulename, other_sources in (
    ('dipy.reconst.recspeed', []),
    ('dipy.tracking.distances', []),
    ('dipy.tracking.vox2track', []),
    ('dipy.tracking.propspeed', [])):
    pyx_src = pjoin(*modulename.split('.')) + '.pyx'
    EXTS.append(Extension(modulename,[pyx_src] + other_sources,include_dirs = [np.get_include()]))


def main(**extra_args):
    setup(name=NAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          url=URL,
          download_url=DOWNLOAD_URL,
          license=LICENSE,
          classifiers=CLASSIFIERS,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          platforms=PLATFORMS,
          version=VERSION,
          requires=REQUIRES,
          provides=PROVIDES,
          packages     = ['dipy',
                          'dipy.align',
                          'dipy.core',
                          'dipy.core.tests',
                          'dipy.tracking',
                          'dipy.tracking.tests',
                          'dipy.reconst',
                          'dipy.reconst.tests',                          
                          'dipy.io',
                          'dipy.io.tests',
                          'dipy.viz',
                          'dipy.viz.tests',
                          'dipy.testing',
                          'dipy.boots',
                          'dipy.data',
                          'dipy.utils',
                          'dipy.external',
                          'dipy.external.tests'],
          ext_modules = EXTS,
          # The package_data spec has no effect for me (on python 2.6) -- even
          # changing to data_files doesn't get this stuff included in the source
          # distribution -- not sure if it has something to do with the magic
          # above, but distutils is surely the worst piece of code in all of
          # python -- duplicating things into MANIFEST.in but this is admittedly
          # only a workaround to get things started -- not a solution
          package_data = {'dipy':
                          [pjoin('data', '*')
                          ]},
          data_files=[('share/doc/dipy/examples', glob(pjoin('doc','examples','*.py')))],                                       
          scripts      = glob(pjoin('scripts', '*')),
          cmdclass = cmdclass,
          **extra_args
         )

#simple way to test what setup will do
#python setup.py install --prefix=/tmp
if __name__ == "__main__":
    main(**extra_setuptools_args)
