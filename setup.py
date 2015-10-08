#!/usr/bin/env python
''' Installation script for dipy package '''

import numpy as np
import os
import sys
from copy import deepcopy
from os.path import join as pjoin, dirname
from glob import glob

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')


# Get version and release info, which is all stored in dipy/info.py
ver_file = os.path.join('dipy', 'info.py')
# Use exec for compabibility with Python 3
exec(open(ver_file).read())

# force_setuptools can be set from the setup_egg.py script
if not 'force_setuptools' in globals():
    # For some commands, always use setuptools
    if len(set(('develop', 'bdist_egg', 'bdist_rpm', 'bdist', 'bdist_dumb',
                'bdist_mpkg', 'bdist_wheel', 'install_egg_info', 'egg_info',
                'easy_install')).intersection(sys.argv)) > 0:
        force_setuptools = True
    else:
        force_setuptools = False

if force_setuptools:
    import setuptools

# We may just have imported setuptools, or we may have been exec'd from a
# setuptools environment like pip
if 'setuptools' in sys.modules:
    # Try to preempt setuptools monkeypatching of Extension handling when Pyrex
    # is missing.  Otherwise the monkeypatched Extension will change .pyx
    # filenames to .c filenames, and we probably don't have the .c files.
    sys.path.insert(0, pjoin(dirname(__file__), 'fake_pyrex'))
    # Set setuptools extra arguments
    nibabel_spec = 'nibabel>=' + NIBABEL_MIN_VERSION
    extra_setuptools_args = dict(
        tests_require=['nose'],
        test_suite='nose.collector',
        zip_safe=False,
        extras_require = dict(
            doc=['Sphinx>=1.0'],
            test=['nose>=0.10.1']),
        install_requires = [nibabel_spec])
    # I removed numpy and scipy from install requires because easy_install seems
    # to want to fetch these if they are already installed, meaning of course
    # that there's a long fragile and unnecessary compile before the install
    # finishes.
    # We need setuptools install command because we're going to override it
    # further down.  Using distutils install command causes some confusion, due
    # to the Pyrex / setuptools hack above (force_setuptools)
    from setuptools.command import install
    # If running setuptools and nibabel is not installed, we have to force
    # setuptools to install nibabel locally for the script to continue.  This
    # hack is from
    # http://stackoverflow.com/questions/12060925/best-way-to-share-code-across-several-setup-py-scripts
    # with thanks
    from setuptools.dist import Distribution
    Distribution(dict(setup_requires=nibabel_spec))
else:
    extra_setuptools_args = {}
    from distutils.command import install

# Import distutils _after_ potential setuptools import above, and after removing
# MANIFEST
from distutils.core import setup
from distutils.extension import Extension
from distutils.command import build_py, build_ext

from cythexts import cyproc_exts, get_pyx_sdist, derror_maker
from setup_helpers import install_scripts_bat, add_flag_checking

# Define extensions
EXTS = []

# We use some defs from npymath, but we don't want to link against npymath lib
ext_kwargs = {'include_dirs':[np.get_include()]}
ext_kwargs['include_dirs'].append('src')

for modulename, other_sources, language in (
    ('dipy.reconst.peak_direction_getter', [], 'c'),
    ('dipy.reconst.recspeed', [], 'c'),
    ('dipy.reconst.vec_val_sum', [], 'c'),
    ('dipy.reconst.quick_squash', [], 'c'),
    ('dipy.tracking.distances', [], 'c'),
    ('dipy.tracking.streamlinespeed', [], 'c'),
    ('dipy.tracking.local.localtrack', [], 'c'),
    ('dipy.tracking.local.direction_getter', [], 'c'),
    ('dipy.tracking.local.tissue_classifier', [], 'c'),
    ('dipy.tracking.local.interpolation', [], 'c'),
    ('dipy.tracking.vox2track', [], 'c'),
    ('dipy.tracking.propspeed', [], 'c'),
    ('dipy.segment.cythonutils', [], 'c'),
    ('dipy.segment.featurespeed', [], 'c'),
    ('dipy.segment.metricspeed', [], 'c'),
    ('dipy.segment.clusteringspeed', [], 'c'),
    ('dipy.segment.clustering_algorithms', [], 'c'),
    ('dipy.denoise.denspeed', [], 'c'),
    ('dipy.align.vector_fields', [], 'c'),
    ('dipy.align.sumsqdiff', [], 'c'),
    ('dipy.align.expectmax', [], 'c'),
    ('dipy.align.crosscorr', [], 'c'),
    ('dipy.align.bundlemin', [], 'c'),
    ('dipy.align.transforms', [], 'c'),
    ('dipy.align.parzenhist', [], 'c')):

    pyx_src = pjoin(*modulename.split('.')) + '.pyx'
    EXTS.append(Extension(modulename, [pyx_src] + other_sources,
                          language=language,
                          **deepcopy(ext_kwargs)))  # deepcopy lists


# Do our own build and install time dependency checking. setup.py gets called in
# many different ways, and may be called just to collect information (egg_info).
# We need to set up tripwires to raise errors when actually doing things, like
# building, rather than unconditionally in the setup.py import or exec
# We may make tripwire versions of build_ext, build_py, install
try:
    from nisext.sexts import package_check, get_comrec_build
except ImportError: # No nibabel
    msg = ('Need nisext package from nibabel installation'
           ' - please install nibabel first')
    pybuilder = derror_maker(build_py.build_py, msg)
    extbuilder = derror_maker(build_ext.build_ext, msg)
    def package_check(*args, **kwargs):
        raise RuntimeError(msg + " or try 'python setup_egg.py install'")
else: # We have nibabel
    pybuilder = get_comrec_build('dipy')
    # Cython is a dependency for building extensions, iff we don't have stamped
    # up pyx and c files.
    build_ext = cyproc_exts(EXTS, CYTHON_MIN_VERSION, 'pyx-stamps')
    # Add openmp flags if they work
    simple_test_c = """int main(int argc, char** argv) { return(0); }"""
    omp_test_c = """#include <omp.h>
    int main(int argc, char** argv) { return(0); }"""
    extbuilder = add_flag_checking(
        build_ext, [[['/arch:SSE2'], [], simple_test_c, 'USING_VC_SSE2'],
            [['-msse2', '-mfpmath=sse'], [], simple_test_c, 'USING_GCC_SSE2'],
            [['-fopenmp'], ['-fopenmp'], omp_test_c, 'HAVE_OPENMP']], 'dipy')

# Installer that checks for install-time dependencies
class installer(install.install):
    def run(self):
        package_check('numpy', NUMPY_MIN_VERSION)
        package_check('scipy', SCIPY_MIN_VERSION)
        package_check('nibabel', NIBABEL_MIN_VERSION)
        install.install.run(self)


cmdclass = dict(
    build_py=pybuilder,
    build_ext=extbuilder,
    install=installer,
    install_scripts=install_scripts_bat,
    sdist=get_pyx_sdist(include_dirs=['src']))


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
                          'dipy.tests',
                          'dipy.align',
                          'dipy.align.tests',
                          'dipy.core',
                          'dipy.core.tests',
                          'dipy.direction',
                          'dipy.direction.tests',
                          'dipy.tracking',
                          'dipy.tracking.local',
                          'dipy.tracking.local.tests',
                          'dipy.tracking.tests',
                          'dipy.tracking.benchmarks',
                          'dipy.reconst',
                          'dipy.reconst.benchmarks',
                          'dipy.reconst.tests',
                          'dipy.io',
                          'dipy.io.tests',
                          'dipy.viz',
                          'dipy.viz.tests',
                          'dipy.testing',
                          'dipy.testing.tests',
                          'dipy.boots',
                          'dipy.data',
                          'dipy.utils',
                          'dipy.utils.tests',
                          'dipy.fixes',
                          'dipy.external',
                          'dipy.external.tests',
                          'dipy.segment',
                          'dipy.segment.benchmarks',
                          'dipy.segment.tests',
                          'dipy.sims',
                          'dipy.sims.tests',
                          'dipy.denoise',
                          'dipy.denoise.tests'],

          ext_modules = EXTS,
          # The package_data spec has no effect for me (on python 2.6) -- even
          # changing to data_files doesn't get this stuff included in the source
          # distribution -- not sure if it has something to do with the magic
          # above, but distutils is surely the worst piece of code in all of
          # python -- duplicating things into MANIFEST.in but this is admittedly
          # only a workaround to get things started -- not a solution
          package_data = {'dipy':
                          [pjoin('data', 'files', '*')
                          ]},
          data_files=[('share/doc/dipy/examples',
                       glob(pjoin('doc','examples','*.py')))],
          scripts      = [pjoin('bin', 'dipy_peak_extraction'),
                          pjoin('bin', 'dipy_fit_tensor'),
                          pjoin('bin', 'dipy_sh_estimate'),
                          pjoin('bin', 'dipy_quickbundles')],
          cmdclass = cmdclass,
          **extra_args
        )


#simple way to test what setup will do
#python setup.py install --prefix=/tmp
if __name__ == "__main__":
    main(**extra_setuptools_args)
