#!/usr/bin/env python
""" Installation script for dipy package """

import os
import sys
import platform
from copy import deepcopy
from os.path import join as pjoin, dirname, exists
from glob import glob

# BEFORE importing setuptools, remove MANIFEST. setuptools doesn't properly
# update it when the contents of directories change.
if exists('MANIFEST'):
    os.remove('MANIFEST')

# force_setuptools can be set from the setup_egg.py script
if 'force_setuptools' not in globals():
    # For some commands, always use setuptools
    if len(set(('develop', 'bdist_egg', 'bdist_rpm', 'bdist', 'bdist_dumb',
                'bdist_mpkg', 'bdist_wheel', 'install_egg_info', 'egg_info',
                'easy_install')).intersection(sys.argv)) > 0:
        force_setuptools = True
    else:
        force_setuptools = False

if force_setuptools:
    import setuptools

# Import setuptools _after_ potential setuptools import above, and after
# removing MANIFEST
from setuptools import setup
from setuptools.extension import Extension

from cythexts import cyproc_exts, get_pyx_sdist
from setup_helpers import (install_scripts_bat, add_flag_checking,
                           SetupDependency, read_vars_from,
                           make_np_ext_builder)
from version_helpers import get_comrec_build

# Get version and release info, which is all stored in dipy/info.py
info = read_vars_from(pjoin('dipy', 'info.py'))

# We may just have imported setuptools, or we may have been exec'd from a
# setuptools environment like pip
using_setuptools = 'setuptools' in sys.modules
extra_setuptools_args = {}
if using_setuptools:
    # Try to preempt setuptools monkeypatching of Extension handling when Pyrex
    # is missing.  Otherwise the monkeypatched Extension will change .pyx
    # filenames to .c filenames, and we probably don't have the .c files.
    sys.path.insert(0, pjoin(dirname(__file__), 'fake_pyrex'))
    # Set setuptools extra arguments
    extra_setuptools_args = dict(
        tests_require=['pytest'],
        zip_safe=False,
        extras_require=info.EXTRAS_REQUIRE,
        python_requires=">= 3.6",
        )

# Define extensions
EXTS = []

# We use some defs from npymath, but we don't want to link against npymath lib
ext_kwargs = {
    'include_dirs': ['src'],  # We add np.get_include() later
    'define_macros': [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    }

for modulename, other_sources, language in (
        ('dipy.core.interpolation', [], 'c'),
        ('dipy.direction.pmf', [], 'c'),
        ('dipy.direction.probabilistic_direction_getter', [], 'c'),
        ('dipy.direction.closest_peak_direction_getter', [], 'c'),
        ('dipy.direction.bootstrap_direction_getter', [], 'c'),
        ('dipy.reconst.eudx_direction_getter', [], 'c'),
        ('dipy.reconst.recspeed', [], 'c'),
        ('dipy.reconst.vec_val_sum', [], 'c'),
        ('dipy.reconst.quick_squash', [], 'c'),
        ('dipy.tracking.distances', [], 'c'),
        ('dipy.tracking.streamlinespeed', [], 'c'),
        ('dipy.tracking.localtrack', [], 'c'),
        ('dipy.tracking.direction_getter', [], 'c'),
        ('dipy.tracking.stopping_criterion', [], 'c'),
        ('dipy.tracking.vox2track', [], 'c'),
        ('dipy.tracking.propspeed', [], 'c'),
        ('dipy.tracking.fbcmeasures', [], 'c'),
        ('dipy.segment.cythonutils', [], 'c'),
        ('dipy.segment.featurespeed', [], 'c'),
        ('dipy.segment.metricspeed', [], 'c'),
        ('dipy.segment.clusteringspeed', [], 'c'),
        ('dipy.segment.clustering_algorithms', [], 'c'),
        ('dipy.segment.mrf', [], 'c'),
        ('dipy.denoise.denspeed', [], 'c'),
        ('dipy.denoise.pca_noise_estimate', [], 'c'),
        ('dipy.denoise.nlmeans_block', [], 'c'),
        ('dipy.denoise.enhancement_kernel', [], 'c'),
        ('dipy.denoise.shift_twist_convolution', [], 'c'),
        ('dipy.align.vector_fields', [], 'c'),
        ('dipy.align.sumsqdiff', [], 'c'),
        ('dipy.align.expectmax', [], 'c'),
        ('dipy.align.crosscorr', [], 'c'),
        ('dipy.align.bundlemin', [], 'c'),
        ('dipy.align.transforms', [], 'c'),
        ('dipy.align.parzenhist', [], 'c'),
        ('dipy.utils.omp', [], 'c'),
        ('dipy.utils.fast_numpy', [], 'c')):
    pyx_src = pjoin(*modulename.split('.')) + '.pyx'
    EXTS.append(Extension(modulename, [pyx_src] + other_sources,
                          language=language,
                          **deepcopy(ext_kwargs)))  # deepcopy lists

# Do our own build and install time dependency checking. setup.py gets called
# in many different ways, and may be called just to collect information
# (egg_info). We need to set up tripwires to raise errors when actually doing
# things, like building, rather than unconditionally in the setup.py import or
# exec We may make tripwire versions of build_ext, build_py, install
need_cython = True
pybuilder = get_comrec_build('dipy')
# Cython is a dependency for building extensions, iff we don't have stamped
# up pyx and c files.
build_ext, need_cython = cyproc_exts(EXTS,
                                     info.CYTHON_MIN_VERSION,
                                     'pyx-stamps')
# Add openmp flags if they work
simple_test_c = """int main(int argc, char** argv) { return(0); }"""
omp_test_c = """#include <omp.h>
int main(int argc, char** argv) { return(0); }"""

msc_flag_defines = [[['/openmp'], [], omp_test_c, 'HAVE_VC_OPENMP'],
                    ]
gcc_flag_defines = [[['-msse2', '-mfpmath=sse'], [], simple_test_c, 'USING_GCC_SSE2'],
                    ]

if 'clang' not in platform.python_compiler().lower():
    gcc_flag_defines += [[['-fopenmp'], ['-fopenmp'], omp_test_c, 'HAVE_OPENMP'], ]

# Test if it is a 32 bits version
if not sys.maxsize > 2 ** 32:
    # This flag is needed only on 32 bits
    msc_flag_defines += [[['/arch:SSE2'], [], simple_test_c, 'USING_VC_SSE2'], ]

flag_defines = msc_flag_defines if 'msc' in platform.python_compiler().lower() else gcc_flag_defines

extbuilder = add_flag_checking(build_ext, flag_defines, 'dipy')

# Use ext builder to add np.get_include() at build time, not during setup.py
# execution.
extbuilder = make_np_ext_builder(extbuilder)
if need_cython:
    SetupDependency('Cython', info.CYTHON_MIN_VERSION,
                    req_type='install_requires',
                    heavy=True).check_fill(extra_setuptools_args)
SetupDependency('numpy', info.NUMPY_MIN_VERSION,
                req_type='install_requires',
                heavy=True).check_fill(extra_setuptools_args)
SetupDependency('scipy', info.SCIPY_MIN_VERSION,
                req_type='install_requires',
                heavy=True).check_fill(extra_setuptools_args)
SetupDependency('nibabel', info.NIBABEL_MIN_VERSION,
                req_type='install_requires',
                heavy=False).check_fill(extra_setuptools_args)
SetupDependency('h5py', info.H5PY_MIN_VERSION,
                req_type='install_requires',
                heavy=False).check_fill(extra_setuptools_args)
SetupDependency('tqdm', info.TQDM_MIN_VERSION,
                req_type='install_requires',
                heavy=False).check_fill(extra_setuptools_args)

cmdclass = dict(
    build_py=pybuilder,
    build_ext=extbuilder,
    install_scripts=install_scripts_bat,
    sdist=get_pyx_sdist(include_dirs=['src']))


def main(**extra_args):
    setup(name=info.NAME,
          maintainer=info.MAINTAINER,
          maintainer_email=info.MAINTAINER_EMAIL,
          description=info.DESCRIPTION,
          long_description=info.LONG_DESCRIPTION,
          url=info.URL,
          download_url=info.DOWNLOAD_URL,
          license=info.LICENSE,
          classifiers=info.CLASSIFIERS,
          author=info.AUTHOR,
          author_email=info.AUTHOR_EMAIL,
          platforms=info.PLATFORMS,
          version=info.VERSION,
          requires=info.REQUIRES,
          provides=info.PROVIDES,
          packages=['dipy',
                    'dipy.tests',
                    'dipy.align',
                    'dipy.align.tests',
                    'dipy.core',
                    'dipy.core.tests',
                    'dipy.direction',
                    'dipy.direction.tests',
                    'dipy.tracking',
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
                    'dipy.data.tests',
                    'dipy.utils',
                    'dipy.utils.tests',
                    'dipy.segment',
                    'dipy.segment.benchmarks',
                    'dipy.segment.tests',
                    'dipy.sims',
                    'dipy.sims.tests',
                    'dipy.stats',
                    'dipy.stats.tests',
                    'dipy.denoise',
                    'dipy.denoise.tests',
                    'dipy.workflows',
                    'dipy.workflows.tests',
                    'dipy.nn',
                    'dipy.nn.tests'],

          ext_modules=EXTS,
          package_data={'dipy': [pjoin('data', 'files', '*')],
                        },
          data_files=[('share/doc/dipy/examples',
                       glob(pjoin('doc', 'examples', '*.py')))],
          scripts=glob(pjoin('bin', 'dipy_*')),
          cmdclass=cmdclass,
          **extra_args
          )


# simple way to test what setup will do
# python setup.py install --prefix=/tmp
if __name__ == "__main__":
    main(**extra_setuptools_args)
