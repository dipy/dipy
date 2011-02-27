#!/usr/bin/env python
''' Installation script for dipy package '''

import os
import sys
from os.path import join as pjoin, splitext, dirname
from subprocess import check_call
from glob import glob

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

import numpy as np

# Get version and release info, which is all stored in dipy/info.py
ver_file = pjoin('dipy', 'info.py')
execfile(ver_file)

# force_setuptools can be set from the setup_egg.py script
if not 'force_setuptools' in globals():
    # For some commands, use setuptools
    if len(set(('develop', 'bdist_egg', 'bdist_rpm', 'bdist', 'bdist_dumb',
                'bdist_wininst', 'install_egg_info', 'egg_info',
                'easy_install')).intersection(sys.argv)) > 0:
        force_setuptools = True
    else:
        force_setuptools = False

if force_setuptools:
    # Try to preempt setuptools monkeypatching of Extension handling when Pyrex
    # is missing.  Otherwise the monkeypatched Extension will change .pyx
    # filenames to .c filenames, and we probably don't have the .c files.
    sys.path.insert(0, pjoin(dirname(__file__), 'fake_pyrex'))
    import setuptools

# We may just have imported setuptools, or we may have been exec'd from a
# setuptools environment like pip
if 'setuptools' in sys.modules:
    extra_setuptools_args = dict(
        tests_require=['nose'],
        test_suite='nose.collector',
        zip_safe=False,
        extras_require = dict(
            doc=['Sphinx>=1.0'],
            test=['nose>=0.10.1']),
        install_requires = ['nibabel>=' + NIBABEL_MIN_VERSION])
    # I removed numpy and scipy from install requires because easy_install seems
    # to want to fetch these if they are already installed, meaning of course
    # that there's a long fragile and unnecessary compile before the install
    # finishes.
    # We need setuptools install command because we're going to override it
    # further down.  Using distutils install command causes some confusion, due
    # to the Pyrex fix above.
    from setuptools.command import install
else:
    extra_setuptools_args = {}
    from distutils.command import install

# Import distutils _after_ potential setuptools import above, and after removing
# MANIFEST
from distutils.core import setup
from distutils.extension import Extension
from distutils.version import LooseVersion
from distutils.command import build_py, build_ext, sdist

# Do our own build and install time dependency checking. setup.py gets called in
# many different ways, and may be called just to collect information (egg_info).
# We need to set up tripwires to raise errors when actually doing things, like
# building, rather than unconditionally in the setup.py import or exec
def derror_maker(klass, msg):
    """ Decorate distutils class to make run method raise error """
    class K(klass):
        def run(self):
            raise RuntimeError(msg)
    return K

# We may make tripwire versions of these commands
try:
    from nisext.sexts import package_check, get_comrec_build
except ImportError: # No nibabel
    msg = ('Need nisext package from nibabel installation'
           ' - please install nibabel first')
    pybuilder = derror_maker(build_py.build_py, msg)
    extbuilder = derror_maker(build_ext.build_ext, msg)
    installer = derror_maker(install.install, msg)
else: # We have nibabel
    pybuilder = get_comrec_build('dipy')
    # Cython is a dependency for building extensions
    try:
        from Cython.Compiler.Version import version as cyversion
    except ImportError:
        cython_ok = False
    else:
        cython_ok = LooseVersion(cyversion) >= CYTHON_MIN_VERSION
    if not cython_ok:
        extbuilder = derror_maker(build_ext.build_ext,
                                  'Need cython>=%s to build extensions'
                                  % CYTHON_MIN_VERSION)
    else: # We have a good-enough cython
        from Cython.Distutils import build_ext as extbuilder
    class installer(install.install):
        def run(self):
            package_check('numpy', NUMPY_MIN_VERSION)
            package_check('scipy', SCIPY_MIN_VERSION)
            package_check('nibabel', NIBABEL_MIN_VERSION)
            install.install.run(self)

cmdclass = dict(
    build_py=pybuilder,
    build_ext=extbuilder,
    install=installer)

EXTS = []
for modulename, other_sources in (
    ('dipy.reconst.recspeed', []),
    ('dipy.tracking.distances', []),
    ('dipy.tracking.vox2track', []),
    ('dipy.tracking.propspeed', [])):
    pyx_src = pjoin(*modulename.split('.')) + '.pyx'
    EXTS.append(Extension(modulename,[pyx_src] + other_sources,
                          include_dirs = [np.get_include()]))

# Custom sdist command to generate .c files from pyx files.  We need the .c
# files because pip will not allow us to preempt setuptools from checking for
# Pyrex. When setuptools doesn't find Pyrex, it changes .pyx filenames in the
# extension sources into .c filenames.  By putting the .c files into the source
# archive, at least pip does not crash in a confusing way.
class SDist(sdist.sdist):
    def make_distribution(self):
        """ Compile up C files and add to sources """
        for mod in EXTS:
            for source in mod.sources:
                base, ext = splitext(source)
                if not ext in ('.pyx', '.py'):
                    continue
                c_file = base + '.c'
                check_call('cython ' + source, shell=True)
                self.filelist.append(c_file)
        sdist.sdist.make_distribution(self)

cmdclass['sdist'] = SDist


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
                          'dipy.utils.tests',
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
          data_files=[('share/doc/dipy/examples',
                       glob(pjoin('doc','examples','*.py')))],
          scripts      = glob(pjoin('scripts', '*')),
          cmdclass = cmdclass,
          **extra_args
         )

#simple way to test what setup will do
#python setup.py install --prefix=/tmp
if __name__ == "__main__":
    main(**extra_setuptools_args)
