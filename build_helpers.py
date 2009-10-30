''' Module to automate cython building '''

import os
from os.path import join as pjoin

from distutils.extension import Extension


def make_cython_ext(modulename,
                    has_cython,
                    include_dirs=None,
                    extra_c_sources=None):
    ''' Create Cython extension builder from module names

    Returns extension for building and command class depending on
    whether you want to use Cython and ``.pyx`` files for building
    (`has_cython` == True) or the Cython-generated C files (`has_cython`
    == False).

    Assumes ``pyx`` or C file has the same path as that implied by
    modulename. 

    Parameters
    ----------
    modulename : string
       module name, relative to setup.py path, with python dot
       separators, e.g mypkg.mysubpkg.mymodule
    has_cython : bool
       True if we have cython, False otherwise
    include_dirs : None or sequence
       include directories
    extra_c_sources : None or sequence
       sequence of strings giving extra C source files

    Returns
    -------
    ext : extension object
    cmdclass : dict
       command class dictionary for setup.py

    Examples
    --------
    You will need Cython on your python path to run these tests. 
    
    >>> modulename = 'pkg.subpkg.mymodule'
    >>> ext, cmdclass = make_cython_ext(modulename, True, None,['test.c'])
    >>> ext.name == modulename
    True
    >>> pyx_src = os.path.join('pkg', 'subpkg', 'mymodule.pyx')
    >>> ext.sources == [pyx_src, 'test.c']
    True
    >>> import Cython.Distutils
    >>> cmdclass['build_ext'] == Cython.Distutils.build_ext
    True
    >>> ext, cmdclass = make_cython_ext(modulename, False, None, ['test.c'])
    >>> ext.name == modulename
    True
    >>> pyx_src = os.path.join('pkg', 'subpkg', 'mymodule.c')
    >>> ext.sources == [pyx_src, 'test.c']
    True
    >>> cmdclass
    {}
    '''
    if include_dirs is None:
        include_dirs = []
    if extra_c_sources is None:
        extra_c_sources = []
    if has_cython:
        src_ext = '.pyx'
    else:
        src_ext = '.c'
    pyx_src = pjoin(*modulename.split('.')) + src_ext
    sources = [pyx_src] + extra_c_sources
    ext = Extension(modulename, sources, include_dirs = include_dirs)
    if has_cython:
        from Cython.Distutils import build_ext
        cmdclass = {'build_ext': build_ext}
    else:
        cmdclass = {}
    return ext, cmdclass
        

