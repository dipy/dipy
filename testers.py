''' Test package information in various install settings

The routines here install the package in various settings and print out the
corresponding version info from the installation.

The typical use for this module is as a makefile target, as in::

    # Print out info for possible install methods
    check-version-info:
        $(PYTHON) -c 'from nisext.testers import info_from_here; info_from_here("mypackage")'
    
'''

import os
from os.path import join as pjoin
import shutil
import tempfile
import zipfile
from subprocess import call
from functools import partial

my_call = partial(call, shell=True)

PY_LIB_SDIR = 'pylib'

def sys_print_info(mod_name, pkg_path):
    ''' Run info print in own process in anonymous path
    '''
    cwd = os.getcwd()
    tmpdir = tempfile.mkdtemp()
    try:
        os.chdir(tmpdir)
        my_call('python -c "import sys; sys.path.insert(1,\'%s\'); '
                'import %s;'
                'print %s.__file__;'
                'print %s.get_info()"' % (pkg_path,
                                          mod_name,
                                          mod_name,
                                          mod_name))
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmpdir)


def zip_extract_all(fname):
    ''' Extract all members from zipfile

    Deals with situation where the directory is stored in the zipfile as a name,
    as well as files that have to go into this directory.
    '''
    zf = zipfile.ZipFile(fname)
    members = zf.namelist()
    # Remove members that are just bare directories
    members = [m for m in members if not m.endswith('/')]
    zf.extractall(members = members)


def contexts_print_info(mod_name, repo_path, install_path):
    ''' Print result of get_info from different installation routes

    Runs installation from:

    * git archive zip file
    * with setup.py install from repository directory
    * just running code from repository directory

    and prints out result of get_info in each case

    Parameters
    ----------
    mod_name : str
       package name that will be installed, and tested
    repo_path : str
       path to location of git repository
    install_path : str
       path into which to install temporary installations
    '''
    site_pkgs_path = os.path.join(install_path, PY_LIB_SDIR)
    py_lib_locs = ' --install-purelib=%s --install-platlib=%s' % (
        site_pkgs_path, site_pkgs_path)
    # first test archive
    os.chdir(repo_path)
    out_fname = pjoin(install_path, 'test.zip')
    my_call('git archive --format zip -o %s master' % out_fname)
    os.chdir(install_path)
    zip_extract_all('test.zip')
    my_call('python setup.py --quiet install --prefix=%s %s' % (install_path,
                                                                py_lib_locs))
    sys_print_info(mod_name, site_pkgs_path)
    # remove installation
    shutil.rmtree(site_pkgs_path)
    # now test install into a directory from the repository
    os.chdir(repo_path)
    my_call('python setup.py --quiet install --prefix=%s %s' % (install_path,
                                                                py_lib_locs))
    sys_print_info(mod_name, site_pkgs_path)
    # test from development tree
    sys_print_info(mod_name, repo_path)
    return


def info_from_here(mod_name):
    ''' Run info context checks starting in working directory

    Runs checks from current working directory, installing temporary
    installations into a new temporary directory

    Parameters
    ----------
    mod_name : str
       package name that will be installed, and tested
    '''
    repo_path = os.path.abspath(os.getcwd())
    install_path = tempfile.mkdtemp()
    try:
        contexts_print_info(mod_name, repo_path, install_path)
    finally:
        shutil.rmtree(install_path)
