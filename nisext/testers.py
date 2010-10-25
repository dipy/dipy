''' Test package information in various install settings

The routines here install the package in various settings and print out the
corresponding version info from the installation.

The typical use for this module is as a makefile target, as in::

    # Print out info for possible install methods
    check-version-info:
        python -c 'from nisext.testers import info_from_here; info_from_here("mypackage")'

'''

import os
from os.path import join as pjoin
from glob import glob
import shutil
import tempfile
import zipfile
from subprocess import call
from functools import partial

my_call = partial(call, shell=True)

PY_LIB_SDIR = 'pylib'

def run_mod_cmd(mod_name, pkg_path, cmd):
    ''' Run command in own process in anonymous path

    Parameters
    ----------
    mod_name : str
        Name of module to import - e.g. 'nibabel'
    pkg_path : str
        directory containing `mod_name` package.  Typically that will be the
        directory containing the e.g. 'nibabel' directory.
    '''
    cwd = os.getcwd()
    tmpdir = tempfile.mkdtemp()
    try:
        os.chdir(tmpdir)
        my_call('python -c "import sys; sys.path.insert(1,\'%s\'); '
                'import %s;'
                'print %s.__file__;'
                '%s"' % (pkg_path,
                         mod_name,
                         mod_name,
                         cmd))
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmpdir)


def zip_extract_all(fname, path=None):
    ''' Extract all members from zipfile

    Deals with situation where the directory is stored in the zipfile as a name,
    as well as files that have to go into this directory.
    '''
    zf = zipfile.ZipFile(fname)
    members = zf.namelist()
    # Remove members that are just bare directories
    members = [m for m in members if not m.endswith('/')]
    for zipinfo in members:
        zf.extract(zipinfo, path, None)


def install_from_to(from_dir, to_dir, py_lib_sdir):
    """ Install package in `from_dir` to standard location in `to_dir`

    Return path to directory containing package directory. The package directory
    is the directory containing __init__.py
    """
    site_pkgs_path = os.path.join(to_dir, py_lib_sdir)
    py_lib_locs = ' --install-purelib=%s --install-platlib=%s' % (
        site_pkgs_path, site_pkgs_path)
    pwd = os.path.abspath(os.getcwd())
    try:
        os.chdir(from_dir)
        my_call('python setup.py --quiet install --prefix=%s %s' % (to_dir,
                                                                    py_lib_locs))
    finally:
        os.chdir(pwd)
    return site_pkgs_path


def check_installed_files(repo_mod_path, install_mod_path):
    """ Check files in `repo_mod_path` are installed at `install_mod_path`

    At the moment, all this does is check that all the ``*.py`` files in
    `repo_mod_path` are installed at `install_mod_path`.

    Parameters
    ----------
    repo_mod_path : str
        repository path containing package files, e.g. <nibabel-repo>/nibabel>
    install_mod_path : str
        path at which package has been installed.  This is the path where the
        root package ``__init__.py`` lives.

    Return
    ------
    uninstalled : list
        list of files that should have been installed, but have not been
        installed
    """
    repo_mod_path = os.path.abspath(repo_mod_path)
    uninstalled = []
    # Walk directory tree to get py files
    for dirpath, dirnames, filenames in os.walk(repo_mod_path):
        out_dirpath = dirpath.replace(repo_mod_path, install_mod_path)
        for fname in filenames:
            if not fname.lower().endswith('.py'):
                continue
            equiv_fname = os.path.join(out_dirpath, fname)
            if not os.path.isfile(equiv_fname):
                uninstalled.append(pjoin(dirpath, fname))
    return uninstalled


def contexts_print_info(mod_name, repo_path, install_path):
    ''' Print result of get_info from different installation routes

    Runs installation from:

    * git archive zip file
    * with setup.py install from repository directory
    * just running code from repository directory

    and prints out result of get_info in each case.  There will be many files
    written into `install_path` that you may want to clean up somehow.

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
    # first test archive
    pwd = os.path.abspath(os.getcwd())
    out_fname = pjoin(install_path, 'test.zip')
    try:
        os.chdir(repo_path)
        my_call('git archive --format zip -o %s HEAD' % out_fname)
    finally:
        os.chdir(pwd)
    install_from = pjoin(install_path, mod_name)
    zip_extract_all(out_fname, install_from)
    site_pkgs_path = install_from_to(install_from,
                                     install_path,
                                     PY_LIB_SDIR)
    cmd_str = 'print %s.get_info()' % mod_name
    run_mod_cmd(mod_name, site_pkgs_path, cmd_str)
    # now test install into a directory from the repository
    site_pkgs_path = install_from_to(repo_path,
                                     install_path,
                                     PY_LIB_SDIR)
    run_mod_cmd(mod_name, site_pkgs_path, cmd_str)
    # Take the opportunity to audit the py files
    repo_mod_path = os.path.join(repo_path, mod_name)
    install_mod_path = os.path.join(site_pkgs_path, mod_name)
    print 'Files not taken across by the installation:'
    print check_installed_files(repo_mod_path, install_mod_path)
    # test from development tree
    try:
        os.chdir(repo_path)
        my_call('python setup.py build_ext -i')
    finally:
        os.chdir(pwd)
    run_mod_cmd(mod_name, repo_path, cmd_str)
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


def tests_installed(mod_name, source_path=None):
    """ Install from `source_path` into temporary directory; run tests

    Parameters
    ----------
    mod_name : str
        name of module - e.g. 'nibabel'
    source_path : None or str
        Path from which to install.  If None, defaults to working directory
    """
    if source_path is None:
        source_path = os.path.abspath(os.getcwd())
    install_path = tempfile.mkdtemp()
    try:
        site_pkgs_path = install_from_to(source_path,
                                         install_path,
                                         PY_LIB_SDIR)
        run_mod_cmd(mod_name, site_pkgs_path, mod_name + '.test()')
    finally:
        shutil.rmtree(install_path)


def tests_from_zip(mod_name, zip_fname):
    """ Runs test from sdist zip source archive """
    install_path = tempfile.mkdtemp()
    try:
        zip_extract_all(zip_fname, install_path)
        pkg_dirs = glob(pjoin(install_path, mod_name + "*"))
        if len(pkg_dirs) != 1:
            raise OSError('There must be one and only one package dir')
        pkg_contents = pjoin(install_path, pkg_dirs[0])
        tests_installed(mod_name, pkg_contents)
    finally:
        shutil.rmtree(install_path)


def sdist_tests(mod_name, repo_path=None):
    """ Make sdist zip, install from it, and run tests """
    pwd = os.path.abspath(os.getcwd())
    if repo_path is None:
        repo_path = pwd
    install_path = tempfile.mkdtemp()
    try:
        os.chdir(repo_path)
        my_call('python setup.py sdist --formats=zip --dist-dir='
                + install_path)
        zip_fnames = glob(pjoin(install_path, mod_name + "*.zip"))
        if len(zip_fnames) != 1:
            raise OSError('There must be one and only one zip file, '
                          'but I found "%s"' % ': '.join(zip_fnames))
        tests_from_zip(mod_name, zip_fnames[0])
    finally:
        os.chdir(pwd)
        shutil.rmtree(install_path)

