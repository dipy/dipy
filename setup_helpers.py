""" Distutils / setuptools helpers

"""
import os
import sys
from os.path import join as pjoin, split as psplit, splitext, dirname, exists
import tempfile
import shutil
import logging as log

from setuptools.command.install_scripts import install_scripts
try:
    from setuptools.errors import CompileError, LinkError
except ImportError:
    # can remove this except case once we require setuptools>=59.0
    from distutils.errors import CompileError, LinkError
from packaging.version import Version


BAT_TEMPLATE = \
    r"""@echo off
REM wrapper to use shebang first line of {FNAME}
set mypath=%~dp0
set pyscript="%mypath%{FNAME}"
set /p line1=<%pyscript%
if "%line1:~0,2%" == "#!" (goto :goodstart)
echo First line of %pyscript% does not start with "#!"
exit /b 1
:goodstart
set py_exe=%line1:~2%
REM quote exe in case of spaces in path name
set py_exe="%py_exe%"
call %py_exe% %pyscript% %*
"""

# Path of file to which to write C conditional vars from build-time checks
CONFIG_H = pjoin('build', 'config.h')
# File name (no directory) to which to write Python vars from build-time checks
CONFIG_PY = '__config__.py'
# Directory to which to write libraries for building
LIB_DIR_TMP = pjoin('build', 'extra_libs')


class install_scripts_bat(install_scripts):
    """ Make scripts executable on Windows

    Scripts are bare file names without extension on Unix, fitting (for example)
    Debian rules. They identify as python scripts with the usual ``#!`` first
    line. Unix recognizes and uses this first "shebang" line, but Windows does
    not. So, on Windows only we add a ``.bat`` wrapper of name
    ``bare_script_name.bat`` to call ``bare_script_name`` using the python
    interpreter from the #! first line of the script.

    Notes
    -----
    See discussion at
    http://matthew-brett.github.com/pydagogue/installing_scripts.html and
    example at git://github.com/matthew-brett/myscripter.git for more
    background.
    """

    def run(self):
        install_scripts.run(self)
        if not os.name == "nt":
            return
        for filepath in self.get_outputs():
            # If we can find an executable name in the #! top line of the script
            # file, make .bat wrapper for script.
            with open(filepath, 'rt') as fobj:
                first_line = fobj.readline()
            if not (first_line.startswith('#!') and 'python' in first_line.lower()):
                log.info("No #!python executable found, skipping .bat wrapper")
                continue
            pth, fname = psplit(filepath)
            froot, ext = splitext(fname)
            bat_file = pjoin(pth, froot + '.bat')
            bat_contents = BAT_TEMPLATE.replace('{FNAME}', fname)
            log.info("Making %s wrapper for %s" % (bat_file, filepath))
            if self.dry_run:
                continue
            with open(bat_file, 'wt') as fobj:
                fobj.write(bat_contents)


def add_flag_checking(build_ext_class, flag_defines, top_package_dir=''):
    """ Override input `build_ext_class` to check compiler `flag_defines`

    Parameters
    ----------
    build_ext_class : class
        Class implementing ``setuptools.command.build_ext.build_ext`` interface,
        with a ``build_extensions`` method.
    flag_defines : sequence
        A sequence of elements, where the elements are sequences of length 4
        consisting of (``compile_flags``, ``link_flags``, ``code``,
        ``defvar``). ``compile_flags`` is a sequence of compiler flags;
        ``link_flags`` is a sequence of linker flags. We
        check ``compile_flags`` to see whether a C source string ``code`` will
        compile, and ``link_flags`` to see whether the resulting object file
        will link.  If both compile and link works, we add ``compile_flags`` to
        ``extra_compile_args`` and ``link_flags`` to ``extra_link_args`` of
        each extension when we build the extensions.  If ``defvar`` is not
        None, it is the name of C variable to be defined in ``build/config.h``
        with 1 if the combination of (``compile_flags``, ``link_flags``,
        ``code``) will compile and link, 0 otherwise. If None, do not write
        variable.
    top_package_dir : str
        String giving name of top-level package, for writing Python file
        containing configuration variables.  If empty, do not write this file.
        Variables written are the same as the Cython variables generated via
        the `flag_defines` setting.

    Returns
    -------
    checker_class : class
        A class with similar interface to
        ``setuptools.command.build_ext.build_ext``, that adds all working
        ``compile_flags`` values to the ``extra_compile_args`` and working
        ``link_flags`` to ``extra_link_args`` attributes of extensions, before
        compiling.
    """

    class Checker(build_ext_class):
        flag_defs = tuple(flag_defines)

        def can_compile_link(self, compile_flags, link_flags, code):
            cc = self.compiler
            fname = 'test.c'
            cwd = os.getcwd()
            tmpdir = tempfile.mkdtemp()
            try:
                os.chdir(tmpdir)
                with open(fname, 'wt') as fobj:
                    fobj.write(code)
                try:
                    objects = cc.compile([fname],
                                         extra_postargs=compile_flags)
                except CompileError:
                    return False
                try:
                    # Link shared lib rather then executable to avoid
                    # http://bugs.python.org/issue4431 with MSVC 10+
                    cc.link_shared_lib(objects, "testlib",
                                       extra_postargs=link_flags)
                except (LinkError, TypeError):
                    return False
            finally:
                os.chdir(cwd)
                shutil.rmtree(tmpdir)
            return True

        def build_extensions(self):
            """ Hook into extension building to check compiler flags """
            def_vars = []
            good_compile_flags = []
            good_link_flags = []
            config_dir = dirname(CONFIG_H)
            for compile_flags, link_flags, code, def_var in self.flag_defs:
                compile_flags = list(compile_flags)
                link_flags = list(link_flags)
                flags_good = self.can_compile_link(compile_flags,
                                                   link_flags,
                                                   code)
                if def_var:
                    def_vars.append((def_var, flags_good))
                if flags_good:
                    good_compile_flags += compile_flags
                    good_link_flags += link_flags
                else:
                    log.warn("Flags {0} omitted because of compile or link "
                             "error".format(compile_flags + link_flags))
            if def_vars:  # write config.h file
                if not exists(config_dir):
                    self.mkpath(config_dir)
                with open(CONFIG_H, 'wt') as fobj:
                    fobj.write('/* Automatically generated; do not edit\n')
                    fobj.write('   C defines from build-time checks */\n')
                    for v_name, v_value in def_vars:
                        fobj.write('int {0} = {1};\n'.format(
                            v_name, 1 if v_value else 0))
            if def_vars and top_package_dir:  # write __config__.py file
                config_py_dir = (top_package_dir if self.inplace else
                                 pjoin(self.build_lib, top_package_dir))
                if not exists(config_py_dir):
                    self.mkpath(config_py_dir)
                config_py = pjoin(config_py_dir, CONFIG_PY)
                with open(config_py, 'wt') as fobj:
                    fobj.write('# Automatically generated; do not edit\n')
                    fobj.write('# Variables from compile checks\n')
                    for v_name, v_value in def_vars:
                        fobj.write('{0} = {1}\n'.format(v_name, v_value))
            if def_vars or good_compile_flags or good_link_flags:
                for ext in self.extensions:
                    ext.extra_compile_args += good_compile_flags
                    ext.extra_link_args += good_link_flags
                    if def_vars:
                        ext.include_dirs.append(config_dir)
            self.cython_directives = {
                'language_level': '3',
            }
            build_ext_class.build_extensions(self)

    return Checker


def get_pkg_version(pkg_name):
    """ Return package version for `pkg_name` if installed

    Returns
    -------
    pkg_version : str or None
        Return None if package not importable.  Return 'unknown' if standard
        ``__version__`` string not present. Otherwise return version string.
    """
    try:
        pkg = __import__(pkg_name)
    except ImportError:
        return None
    try:
        return pkg.__version__
    except AttributeError:
        return 'unknown'


def version_error_msg(pkg_name, found_ver, min_ver):
    """ Return informative error message for version or None
    """
    if found_ver is None:
        return 'We need package {0}, but not importable'.format(pkg_name)
    if found_ver == 'unknown':
        return 'We need {0} version {1}, but cannot get version'.format(
            pkg_name, min_ver)
    if Version(found_ver) >= Version(min_ver):
        return None
    return 'We need {0} version {1}, but found version {2}'.format(pkg_name, min_ver, found_ver)


class SetupDependency(object):
    """ SetupDependency class

    Parameters
    ----------
    import_name : str
        Name with which required package should be ``import``ed.
    min_ver : str
        Version string giving minimum version for package.
    req_type : {'install_requires', 'setup_requires'}, optional
        Setuptools dependency type.
    heavy : {False, True}, optional
        If True, and package is already installed (importable), then do not add
        to the setuptools dependency lists.  This prevents setuptools
        reinstalling big packages when the package was installed without using
        setuptools, or this is an upgrade, and we want to avoid the pip default
        behavior of upgrading all dependencies.
    install_name : str, optional
        Name identifying package to install from pypi etc, if different from
        `import_name`.
    """

    def __init__(self, import_name,
                 min_ver,
                 req_type='install_requires',
                 heavy=False,
                 install_name=None):
        self.import_name = import_name
        self.min_ver = min_ver
        self.req_type = req_type
        self.heavy = heavy
        self.install_name = (import_name if install_name is None
                             else install_name)

    def check_fill(self, setuptools_kwargs):
        """ Process this dependency, maybe filling `setuptools_kwargs`

        Run checks on this dependency.  If not using setuptools, then raise
        error for unmet dependencies.  If using setuptools, add missing or
        not-heavy dependencies to `setuptools_kwargs`.

        A heavy dependency is one that is inconvenient to install
        automatically, such as numpy or (particularly) scipy, matplotlib.

        Parameters
        ----------
        setuptools_kwargs : dict
            Dictionary of setuptools keyword arguments that may be modified
            in-place while checking dependencies.
        """
        found_ver = get_pkg_version(self.import_name)
        ver_err_msg = version_error_msg(self.import_name,
                                        found_ver,
                                        self.min_ver)
        if 'setuptools' not in sys.modules:
            # Not using setuptools; raise error for any unmet dependencies
            if ver_err_msg is not None:
                raise RuntimeError(ver_err_msg)
            return
        # Using setuptools; add packages to given section of
        # setup/install_requires, unless it's a heavy dependency for which we
        # already have an acceptable importable version.
        if self.heavy and ver_err_msg is None:
            return
        new_req = '{0}>={1}'.format(self.import_name, self.min_ver)
        old_reqs = setuptools_kwargs.get(self.req_type, [])
        setuptools_kwargs[self.req_type] = old_reqs + [new_req]


class Bunch(object):
    def __init__(self, vars):
        for key, name in vars.items():
            if key.startswith('__'):
                continue
            self.__dict__[key] = name


def read_vars_from(ver_file):
    """ Read variables from Python text file

    Parameters
    ----------
    ver_file : str
        Filename of file to read

    Returns
    -------
    info_vars : Bunch instance
        Bunch object where variables read from `ver_file` appear as
        attributes
    """
    # Use exec for compabibility with Python 3
    ns = {}
    with open(ver_file, 'rt') as fobj:
        exec(fobj.read(), ns)
    return Bunch(ns)


def make_np_ext_builder(build_ext_class):
    """ Override input `build_ext_class` to add numpy includes to extension

    This is useful to delay call of ``np.get_include`` until the extension is
    being built.

    Parameters
    ----------
    build_ext_class : class
        Class implementing ``setuptools.command.build_ext.build_ext`` interface,
        with a ``build_extensions`` method.

    Returns
    -------
    np_build_ext_class : class
        A class with similar interface to
        ``setuptools.command.build_ext.build_ext``, that adds libraries in
        ``np.get_include()`` to include directories of extension.
    """

    class NpExtBuilder(build_ext_class):
        def build_extensions(self):
            """ Hook into extension building to add np include dirs
            """
            # Delay numpy import until last moment
            import numpy as np
            for ext in self.extensions:
                ext.include_dirs.append(np.get_include())
            build_ext_class.build_extensions(self)

    return NpExtBuilder
