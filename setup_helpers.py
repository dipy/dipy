''' Distutils / setuptools helpers

'''
import os
from os.path import join as pjoin, split as psplit, splitext
import tempfile
import shutil

from distutils.command.install_scripts import install_scripts
from distutils.errors import CompileError, LinkError

from distutils import log

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
            if not (first_line.startswith('#!') and
                    'python' in first_line.lower()):
                log.info("No #!python executable found, skipping .bat "
                            "wrapper")
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

TEST_C = """
int main(int argc, char** argv) { return(0); }
"""

def add_flag_checking(build_ext_class, input_flags):
    """ Override input `build_ext_class` to check compiler `input_flags`

    Parameters
    ----------
    build_ext_class : class
        Class implementing ``distutils.command.build_ext.build_ext`` interface,
        with a ``build_extensions`` method.
    input_flags : sequence
        A sequence of compiler flags.  We check each to see whether a simple C
        source file will compile, and omit flags that cause a compile error

    Returns
    -------
    checker_class : class
        A class with similar interface to
        ``distutils.command.build_ext.build_ext``, that adds all working
        `input_flag` values to the ``extra_compile_args`` and
        ``extra_link_args`` attributes of extensions, before compiling.
    """
    class Checker(build_ext_class):
        flags = tuple(input_flags)

        def can_compile_link(self, flags):
            cc = self.compiler
            fname = 'test.c'
            cwd = os.getcwd()
            tmpdir = tempfile.mkdtemp()
            try:
                os.chdir(tmpdir)
                with open(fname, 'wt') as fobj:
                    fobj.write(TEST_C)
                try:
                    objects = cc.compile([fname], extra_postargs=flags)
                except CompileError:
                    return False
                try:
                    cc.link_executable(objects, "a.out", extra_postargs=flags)
                except (LinkError, TypeError):
                    return False
            finally:
                os.chdir(cwd)
                shutil.rmtree(tmpdir)
            return True

        def build_extensions(self):
            """ Hook into extension building to check compiler flags """
            for flag in self.flags:
                if not self.can_compile_link([flag]):
                    log.warn("Flag {0} omitted because of compile or link "
                             "error".format(flag))
                    continue
                for ext in self.extensions:
                    ext.extra_compile_args.append(flag)
                    ext.extra_link_args.append(flag)
            build_ext_class.build_extensions(self)

    return Checker
