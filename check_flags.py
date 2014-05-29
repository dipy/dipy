from __future__ import print_function

from io import BytesIO
import sys
from os.path import join as pjoin
import shutil
from tempfile import mkdtemp
import uuid

from distutils import core
from distutils.errors import CCompilerError, DistutilsArgError

TEST_C = """
int func() { return(1); }
"""

def check_flags(compile_args,
                script_name = 'setup.py',
                script_args=('build_ext',),
                code=TEST_C):
    """ Check if flags in `compile_args` will compile extension code `code`

    Parameters
    ----------
    compile_args : sequence
        extra compile args to pass to extension
    script_name : str
        Name of setup script (for error reporting)
    script_args : sequence, optional
        Command line input arguments (same arguments as to
        ``distutils.core.setup()``).  By default just ``('build_ext',)``
    code : str, optional
        Code to compile, By default a tiny function
    """
    fbase = '{0}.c'.format(uuid.uuid4())
    stdout = sys.stdout
    stderr = sys.stderr
    tmpdir = mkdtemp()
    try:
        sys.stdout = BytesIO()
        sys.stderr = BytesIO()
        fname = pjoin(tmpdir, fbase)
        with open(fname, 'wt') as fobj:
            fobj.write(TEST_C)
        mod = core.Extension(fbase, sources=[fname],
                             extra_compile_args=compile_args)
        try: # check with the current script args
            dist = _make_dist(mod, script_name, script_args)
        except DistutilsArgError:
            dist = None # No commands to run
        if dist is None or not 'build_ext' in dist.commands:
            return None
        try:
            dist.run_command('build_ext')
        except CCompilerError:
             return False
    finally:
        shutil.rmtree(tmpdir)
        sys.stdout = stdout
        sys.stderr = stderr
    return True


def _make_dist(mod, script_name, script_args):
    dist = core.Distribution(dict(
        name = 'test',
        script_name = script_name,
        ext_modules = [mod],
        script_args = script_args))
    dist.parse_config_files()
    dist.parse_command_line()
    return dist


def main():
    print(check_flags(['-fopenmp'], sys.argv[1:]))


if __name__ == '__main__':
    main()
