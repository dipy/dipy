''' Distutils / setuptools helpers '''

from os.path import join as pjoin
from ConfigParser import ConfigParser

from distutils.version import LooseVersion
from distutils.command.build_py import build_py

from distutils import log

def get_comrec_build(pkg_dir, build_cmd=build_py):
    """ Return extended build command class for recording commit

    The extended command tries to run git to find the current commit, getting
    the empty string if it fails.  It then writes the commit hash into a file
    in the `pkg_dir` path, named ``COMMIT_INFO.txt``.

    In due course this information can be used by the package after it is
    installed, to tell you what commit it was installed from if known.

    To make use of this system, you need a package with a COMMIT_INFO.txt file -
    e.g. ``myproject/COMMIT_INFO.txt`` - that might well look like this::

        # This is an ini file that may contain information about the code state
        [commit hash]
        # The line below may contain a valid hash if it has been substituted during 'git archive'
        archive_subst_hash=$Format:%h$
        # This line may be modified by the install process
        install_hash=

    The COMMIT_INFO file above is also designed to be used with git substitution
    - so you probably also want a ``.gitattributes`` file in the root directory
    of your working tree that contains something like this::

       myproject/COMMIT_INFO.txt export-subst

    That will cause the ``COMMIT_INFO.txt`` file to get filled in by ``git
    archive`` - useful in case someone makes such an archive - for example with
    via the github 'download source' button.

    Although all the above will work as is, you might consider having something
    like a ``get_info()`` function in your package to display the commit
    information at the terminal.  See the ``pkg_info.py`` module in the nipy
    package for an example.
    """
    class MyBuildPy(build_cmd):
        ''' Subclass to write commit data into installation tree '''
        def run(self):
            build_py.run(self)
            import subprocess
            proc = subprocess.Popen('git rev-parse --short HEAD',
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    shell=True)
            repo_commit, _ = proc.communicate()
            # We write the installation commit even if it's empty
            cfg_parser = ConfigParser()
            cfg_parser.read(pjoin(pkg_dir, 'COMMIT_INFO.txt'))
            cfg_parser.set('commit hash', 'install_hash', repo_commit)
            out_pth = pjoin(self.build_lib, pkg_dir, 'COMMIT_INFO.txt')
            cfg_parser.write(open(out_pth, 'wt'))
    return MyBuildPy


# Dependency checks
def package_check(pkg_name, version=None,
                  optional=False,
                  checker=LooseVersion,
                  version_getter=None,
                  messages=None
                  ):
    ''' Check if package `pkg_name` is present, and correct version

    Parameters
    ----------
    pkg_name : str
       name of package as imported into python
    version : {None, str}, optional
       minimum version of the package that we require. If None, we don't
       check the version.  Default is None
    optional : {False, True}, optional
       If False, raise error for absent package or wrong version;
       otherwise warn
    checker : callable, optional
       callable with which to return comparable thing from version
       string.  Default is ``distutils.version.LooseVersion``
    version_getter : {None, callable}:
       Callable that takes `pkg_name` as argument, and returns the
       package version string - as in::

          ``version = version_getter(pkg_name)``

       If None, equivalent to::

          mod = __import__(pkg_name); version = mod.__version__``
    messages : None or dict, optional
       dictionary giving output messages
    '''
    if version_getter is None:
        def version_getter(pkg_name):
            mod = __import__(pkg_name)
            return mod.__version__
    if messages is None:
        messages = {}
    msgs = {
         'missing': 'Cannot import package "%s" - is it installed?',
         'missing opt': 'Missing optional package "%s"',
         'opt suffix' : '; you may get run-time errors',
         'version too old': 'You have version %s of package "%s"'
         ' but we need version >= %s', }
    msgs.update(messages)
    try:
        __import__(pkg_name)
    except ImportError:
        if not optional:
            raise RuntimeError(msgs['missing'] % pkg_name)
        log.warn(msgs['missing opt'] % pkg_name +
                 msgs['opt suffix'])
        return
    if not version:
        return
    try:
        have_version = version_getter(pkg_name)
    except AttributeError:
        raise RuntimeError('Cannot find version for %s' % pkg_name)
    if checker(have_version) < checker(version):
        if optional:
            log.warn(msgs['version too old'] + msgs['opt suffix'],
                     have_version,
                     pkg_name,
                     version)
        else:
            raise RuntimeError(msgs['version too old'],
                               have_version,
                               pkg_name,
                               version)
