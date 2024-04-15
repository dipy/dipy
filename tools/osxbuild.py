"""Python script to build the macOS universal binaries.

Stolen with thankfulness from the numpy distribution

This is a simple script, most of the heavy lifting is done in bdist_mpkg.

To run this script:  'python build.py'

Installer is built using sudo so file permissions are correct when installed on
user system.  Script will prompt for sudo pwd.

"""

from getpass import getuser
from optparse import OptionParser
import os
import shutil
import subprocess
import sys

# USER_README = 'docs/README.rst'
# DEV_README = SRC_DIR + 'README.rst'

BUILD_DIR = 'build'
DIST_DIR = 'dist'
DIST_DMG_DIR = 'dist-dmg'


def remove_dirs(sudo):
    print('Removing old build and distribution directories...')
    print("""The distribution is built as root, so the files have the correct
    permissions when installed by the user.  Chown them to user for removal.""")
    if os.path.exists(BUILD_DIR):
        cmd = f'chown -R {getuser()} {BUILD_DIR}'
        if sudo:
            cmd = f'sudo {cmd}'
        shellcmd(cmd)
        shutil.rmtree(BUILD_DIR)
    if os.path.exists(DIST_DIR):
        cmd = f'sudo chown -R {getuser()} {DIST_DIR}'
        if sudo:
            cmd = f'sudo {cmd}'
        shellcmd(cmd)
        shutil.rmtree(DIST_DIR)


def build_dist(readme, python_exe, sudo):
    print('Building distribution... (using sudo)')
    cmd = f'{python_exe} setup_egg.py bdist_mpkg --readme={readme}'
    if sudo:
        cmd = f'sudo {cmd}'
    shellcmd(cmd)


def build_dmg(sudo):
    print('Building disk image...')
    # Since we removed the dist directory at the start of the script,
    # our pkg should be the only file there.
    pkg = os.listdir(DIST_DIR)[0]
    fn, ext = os.path.splitext(pkg)
    dmg = fn + '.dmg'
    srcfolder = os.path.join(DIST_DIR, pkg)
    dstfolder = os.path.join(DIST_DMG_DIR, dmg)
    # build disk image
    try:
        os.mkdir(DIST_DMG_DIR)
    except OSError:
        pass
    try:
        os.unlink(dstfolder)
    except OSError:
        pass
    cmd = f'hdiutil create -srcfolder {srcfolder} {dstfolder}'
    if sudo:
        cmd = f'sudo {cmd}'
    shellcmd(cmd)


def copy_readme():
    """Copy a user README with info regarding the website, instead of
    the developer README which tells one how to build the source.
    """
    print('Copy user README.rst for installer.')
    shutil.copy(USER_README, DEV_README)


def revert_readme():
    """Revert the developer README."""
    print('Reverting README.rst...')
    cmd = f'svn revert {DEV_README}'
    shellcmd(cmd)


def shellcmd(cmd, verbose=True):
    """Call a shell command."""
    if verbose:
        print(cmd)
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as err:
        msg = f"""
        Error while executing a shell command.
        {err}
        """
        raise Exception(msg)


def build():
    parser = OptionParser()
    parser.add_option("-p", "--python", dest="python",
                      default=sys.executable,
                      help="python interpreter executable",
                      metavar="PYTHON_EXE")
    parser.add_option("-r", "--readme", dest="readme",
                      default='README.rst',
                      help="README file",
                      metavar="README")
    parser.add_option("-s", "--sudo", dest="sudo",
                      default=False,
                      help="Run as sudo or no",
                      metavar="SUDO")
    (options, args) = parser.parse_args()
    try:
        src_dir = args[0]
    except IndexError:
        src_dir = '.'
    # Check source directory
    if not os.path.isfile(os.path.join(src_dir, 'setup.py')):
        raise RuntimeError('Run this script from directory '
                           'with setup.py, or pass in this '
                           'directory on command line')
    # update end-user documentation
    # copy_readme()
    # shellcmd("svn stat %s"%DEV_README)

    # change to source directory
    cwd = os.getcwd()
    os.chdir(src_dir)

    # build distribution
    remove_dirs(options.sudo)
    build_dist(options.readme, options.python, options.sudo)
    build_dmg(options.sudo)

    # change back to original directory
    os.chdir(cwd)
    # restore developer documentation
    # revert_readme()

if __name__ == '__main__':
    build()
