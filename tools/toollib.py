"""Various utilities common to nibabel release and maintenance tools.
"""
# Library imports
import os
import sys

# Useful shorthands
pjoin = os.path.join
cd = os.chdir

# Utility functions
def c(cmd):
    """Run system command, raise SystemExit if it returns an error."""
    print ("$",cmd)
    stat = os.system(cmd)
    #stat = 0  # Uncomment this and comment previous to run in debug mode
    if stat:
        raise SystemExit(f"Command {cmd} failed with code: {stat}")


def get_dipydir():
    """Get dipy directory from command line, or assume it's the one above."""

    # Initialize arguments and check location
    try:
        dipydir = sys.argv[1]
    except IndexError:
        dipydir = '..'

    dipydir = os.path.abspath(dipydir)

    cd(dipydir)
    if not os.path.isdir('dipy') and os.path.isfile('setup.py'):
        raise SystemExit(f'Invalid dipy directory: {dipydir}')
    return dipydir

# import compileall and then get dir os.path.split
def compile_tree():
    """Compile all Python files below current directory."""
    stat = os.system('python3 -m compileall .')
    if stat:
        msg = '*** ERROR: Some Python files in tree do NOT compile! ***\n'
        msg += 'See messages above for the actual file that produced it.\n'
        raise SystemExit(msg)
