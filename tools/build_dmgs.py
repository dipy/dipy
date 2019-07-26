#!/usr/bin/env python
"""Script to build dmgs for buildbot builds

Examples
--------
%(prog)s "dipy-dist/dipy*-0.6.0-py*mpkg"

Note quotes around the globber first argument to protect it from shell
globbing.

"""
import os
from os.path import join as pjoin, isfile, isdir
import sys
import shutil
from glob import glob
from functools import partial
from subprocess import check_call
import warnings

from argparse import ArgumentParser, RawDescriptionHelpFormatter

my_call = partial(check_call, shell=True)

BUILDBOT_LOGIN = "buildbot@nipy.bic.berkeley.edu"
BUILDBOT_HTML = "nibotmi/public_html/"

def main():
    parser = ArgumentParser(description=__doc__,
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('globber', type=str,
                        help='glob to serch for build mpkgs')
    parser.add_argument('--out-path', type=str, default='mpkg-dist',
                        help='path for output files (default="mpkg-dist")',
                        metavar='OUTPATH')
    parser.add_argument('--clobber', action='store_true',
                        help='Delete OUTPATH if exists')
    args = parser.parse_args()
    globber = args.globber
    out_path = args.out_path
    address = "{0}:{1}{2}".format(BUILDBOT_LOGIN, BUILDBOT_HTML, globber)
    if isdir(out_path):
        if not args.clobber:
            raise RuntimeError('Path {0} exists and "clobber" not set'.format(
                out_path))
        shutil.rmtree(out_path)
    os.mkdir(out_path)
    cwd = os.path.abspath(os.getcwd())
    os.chdir(out_path)
    try:
        my_call('scp -r {0} .'.format(address))
        found_mpkgs = sorted(glob('*.mpkg'))
        for mpkg in found_mpkgs:
            pkg_name, ext = os.path.splitext(mpkg)
            assert ext == '.mpkg'
            my_call('sudo reown_mpkg {0} root admin'.format(mpkg))
            os.mkdir(pkg_name)
            pkg_moved = pjoin(pkg_name, mpkg)
            os.rename(mpkg, pkg_moved)
            readme = pjoin(pkg_moved, 'Contents', 'Resources', 'ReadMe.txt')
            if isfile(readme):
                shutil.copy(readme, pkg_name)
            else:
                warnings.warn("Could not find readme with " + readme)
            my_call('sudo hdiutil create {0}.dmg -srcfolder ./{0}/ -ov'.format(pkg_name))
    finally:
        os.chdir(cwd)


if __name__ == '__main__':
    main()
