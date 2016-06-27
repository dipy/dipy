#!/usr/bin/env python
"""
Script to generate documentation for command line utilities
"""
import sys
import re
from os.path import join as pjoin
from os import listdir
from subprocess import Popen, PIPE, CalledProcessError

# version comparison
from distutils.version import LooseVersion as V


def sh3(cmd):
    """Execute command in a subshell, return stdout, stderr
    If anything appears in stderr, print it out to sys.stderr"""
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    out, err = p.communicate()
    retcode = p.returncode
    if retcode:
        raise CalledProcessError(retcode, cmd)
    else:
        return out.rstrip(), err.rstrip()


def abort(error):
    print('*WARNING* API documentation not generated: %s' % error)
    exit()


if __name__ == '__main__':
    # package name: Eg: dipy
    package = sys.argv[1]
    # directory in which the generated rst files will be saved
    outdir = sys.argv[2]

    try:
        __import__(package)
    except ImportError as e:
        abort("Can not import " + package)

    module = sys.modules[package]

    # Check that the source version is equal to the installed
    # version. If the versions mismatch the API documentation sources
    # are not (re)generated. This avoids automatic generation of documentation
    # for older or newer versions if such versions are installed on the system.

    installed_version = V(module.__version__)

    info_file = pjoin('..', package, 'info.py')
    info_lines = open(info_file).readlines()
    source_version = '.'.join(
        [v.split('=')[1].strip(" '\n.")
         for v in info_lines
         if re.match('^_version_(major|minor|micro|extra)', v)])
    print('***', source_version)

    if source_version != installed_version:
        abort("Installed version does not match source version")

    # generate docs
    bin_folder = pjoin('..', 'bin')

    for f in listdir(bin_folder):
        if f.startswith("dipy_"):
            try:
                help_string, err = sh3("%s -h" % (f,))
            except CalledProcessError:
                print("Could not execute command %s" % (f))
                continue
            print("Generating docs for %s..." % (f,))
            help_string = help_string.decode("utf-8")
            err = err.decode("utf-8")
            if help_string == "":
                help_string = err
            out_f = f + ".txt"
            output_file = open(pjoin(outdir, out_f), "w")
            output_file.write(help_string)
            output_file.close()
            print("Done")
