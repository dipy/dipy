#!/usr/bin/env python
"""
Script to generate documentation for command line utilities
"""
from os.path import join as pjoin
from os import listdir
import re
from subprocess import Popen, PIPE, CalledProcessError
import sys
import importlib
import inspect

# version comparison
from distutils.version import LooseVersion as V


def sh3(cmd):
    """
    Execute command in a subshell, return stdout, stderr
    If anything appears in stderr, print it out to sys.stderr

    https://github.com/scikit-image/scikit-image/blob/master/doc/gh-pages.py

    Copyright (C) 2011, the scikit-image team All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
    Neither the name of skimage nor the names of its contributors may be used
    to endorse or promote products derived from this software without specific
    prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
    OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
    IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
    USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
    THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    out, err = p.communicate()
    retcode = p.returncode
    if retcode:
        raise CalledProcessError(retcode, cmd)
    else:
        return out.rstrip(), err.rstrip()


def abort(error):
    print('*WARNING* Command line API documentation not generated: %s' % error)
    exit()


def get_rst_string(module_name, help_string):
    """
    Generate rst text for module
    """
    dashes = "========================\n"

    rst_text = ""
    rst_text += dashes
    rst_text += module_name + "\n"
    rst_text += dashes + "\n"
    rst_text += help_string
    return rst_text


if __name__ == '__main__':
    # package name: Eg: dipy
    package = sys.argv[1]
    # directory in which the generated rst files will be saved
    outdir = sys.argv[2]

    try:
        __import__(package)
    except ImportError as e:
        abort("Cannot import " + package)

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
    command_list = []

    workflows_folder = pjoin('..', 'dipy', 'workflows')
    workflow_class = module = importlib.import_module(
        "dipy.workflows.workflow")

    for f in listdir(workflows_folder):
        fpath = pjoin(workflows_folder, f)
        module_name = inspect.getmodulename(fpath)
        if module_name is not None:
            module = importlib.import_module(
                "dipy.workflows." + module_name)
            members = inspect.getmembers(module)
            for member_name, member_obj in members:
                if(inspect.isclass(member_obj)):
                    if (issubclass(member_obj, workflow_class.Workflow) and
                            not member_obj == workflow_class.Workflow):
                        # member_obj is a workflow
                        print("Generating docs for: ", member_name)
                        if hasattr(member_obj, 'run'):
                            help_string = inspect.getdoc(member_obj.run)

                            doc_string = get_rst_string(member_name,
                                                        help_string)
                            out_f = member_name + ".rst"
                            output_file = open(pjoin(outdir, out_f), "w")
                            output_file.write(doc_string)
                            output_file.close()
                            command_list.append(out_f)
                            print("Done")

    # generate index.rst
    print("Generating index.rst")
    index = open(pjoin(outdir, "index.rst"), "w")
    index.write("Command Line Utilities Reference\n")
    index.write("================================\n\n")
    index.write(".. toctree::\n\n")
    for cmd in command_list:
        index.write("   " + cmd)
        index.write("\n")
    index.close()
    print("Done")
