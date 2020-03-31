#!/usr/bin/env python
"""
Script to generate documentation for command line utilities
"""
import os
from os.path import join as pjoin
import re
from subprocess import Popen, PIPE, CalledProcessError
import sys
import importlib
import inspect

# version comparison
from distutils.version import LooseVersion as V

# List of workflows to ignore
SKIP_WORKFLOWS_LIST = ('Workflow', 'CombinedWorkflow')

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


def get_help_string(class_obj):
    # return inspect.getdoc(class_obj.run)
    try:
        ia_module = importlib.import_module("dipy.workflows.base")
        parser = ia_module.IntrospectiveArgumentParser()
        parser.add_workflow(class_obj())
    except Exception as e:
        abort("Error on {0}: {1}".format(class_obj.__name__, e))

    return parser.format_help()


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

    workflows_folder = pjoin('..', 'bin')
    workflow_module = importlib.import_module("dipy.workflows.workflow")

    workflow_flist = [os.path.abspath(pjoin(workflows_folder, f))
                      for f in os.listdir(workflows_folder)
                      if os.path.isfile(pjoin(workflows_folder, f)) and
                      f.lower().startswith("dipy_")]

    workflow_desc = {}
    # We get all workflows class obj in a dictionary
    for path_file in os.listdir(pjoin('..', 'dipy', 'workflows')):
        module_name = inspect.getmodulename(path_file)
        if module_name is None:
            continue

        module = importlib.import_module("dipy.workflows." + module_name)
        members = inspect.getmembers(module)
        d_wkflw = {name: {"module": obj, "help": get_help_string(obj)}
                   for name, obj in members
                   if inspect.isclass(obj) and
                   issubclass(obj, workflow_module.Workflow) and
                   name not in SKIP_WORKFLOWS_LIST
                   }

        workflow_desc.update(d_wkflw)

    cmd_list = []
    for fpath in workflow_flist:
        fname = os.path.basename(fpath)
        with open(fpath) as file_object:
            flow_name = set(re.findall(r"[A-Z]\w+Flow", file_object.read(),
                                       re.X | re.M))

        if not flow_name or len(flow_name) != 1:
            continue

        flow_name = list(flow_name)[-1]
        print("Generating docs for: {0} ({1})".format(fname, flow_name))
        out_fname = fname + ".rst"
        with open(pjoin(outdir, out_fname), "w") as fp:
            dashes = "========================"
            fp.write("\n{0}\n{1}\n{0}\n\n".format(dashes, fname))
            # Trick to avoid docgen_cmd.py as cmd line
            help_txt = workflow_desc[flow_name]["help"]
            help_txt = help_txt.replace("docgen_cmd.py", fname)
            fp.write(help_txt)

        cmd_list.append(out_fname)
        print("Done")

    # generate index.rst
    print("Generating index.rst")
    with open(pjoin(outdir, "index.rst"), "w") as index:
        index.write(".. _workflows_reference:\n\n")
        index.write("Command Line Utilities Reference\n")
        index.write("================================\n\n")
        index.write(".. toctree::\n\n")
        for cmd in cmd_list:
            index.write("   " + cmd)
            index.write("\n")
    print("Done")
