#!/usr/bin/env python
# Script to upload docs to gh-pages branch of dipy_web that will be
# automatically detected by the dipy website.
import os
import re
import sys
from os import chdir as cd
from subprocess import check_call


def sh(cmd):
    """Execute command in a subshell, return status code."""
    print("--------------------------------------------------")
    print("Executing: %s" % (cmd, ))
    print("--------------------------------------------------")
    return check_call(cmd, shell=True)


# paths
docs_repo_path = "_build/docs_repo"
docs_repo_url = "https://github.com/dipy/dipy_web.git"

if __name__ == '__main__':
    # get current directory
    startdir = os.getcwd()

    # find the source version
    info_file = '../dipy/info.py'
    info_lines = open(info_file).readlines()
    source_version = '.'.join(
        [v.split('=')[1].strip(" '\n.")
         for v in info_lines if re.match(
            '^_version_(major|minor|micro|extra)',
            v)])
    print("Source version: ", source_version)

    # check for dev tag
    if(source_version.split(".")[-1] == "dev"):
        dev = True
        print("Development Version detected")
    else:
        dev = False

    # pull current docs_repo
    if not os.path.exists(docs_repo_path):
        print("docs_repo not found, pulling from git..")
        sh("git clone %s %s" % (docs_repo_url, docs_repo_path))
    cd(docs_repo_path)
    print("Moved to " + os.getcwd())
    try:
        sh("git checkout gh-pages")
    except:
        while(1):
            print("\nLooks like gh-pages branch does not exist!")
            print("Do you want to create a new one? (y/n)")
            choice = str(input()).lower()
            if choice == 'y':
                sh("git checkout -b gh-pages")
                sh("rm -rf *")
                sh("git add .")
                sh("git commit -m 'cleaning gh-pages branch'")
                sh("git push origin gh-pages")
                break
            if choice == 'n':
                print("Please manually create a new gh-pages branch and try again.")
                sys.exit(0)
            else:
                print("Please enter valid choice ..")
    sh("git pull origin gh-pages")

    # check if docs for current version exists
    if (os.path.exists(source_version)) and (dev is not True):
        print("docs for current version already exists")
    else:
        if(dev is True):
            print("Re-building docs for development version")
        else:
            print("Building docs for a release")
        # build docs and copy to docs_repo
        cd(startdir)
        # remove old html and doctree files
        try:
            sh("rm -rf _build/json _build/doctrees")
        except:
            pass
        # generate new doc and copy to docs_repo
        sh("make api")
        sh("make rstexamples")
        sh("make json")
        sh("cp -r _build/json %s/" % (docs_repo_path,))
        cd(docs_repo_path)
        if dev is True:
            try:
                sh("rm -r %s" % (source_version,))
            except:
                pass
        sh("mv json %s" % (source_version,))
        sh("git add .")
        sh("git commit -m \"Add docs for %s\"" % (source_version,))
        sh("git push origin gh-pages")
