#!/usr/bin/env python
"""Run the py->ipynb notebook conversion.
"""
import os
from os.path import join as pjoin, abspath, splitext

from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import nbformat as nbf

import codecs

import sys

# Where things are
EG_INDEX_FNAME = abspath('examples_index.rst')
EG_SRC_DIR = abspath('examples')

# print (os.getcwd())
# if not os.getcwd().endswith(pjoin('doc', 'examples_built')):
#     raise OSError('This must be run from the doc directory')

# Copy the py files; check they are in the examples list and warn if not
eg_index_contents = open(EG_INDEX_FNAME, 'rt').read()


def clean_string(test_str):
    """Take a string and remove the newline characters"""
    if test_str[:2] == "\n\n":
        test_str = test_str[2:]
    if test_str[-2:] == "\n\n":
        test_str = test_str[:-2]
    return test_str


def make_notebook(example):
    """Generate Ipython notebook of the given example
    Paramters
    ---------
    example : str 
              The raw text of the python example

    Returns
    -------
    notebook : str
               Ipython notebook in the form of a raw text which can be
               written to a file.

    """
    allcells = example.split("\"\"\"")

    textcells = [clean_string(allcells[i]) for i in range(1, len(allcells), 2)]
    codecells = [clean_string(allcells[i]) for i in range(2, len(allcells), 2)]
    codecells = [new_code_cell(source=codecells[i], execution_count=i,)
                 for i in range(len(codecells))]
    textcells = [new_markdown_cell(source=textcells[i])
                 for i in range(len(textcells))]

    cells = []
    for i in range(0, len(allcells)):
        try:
            cells.append(textcells[i])
            cells.append(codecells[i])
        except:
            pass

    nb0 = new_notebook(cells=cells,
                       metadata={
                           'language': 'python',
                       }
                       )
    return nb0


def read_example(fname, directory="../doc/examples/"):
    """Read the example python file to convert to Ipython notebook
    Parameters
    ----------
    fname :     str
                Filename of the python example

    directory : str
                Directory in which the .py examples are located. This has
                to specified and changed based on the folder from which we 
                call the make_notebook function

                Default to ../doc/examples/
    Returns
    -------
    None
    """

    file_path = os.path.join(directory, fname)
    f = open(file_path, "r")
    fdata = f.read()
    f.close()
    return fdata


def write_notebook(notebook, fname, directory):
    """Write the given notebook into a file
    Parameters
    ----------
    notebook : str
          Notebook as raw_text

    fname : str
            Filename of the notebook

    directory: str
            Parent directory

    Returns
    -------
        Returns 1 if conversion isn't successful
    """
    if not os.path.isdir("ipython_notebooks"):
        os.mkdir("ipython_notebooks")

    nbname = codecs.open("ipython_notebooks/" + str(fname) + ".ipynb",
                         encoding='utf-8', mode='w')

    nbf.write(notebook, nbname, 4)
    nbname.close()


def valid_examples():
    """Get the valid examples to be converted"""
    flist_name = pjoin(os.path.dirname(os.getcwd()), 'doc', 'examples',
                       'valid_examples.txt')
    flist = open(flist_name, "r")
    validated_examples = flist.readlines()
    flist.close()

    # Parse "#" in lines
    validated_examples = [line.split("#", 1)[0] for line in validated_examples]
    # Remove leading and trailing white space from example names
    validated_examples = [line.strip() for line in validated_examples]
    # Remove blank lines
    validated_examples = filter(None, validated_examples)

    for example in validated_examples:
        fullpath = pjoin(EG_SRC_DIR, example)
        if not example.endswith(".py"):
            print("%s not a python file, skipping." % example)
            continue
        elif not os.path.isfile(fullpath):
            print("Cannot find file, %s, skipping." % example)
            continue

        # Check that example file is included in the docs
        file_root = example[:-3]
        if file_root not in eg_index_contents:
            msg = "Example, %s, not in index file %s."
            msg = msg % (example, EG_INDEX_FNAME)
            print(msg)
    return validated_examples

# if __name__ == "__main__":
validated_examples = valid_examples()
for fname in validated_examples:
    notebook = make_notebook(read_example(fname))
    write_notebook(notebook, fname.split(".")[0], "examples_built")
