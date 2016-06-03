import os
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import nbformat.v4 as nbf
import codecs


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


def write_notebook(nbo, fname, directory):
    """Write the given notebook into a file
    Parameters
    ----------
    nbo : str
          Notebook as raw_text

    fname : str
            Filename of the notebook

    directory: str
            Parent directory

    Returns
    -------
        Returns 1 if conversion isn't successful
    """
    file_path = os.path.join(directory, fname)
    nbname = codecs.open(str(fname) + ".ipynb",
                         encoding='utf-8', mode='w')
    try:
        nbf.write(make_notebook(data), nbname, 4)
        nbname.close()
        return 0
    except:
        return 1
