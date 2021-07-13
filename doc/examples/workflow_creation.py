"""
============================================================
Creating a new workflow.
============================================================

A workflow is a series of DIPY_ operations with fixed inputs and outputs
that is callable via command line or another interface.

For example, after installing DIPY_, you can call anywhere from your command
line::

    dipy_nlmeans t1.nii.gz t1_denoised.nii.gz

First create your workflow (let's name this workflow file as my_workflow.py). Usually this is a python file in
the ``<../dipy/workflows>`` directory.
"""

import shutil

"""
``shutil`` Will be used for sample file manipulation.
"""

from dipy.workflows.workflow import Workflow

"""
``Workflow`` is the base class that will be extended to create our workflow.
"""

class AppendTextFlow(Workflow):

    def run(self, input_files, text_to_append='dipy', out_dir='',
            out_file='append.txt'):
        """
        Parameters
        ----------
        input_files : string
            Path to the input files. This path may contain wildcards to
            process multiple inputs at once.

        text_to_append : string, optional
            Text that will be appended to the file. (default 'dipy')

        out_dir : string, optional
            Where the resulting file will be saved. (default '')

        out_file : string, optional
            Name of the result file to be saved. (default 'append.txt')
        """

        """
        ``AppendTextFlow`` is the name of our workflow. Note that it needs
        to extend Workflow for everything to work properly. It will append
        text to a file.

        It is mandatory to have out_dir as a parameter. It is also mandatory
        to put `out_` in front of every parameter that is going to be an
        output. Lastly, all `out_` params needs to be at the end of the params
        list.

        The ``run`` docstring is very important, you need to document every
        parameter as they will be used with inspection to build the command line
        argument parser.
        """

        io_it = self.get_io_iterator()

        for in_file, out_file in io_it:

            shutil.copy(in_file, out_file)

            with open(out_file, 'a') as myfile:

                myfile.write(text_to_append)

"""
Use self.get_io_iterator() in every workflow you create. This creates
an ``IOIterator`` object that create output file names and directory structure
based on the inputs and some other advanced output strategy parameters.

By iterating on the ``IOIterator`` object you created previously you
conveniently get all input and output paths for every input file
found when globbing the input parameters.

The code in the loop is the actual workflow processing code. It can be
anything. For example, it just appends text to an input file.

This is it for the workflow! Now to be able to call it easily via command
line, you need to add this bit of code. Usually this is in a separate
executable file located in ``bin``.

The first line imports the run_flow method from the flow_runner class.
"""

from dipy.workflows.flow_runner import run_flow

"""
The second line imports the ``AppendTextFlow`` class from the newly created
``my_workflow.py`` file. In this specific case, we comment on this import
since the ``AppendTextFlow`` class is not in an external file but in the
current file.
"""

# from dipy.workflows.my_workflow import AppendTextFlow

"""
This is the method that will wrap everything that is needed to make a flow
command line ready then run it.
"""

if __name__ == "__main__":
    run_flow(AppendTextFlow())

"""
This is the only thing needed to make your workflow available through the
command line.

Now just call the script you just made with ``-h`` to see the argparser help
text::

   python workflow_creation.py --help

You should see all your parameters available along with some extra common ones
like logging file and force overwrite. Also all the documentation you wrote
about each parameter is there.

Now call it for real with a text file::

   python workflow_creation.py ./text_file.txt


.. include:: ../links_names.inc

"""
