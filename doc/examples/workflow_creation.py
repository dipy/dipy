"""
============================================================
Creating a new workflow.
============================================================

A workflow is a series of dipy operations with fixed inputs and outputs
that is callable via commandline.

For example, after installing dipy, you can call anywhere from your command line:
$ dipy_nlmeans t1.nii.gz t1_denoised.nii.gz
"""

"""
First create your workflow file append_text_flow.py in the ../dipy/workflows
directory. Then put the following code in it.
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
    """
    ``AppendTextFlow`` is the name of our workflow. Note that it needs to extend
    Workflow for everything to work properly. It will append text to a file.
    """

    def run(self, input_files, text_to_append='dipy', out_dir='',
            out_file='append.txt'):
        """
            ``AppendTextFlow`` is the name of our workflow. Note that it needs
            to extend Workflow for everything to work properly. It will append
            text to a file.

            It is mandatory to have out_dir as a parameter. It is also mandatory
            to put 'out_' in front of every parameter that is going to be an
            output. Lastly, all out_ params needs to be at the end of the params
            list.

            The following docstring part is very important, you need to document
            every parameter as they will be used with inspection to build the
            command line argument parser.

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

        io_it = self.get_io_iterator()
        """
        Use this in every workflow you create. This creates and ``IOIterator``
        object that create output file names and directory structure based on
        the inputs and some other advnaced output strategy parameters.
        """

        for in_file, out_file in io_it:

            """
            Iterating on the ``IOIterator`` object you created previously you
            conveniently get all input and output paths for every input file
            found when globbin the input parameters.
            """


            shutil.copy(in_file, out_file)

            """
            Create the new file.
            """

            with open(out_file, 'a') as myfile:
                myfile.write(text_to_append)

            """
            Append the text and close file.
            """

"""
This is it for the workflow file! Now to be able to call it easily via command
line, you need to create one last file. In ../dipy/bin/ create a file named
dipy_append_text (note that there is not extension, this file is to be
executable)
Then put the folowing code in you new file.
"""

from dipy.workflows.append_text_flow import AppendTextFlow
"""
This is your previously created workflow.
"""

from dipy.workflows.flow_runner import run_flow
"""
This is the method that will wrap everything that is need to make a flow
command line ready then run it.
"""

if __name__ == "__main__":
    run_flow(AppendTextFlow())
"""
This is the only thing needed to make your workflow available through command
line.

Now just call the script you just made with -h to see the argparser help text.

\> dipy_append_text --help

You should see all your parameters available along with some extra common ones
like logging file and force overwrite. Also all the documentation you wrote
about each parameter is there.

Now call it for real with a text file

\> dipy_append_text ./text_file.txt
"""








