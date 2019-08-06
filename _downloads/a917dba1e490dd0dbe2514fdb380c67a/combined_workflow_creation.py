"""
================================
Creating a new combined workflow
================================

A ``CombinedWorkflow`` is a series of DIPY_ workflows organized together in a
way that the output of a workflow serves as input for the next one.
"""

"""
First create your ``CombinedWorkflow`` class. Your ``CombinedWorkflow`` class
file is usually located in the ``dipy/workflows`` directory.
"""

from dipy.workflows.combined_workflow import CombinedWorkflow

"""
``CombinedWorkflow`` is the base class that will be extended to create our
combined workflow.
"""

from dipy.workflows.denoise import NLMeansFlow
from dipy.workflows.segment import MedianOtsuFlow

"""
``MedianOtsuFlow`` and ``NLMeansFlow`` will be combined to create our
processing section.
"""


class DenoiseAndSegment(CombinedWorkflow):

    """
    ``DenoiseAndSegment`` is the name of our combined workflow. Note that
    it needs to extend CombinedWorkflow for everything to work properly.
    """

    def _get_sub_flows(self):
        return [
            NLMeansFlow,
            MedianOtsuFlow
        ]

        """
        It is mandatory to implement this method if you want to make all the sub
        workflows parameters available in commandline.
        """

    def run(self, input_files, out_dir='', out_file='processed.nii.gz'):
        """
        Parameters
        ----------
        input_files : string
            Path to the input files. This path may contain wildcards to
            process multiple inputs at once.

        out_dir : string, optional
            Where the resulting file will be saved. (default '')

        out_file : string, optional
            Name of the result file to be saved. (default 'processed.nii.gz')
        """

        """
        Just like a normal workflow, it is mandatory to have out_dir as a
        parameter. It is also mandatory to put 'out_' in front of every
        parameter that is going to be an output. Lastly, all out_ params needs
        to be at the end of the params list.

        The class docstring part is very important, you need to document
        every parameter as they will be used with inspection to build the
        command line argument parser.
        """

        io_it = self.get_io_iterator()

        for in_file, out_file in io_it:
            nl_flow = NLMeansFlow()
            self.run_sub_flow(nl_flow, in_file, out_dir=out_dir)
            denoised = nl_flow.last_generated_outputs['out_denoised']

            me_flow = MedianOtsuFlow()
            self.run_sub_flow(me_flow, denoised, out_dir=out_dir)

"""
Use ``self.get_io_iterator()`` in every workflow you create. This creates
an ``IOIterator`` object that create output file names and directory structure
based on the inputs and some other advanced output strategy parameters.

Iterating on the ``IOIterator`` object you created previously you
conveniently get all input and output paths for every input file
found when globbin the input parameters.

In the ``IOIterator`` loop you can see how we create a new ``NLMeans`` workflow
then run it using ``self.run_sub_flow``. Running it this way will pass any
workflow specific parameter that was retreived from the command line and will
append the ones you specify as optional parameters (``out_dir`` in this case).

Lastly, the outputs paths are retrived using
``workflow.last_generated_outputs``. This allows to use ``denoise`` as the
input for the ``MedianOtsuFlow``.
"""

"""

This is it for the combined workflow class! Now to be able to call it easily via
command line, you need this last bit of code. It is usually in an executable
file located in ``bin``.
"""

from dipy.workflows.flow_runner import run_flow
"""
This is the method that will wrap everything that is needed to make a workflow
ready then run it.
"""

if __name__ == "__main__":
    run_flow(DenoiseAndSegment())
"""
This is the only thing needed to make your workflow available through command
line.

Now just call the script you just made with ``-h`` to see the argparser help
text::

   python combined_workflow_creation.py --help

You should see all your parameters available along with some extra common ones
like logging file and force overwrite. Also all the documentation you wrote
about each parameter is there. Also note that every sub workflow optional
parameter is available.

Now call it for real with a nifti file to see the results. Experiment
with the parameters and see the results::

   python combined_workflow_creation.py volume.nii.gz

.. include:: ../links_names.inc

"""
