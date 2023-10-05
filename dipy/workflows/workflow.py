import inspect
import logging
import os

from dipy.workflows.multi_io import io_iterator_


class Workflow:
    def __init__(self, output_strategy='absolute', mix_names=False,
                 force=False, skip=False):
        """Initialize the basic workflow object.

        This object takes care of any workflow operation that is common to all
        the workflows. Every new workflow should extend this class.
        """
        self._output_strategy = output_strategy
        self._mix_names = mix_names
        self.last_generated_outputs = None
        self._force_overwrite = force
        self._skip = skip

    def get_io_iterator(self):
        """Create an iterator for IO.

        Use a couple of inspection tricks to build an IOIterator using the
        previous frame (values of local variables and other contextuals) and
        the run method's docstring.
        """
        # To manage different python versions.
        frame = inspect.stack()[1]
        if isinstance(frame, tuple):
            frame = frame[0]
        else:
            frame = frame.frame

        io_it = io_iterator_(frame, self.run,
                             output_strategy=self._output_strategy,
                             mix_names=self._mix_names)

        # Make a list out of a list of lists
        self.flat_outputs = [item for sublist in io_it.outputs for item
                             in sublist]

        if io_it.out_keys:
            self.last_generated_outputs = dict(zip(io_it.out_keys,
                                                   self.flat_outputs))
        else:
            self.last_generated_outputs = self.flat_outputs

        if self.manage_output_overwrite():
            return io_it
        else:
            return []

    def manage_output_overwrite(self):
        """Check if a file will be overwritten upon processing the inputs.

        If it is bound to happen, an action is taken depending on
        self._force_overwrite (or --force via command line). A log message is
        output independently of the outcome to tell the user something
        happened.
        """
        duplicates = []
        for output in self.flat_outputs:
            if os.path.isfile(output):
                duplicates.append(output)

        if len(duplicates) > 0:
            if self._force_overwrite:
                logging.info('The following output files are about to be'
                             ' overwritten.')

            else:
                logging.info('The following output files already exist, the '
                             'workflow will not continue processing any '
                             'further. Add the --force flag to allow output '
                             'files overwrite.')

            for dup in duplicates:
                logging.info(dup)

            return self._force_overwrite

        return True

    def run(self, *args, **kwargs):
        """Execute the workflow.

        Since this is an abstract class, raise exception if this code is
        reached (not implemented in child class or literally called on this
        class)
        """
        raise Exception('Error: {} does not have a run method.'.
                        format(self.__class__))

    def get_sub_runs(self):
        """Return No sub runs since this is a simple workflow."""
        return []

    @classmethod
    def get_short_name(cls):
        """Return A short name for the workflow used to subdivide.

        The short name is used by CombinedWorkflows and the argparser to
        subdivide the commandline parameters avoiding the trouble of having
        subworkflows parameters with the same name.

        For example, a combined workflow with dti reconstruction and csd
        reconstruction might en up with the b0_threshold parameter. Using short
        names, we will have dti.b0_threshold and csd.b0_threshold available.

        Returns class name by default but it is strongly advised to set it to
        something shorter and easier to write on commandline.

        """
        return cls.__name__
