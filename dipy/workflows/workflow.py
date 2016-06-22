from __future__ import division, print_function, absolute_import

import inspect
import logging
import os

from dipy.workflows.multi_io import io_iterator_


class Workflow(object):
    def __init__(self, output_strategy='append', mix_names=False,
                 force=False, skip=False):

        self._output_strategy = output_strategy
        self._mix_names = mix_names
        self.last_generated_outputs = None
        self._force_overwrite = force
        self._skip = skip

    def get_io_iterator(self):
        io_it = io_iterator_(inspect.currentframe(1), self.run,
                             output_strategy=self._output_strategy,
                             mix_names=self._mix_names)

        self.last_generated_outputs = io_it.outputs
        if self.manage_output_overwrite():
            return io_it
        else:
            return []

    def manage_output_overwrite(self):
        duplicates = []
        for output_list in self.last_generated_outputs:
            for output in output_list:
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

    def run(self):
        raise Exception('Error: {} does not have a run method.'.
                        format(self.__class__))

    def get_sub_runs(self):
        return []

    def set_sub_flows_optionals(self, opts):
        raise Exception('Error: {} does not have subworkflows.'.
                        format(self.__class__))
