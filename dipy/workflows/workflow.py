from __future__ import division, print_function, absolute_import
from dipy.workflows.multi_io import io_iterator_


class Workflow(object):

    def __init__(self, output_strategy='append', mix_names=False):
        self._output_strategy = output_strategy
        self._mix_names = mix_names

    def get_io_iterator(self, frame):
        return io_iterator_(frame, self.run,
                            output_strategy=self._output_strategy,
                            mix_names=self._mix_names)

    def run(self):
        pass
