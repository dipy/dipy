from __future__ import division, print_function, absolute_import

import logging
import numpy as np

from dipy.workflows.workflow import Workflow

class SummarizeData(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'dti'