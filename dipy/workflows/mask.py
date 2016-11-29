#!/usr/bin/env python
from __future__ import division

import inspect
import logging
import numpy as np

from dipy.io.image import load_nifti, save_nifti
from dipy.workflows.workflow import Workflow


class MaskFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'mask'

    def run(self, input_files, lb, ub=np.inf, out_dir='',
            out_mask='mask.nii.gz'):

        """ Workflow for creating a binary mask

        Parameters
        ----------
        input_files : string
           Path to image to be masked.
        lb : float
            Lower bound value.
        ub : float
            Upper bound value (default Inf)
        out_dir : string, optional
           Output directory (default input file directory)
        out_mask : string, optional
           Name of the masked file (default 'mask.nii.gz')
        """
        if lb >= ub:
            logging.error('The upper bound(less than) should be greater'
                          ' than the lower bound (greather_than).')
            return

        io_it = self.get_io_iterator()

        for input_path, out_mask_path in io_it:
            logging.info('Creating mask of {0}'.format(input_path))
            data, affine = load_nifti(input_path)
            mask = np.bitwise_and(data > lb, data < ub)
            save_nifti(out_mask_path, mask.astype(np.ubyte), affine)
            logging.info('Mask saved at {0}'.format(out_mask_path))
