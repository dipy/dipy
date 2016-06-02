#!/usr/bin/env python
from __future__ import division

import numpy as np
import logging
import inspect

from dipy.io.image import load_nifti, save_nifti
from dipy.workflows.multi_io import io_iterator_


def mask_flow(input_files,
              greater_than=0.2,
              less_than=np.inf,
              out_dir='',
              out_mask='mask.nii.gz'):

    """ Workflow for creating a binary mask

    Parameters
    ----------
    input_files : string
       Path to image to be masked.
    greater_than : float
        Default is 0.2.
    less_than : float
        Default is Inf.
    out_dir : string, optional
       Output directory (default input file directory)
    out_mask : string, optional
       Name of the masked file (default 'mask.nii.gz')
    """
    io_it = io_iterator_(inspect.currentframe(), mask_flow,
                         output_strategy='append')

    for input_path, out_mask_path in io_it:
        logging.info('Creating mask of {0}'
                     .format(input_path))
        data, affine = load_nifti(input_path)
        mask = np.bitwise_and(data > greater_than, data < less_than)
        save_nifti(out_mask_path, mask.astype(np.ubyte), affine)
        logging.info('Mask saved at {0}'.format(out_mask_path))

# TODO do similar for applying a mask
