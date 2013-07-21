"""

Test the implementation of DKI.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import numpy.testing as npt


import dipy.reconst.dki as dki
import dipy.data as dpd


def test_DKIModel():
    data, gtab = dpd.dsi_voxels()
    dkim = dki.DiffusionKurtosisModel(gtab)
