"""

Test the implementation of DKI.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import numpy.testing as npt

import dipy.reconst.dki as dki
import dipy.data as dpd
import dipy.core.gradients as gt

def test_DKIModel():

    dkim = dki.DiffusionKurtosisModel(gtab)
