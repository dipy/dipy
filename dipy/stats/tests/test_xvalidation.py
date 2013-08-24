"""

Testing cross-validation analysis


"""
from __future__ import division, print_function, absolute_import
import numpy as np
import numpy.testing as npt
import dipy.stats.xvalidation as xval
import dipy.data as dpd
import dipy.reconst.dti as dti

def test_kfold_xval():
    """
    Test k-fold cross-validation
    """
    data, gtab = dpd.dsi_voxels()
    dm = dti.TensorModel(gtab, 'LS')
    # The data has 102 directions, so will not divide neatly into 10 bits
    npt.assert_raises(ValueError, xval.kfold_xval, dm, data, 10)
