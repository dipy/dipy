"""

Test the implementation of DKI.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import numpy.testing as npt

import nibabel as nib

import dipy.reconst.dki as dki
import dipy.reconst.dti as dti

import dipy.data as dpd
import dipy.core.gradients as gt

def test_DKIModel():
    fdata, fbval, fbvec = dpd.get_data('small2bval')
    data = nib.load(fdata).get_data()
    # Make sure not to generate funky values:
    data[np.where(data==0)] = 1
    gtab = gt.gradient_table(fbval, fbvec)
    dkim = dki.DiffusionKurtosisModel(gtab)
    dkif = dkim.fit(data)
    dtim = dti.TensorModel(gtab)
    dtif = dtim.fit(data)
    S0 = np.mean(data[..., gtab.b0s_mask], -1)
    pt = dtif.predict(gtab, S0)
    pk = dkif.predict(gtab, S0)
