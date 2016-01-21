""" Testing Free Water Elimination Model """

from __future__ import division, print_function, absolute_import

import numpy as np
import random
import dipy.reconst.dti as dti
import dipy.reconst.fwdti as fwdti
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_almost_equal)
from nose.tools import assert_raises
from dipy.reconst.dti import (from_lower_triangular, decompose_tensor)
from dipy.sims.voxel import (multi_tensor, single_tensor)
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.data import get_data

fimg, fbvals, fbvecs = get_data('small_64D')
bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
gtab = gradient_table(bvals, bvecs)

# FW model requires multishell data
bvals_2s = np.concatenate((bvals, bvals * 1.5), axis=0)
bvecs_2s = np.concatenate((bvecs, bvecs), axis=0)
gtab_2s = gradient_table(bvals_2s, bvecs_2s)


def test_fwdti_singlevoxel():
    # Simulation a typical DT and DW signal for no water contamination
    dt = np.array([0.0017, 0, 0.0003, 0, 0, 0.0003])
    evals, evecs = decompose_tensor(from_lower_triangular(dt))
    S_tissue = single_tensor(gtab_2s, S0=100, evals=evals, evecs=evecs,
                             snr=None)
    dm = dti.TensorModel(gtab_2s, 'WLS')
    dtifit = dm.fit(S_tissue)
    FAdti = dtifit.fa

    # Simulation when water contamination is added
    mevals = np.array([[0.0017, 0.0003, 0.0003], [0.003, 0.003, 0.003]])
    S_conta, peaks = multi_tensor(gtab_2s, mevals, S0=100,
                                  angles=[(0, 0), (0, 0)],
                                  fractions=[50, 50], snr=None)
    fwdm = fwdti.FreeWaterTensorModel(gtab_2s, 'WLS')
    fwefit = fwdm.fit(S_conta)
    FAfwe = fwefit.fa

    assert_almost_equal(FAdti, FAfwe)

