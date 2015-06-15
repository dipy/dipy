"""
Testing DKI
"""

from __future__ import division, print_function, absolute_import

import numpy as np

from nose.tools import (assert_true, assert_equal,
                        assert_almost_equal, assert_raises)
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_)

from dipy.sims.voxel import multi_tensor_dki

import dipy.reconst.dti as dti

import dipy.reconst.dki as dki

from dipy.io.gradients import read_bvals_bvecs

from dipy.core.gradients import gradient_table

from dipy.data import get_data

from dipy.reconst.dti import (from_lower_triangular, decompose_tensor)


fimg, fbvals, fbvecs = get_data('small_64D')
bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
gtab = gradient_table(bvals, bvecs)

# 2 shells for techniques that requires multishell data
bvals_2s = np.concatenate((bvals, bvals * 2), axis=0)
bvecs_2s = np.concatenate((bvecs, bvecs), axis=0)
gtab_2s = gradient_table(bvals_2s, bvecs_2s)


def test_dki_ols_fit():
    """DKI ols fit is tested"""

    # Two crossing fibers are simulated
    mevals = np.array([[0.00099, 0, 0], [0.00226, 0.00087, 0.00087],
                       [0.00099, 0, 0], [0.00226, 0.00087, 0.00087]])
    angles = [(80, 10), (80, 10), (20, 30), (20, 30)]
    fie = 0.49
    frac = [fie*50, (1 - fie)*50, fie*50, (1 - fie)*50]
    signal, dt, kt = multi_tensor_dki(gtab_2s, mevals, angles=angles,
                                      fractions=frac, snr=None)

    evals, evecs = decompose_tensor(from_lower_triangular(dt))
    ref_params = np.concatenate((evals, evecs[0], evecs[1], evecs[2], kt),
                                axis=0)

    dkiM = dki.DKIModel(gtab_2s)
    dkiF = dkiM.fit(signal)

    assert_array_almost_equal(dkiF.model_params, ref_params)
