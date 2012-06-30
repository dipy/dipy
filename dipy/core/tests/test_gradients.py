import numpy as np
from nose.tools import assert_true
from numpy.testing import assert_array_equal
from dipy.data import get_data
from dipy.core.gradients import DiffusionGradients


def test_gtab():

    fimg,fbvals,fbvecs = get_data('small_64D')
    bvals=np.load(fbvals)
    bvecs=np.load(fbvecs)
    gt = DiffusionGradients((bvals,bvecs))
    assert_array_equal(gt.bvecs,bvecs)
    gt2 = DiffusionGradients((bvals,bvecs.T))
    assert_array_equal(gt2.bvecs, bvecs.T)
    gtab = np.concatenate((bvals[:,None], bvecs),axis=1)
    gt3 = DiffusionGradients(gtab)
    assert_array_equal(gt3.bvecs, bvecs)
    assert_array_equal(gt3.bvals, bvals)
    gt4 = DiffusionGradients(gtab.T)
    assert_array_equal(gt4.bvecs, bvecs)
    assert_array_equal(gt4.bvals, bvals)
 
