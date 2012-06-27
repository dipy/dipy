from nose.tools import assert_true
from numpy.testing import assert_array_equal
import numpy as np
from dipy.data import get_data
from dipy.core.gtable import GradientTable

def test_gtab():

    fimg,fbvals,fbvecs = get_data('small_64D')

    bvals=np.load(fbvals)
    bvecs=np.load(fbvecs)

    gt = GradientTable((bvals,bvecs))
    assert_array_equal(gt.bvecs,bvecs)

    gt2 = GradientTable((bvals,bvecs.T))
    assert_array_equal(gt2.bvecs, bvecs.T)

    gtab = np.concatenate((bvals[:,None], bvecs),axis=1)
    gt3 = GradientTable(gtab)
    assert_array_equal(gt3.bvecs, bvecs)
    assert_array_equal(gt3.bvals, bvals)

    gt4 = GradientTable(gtab.T)
    assert_array_equal(gt4.bvecs, bvecs)
    assert_array_equal(gt4.bvals, bvals)
 
