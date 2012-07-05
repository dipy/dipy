import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_raises
from dipy.data import get_data
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import GradientTable


def test_read_bvals_bvecs():
    fimg, fbvals, fbvecs=get_data('small_101D')
    bvals, bvecs=read_bvals_bvecs(fbvals,fbvecs)
    gt=GradientTable(bvals, bvecs)
    assert_array_equal(bvals, gt.bvals)
    assert_array_equal(bvecs, gt.bvecs)

if __name__ == '__main__':
    from numpy.testing import run_module_suite
    run_module_suite()
