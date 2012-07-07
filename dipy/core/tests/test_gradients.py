from nose.tools import assert_true

import numpy as np
import numpy.testing as npt

from dipy.data import get_data
from dipy.core.gradients import GradientTable
from dipy.io.gradients import read_bvals_bvecs

def test_btable_prepare():

    sq2=np.sqrt(2)/2.
    bvals=1500*np.ones(7)
    bvals[0]=0
    bvecs=np.array([[0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [sq2, sq2, 0],
                    [sq2, 0, sq2],
                    [0, sq2, sq2]])
    bt = GradientTable(bvals, bvecs)
    npt.assert_array_equal(bt.bvecs, bvecs)
    bt.info
    fimg, fbvals, fbvecs = get_data('small_64D')
    bvals = np.load(fbvals)
    bvecs = np.load(fbvecs)
    bt = GradientTable(bvals,bvecs)
    npt.assert_array_equal(bt.bvecs,bvecs)
    bt2 = GradientTable(bvals,bvecs.T)
    npt.assert_array_equal(bt2.bvecs, bvecs)
    btab = np.concatenate((bvals[:,None], bvecs),axis=1)
    bt3 = GradientTable(btab)
    npt.assert_array_equal(bt3.bvecs, bvecs)
    npt.assert_array_equal(bt3.bvals, bvals)
    bt4 = GradientTable(btab.T)
    npt.assert_array_equal(bt4.bvecs, bvecs)
    npt.assert_array_equal(bt4.bvals, bvals)



def test_b0s():

    sq2=np.sqrt(2)/2.
    bvals=1500*np.ones(8)
    bvals[0]=0
    bvals[7]=0
    bvecs=np.array([[0, 0, 0],\
                    [1, 0, 0],\
                    [0, 1, 0],\
                    [0, 0, 1],\
                    [sq2, sq2, 0],\
                    [sq2, 0, sq2],\
                    [0, sq2, sq2],\
                    [0, 0, 0]])
    bt = GradientTable(bvals,bvecs)
    npt.assert_array_equal(np.where(bt.b0s_mask>0)[0], np.array([0,7]))
    npt.assert_array_equal(np.where(bt.b0s_mask==0)[0], np.arange(1,7))


def test_gtable_from_files():
    fimg, fbvals, fbvecs = get_data('small_101D')
    gt = GradientTable(fbvals, fbvecs)
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    npt.assert_array_equal(gt.bvals, bvals)
    npt.assert_array_equal(gt.bvecs, bvecs)

if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()

