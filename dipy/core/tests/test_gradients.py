import numpy as np
from nose.tools import assert_true
from numpy.testing import assert_array_equal
from dipy.data import get_data
from dipy.core.gradients import GradientTable


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
    bt = GradientTable(bvals,bvecs)
    assert_array_equal(bt.bvecs,bvecs)
    bt.info
    fimg,fbvals,fbvecs = get_data('small_64D')
    bvals=np.load(fbvals)
    bvecs=np.load(fbvecs)
    bt = GradientTable(bvals,bvecs)
    assert_array_equal(bt.bvecs,bvecs)
    bt2 = GradientTable(bvals,bvecs.T)
    assert_array_equal(bt2.bvecs, bvecs)
    btab = np.concatenate((bvals[:,None], bvecs),axis=1)
    bt3 = GradientTable(btab)
    assert_array_equal(bt3.bvecs, bvecs)
    assert_array_equal(bt3.bvals, bvals)
    bt4 = GradientTable(btab.T)
    assert_array_equal(bt4.bvecs, bvecs)
    assert_array_equal(bt4.bvals, bvals)

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
    assert_array_equal(np.where(bt.b0s_mask>0)[0], np.array([0,7]))
    assert_array_equal(np.where(bt.b0s_mask==0)[0], np.arange(1,7))


if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()

