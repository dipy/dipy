import numpy as np
from nose.tools import assert_true
from numpy.testing import assert_array_equal
from dipy.data import get_data
from dipy.core.gradients import BTable


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
    bt = BTable(bvals,bvecs)
    assert_array_equal(bt.bvecs,bvecs)
    bt.info
    fimg,fbvals,fbvecs = get_data('small_64D')
    bvals=np.load(fbvals)
    bvecs=np.load(fbvecs)
    bt = BTable(bvals,bvecs)
    assert_array_equal(bt.bvecs,bvecs)
    bt2 = BTable(bvals,bvecs.T)
    assert_array_equal(bt2.bvecs, bvecs)
    btab = np.concatenate((bvals[:,None], bvecs),axis=1)
    bt3 = BTable(btab)
    assert_array_equal(bt3.bvecs, bvecs)
    assert_array_equal(bt3.bvals, bvals)
    bt4 = BTable(btab.T)
    assert_array_equal(bt4.bvecs, bvecs)
    assert_array_equal(bt4.bvals, bvals)
    bt5 = BTable(fbvals,fbvecs)
    assert_array_equal(bt4.bvecs, bvecs)
    assert_array_equal(bt4.bvals, bvals)
    fimg,fbvals,fbvecs = get_data('small_101D')
    bt5 = BTable(fbvals,fbvecs)
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
    bt = BTable(bvals,bvecs)
    assert_array_equal(bt.b0s_indices,np.array([0,7]))
    assert_array_equal(bt.nonb0s_indices,np.arange(1,7))


    

 
