""" Testing maskedview

"""

import numpy as np

import dipy.reconst.maskedview as maskedview

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_MaskedView():
    mask = np.random.random((21,22,23)) > .5
    mask[0,0,0] = True
    mask[0,0,1] = False
    shape_contents = (2,2,4)
    data = np.random.random((mask.sum(),)+shape_contents)
    fill_value = 99
    dataS = maskedview.MaskedView(mask, data, fill_value)
    dataS_part = dataS[10:20,10:20,10:20]
    dataS_zero = dataS[0,0,0]
    dataS_one = dataS[0,0,1]
    dataS_ind = dataS[...,0,0,0]

    #test shape
    sz = 21*22*23*2*2*4
    copy_dataS = dataS[:]
    assert_raises(ValueError, copy_dataS._set_shape, (sz,))
    assert_raises(ValueError, copy_dataS._set_shape, sz)
    assert_raises(ValueError, copy_dataS._set_shape, (13,-1))
    assert_raises(ValueError, copy_dataS._set_shape, (21,-1,-1))
    assert_raises(ValueError, copy_dataS._set_shape, (2,3,4,5,6,7,8))
    assert_raises(ValueError, copy_dataS._set_shape, (22,23,2,2,4,21))
    sz = 21*22*23
    copy_dataS.shape = (sz, -1)
    assert_equal(copy_dataS.shape_mask, (sz,))
    assert_equal(copy_dataS.shape_contents, (16,))
    copy_dataS.shape = (21, 22, 23, -1)
    copy_dataS.shape = (21, 22, 23, -1)
    assert_equal(copy_dataS.shape_mask, (21, 22, 23))
    assert_equal(copy_dataS.shape_contents, (16,))
    copy_dataS.shape = (2, 3, 7, 11, 23, 2, 2, 2, 2)
    assert_equal(copy_dataS.shape_mask,  (2, 3, 7, 11, 23))
    assert_equal(copy_dataS.shape_contents, (2, 2, 2, 2))

    #test __init__
    assert_equal(dataS.shape, mask.shape+shape_contents)
    assert_array_equal(dataS.mask, mask)
    assert_equal(dataS.get_size(), np.asarray(dataS).shape[0])
    assert_true(dataS.base is None)
    
    #test filld
    slice_one = dataS[:,:,10,:,:,:].filled()
    assert_equal(type(slice_one), np.ndarray)
    assert_equal(slice_one.shape, mask.shape[:2]+shape_contents)
    
    #test __getitem__
    assert_equal(dataS_part.shape, (10, 10, 10)+shape_contents)
    assert_equal(dataS_part.get_size(), 
                       np.asarray(dataS_part).shape[0])
    assert_array_equal(dataS_part.mask, mask[10:20,10:20,10:20])

    #this is the correct behaviour
    assert_array_equal(dataS_zero, data[0])
    
    #this is the correct behaviour
    assert_array_equal(dataS_one, fill_value)
    
    assert_equal(type(dataS_ind), np.ndarray)
    assert_equal(dataS_ind.shape, mask.shape)
    assert_array_equal(dataS_ind[mask], data[:,0,0,0])

    #test __array_wrap__

    new_view = dataS + np.array(0)
    
    assert_equal(type(new_view), type(dataS))


    

