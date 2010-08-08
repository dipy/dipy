""" Testing maskedview

"""

import numpy as np

import dipy.core.maskedview as maskedview

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric


@parametric
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
    yield assert_raises(ValueError, copy_dataS._set_shape, (sz,))
    yield assert_raises(ValueError, copy_dataS._set_shape, sz)
    yield assert_raises(ValueError, copy_dataS._set_shape, (13,-1))
    yield assert_raises(ValueError, copy_dataS._set_shape, (21,-1,-1))
    yield assert_raises(ValueError, copy_dataS._set_shape, (2,3,4,5,6,7,8))
    yield assert_raises(ValueError, copy_dataS._set_shape, (22,23,2,2,4,21))
    sz = 21*22*23
    copy_dataS.shape = (sz, -1)
    yield assert_equal(copy_dataS.shape_mask, (sz,))
    yield assert_equal(copy_dataS.shape_contents, (16,))
    copy_dataS.shape = (21, 22, 23, -1)
    copy_dataS.shape = (21, 22, 23, -1)
    yield assert_equal(copy_dataS.shape_mask, (21, 22, 23))
    yield assert_equal(copy_dataS.shape_contents, (16,))
    copy_dataS.shape = (2, 3, 7, 11, 23, 2, 2, 2, 2)
    yield assert_equal(copy_dataS.shape_mask,  (2, 3, 7, 11, 23))
    yield assert_equal(copy_dataS.shape_contents, (2, 2, 2, 2))

    #test __init__
    yield assert_equal(dataS.shape, mask.shape+shape_contents)
    yield assert_array_equal(dataS.mask, mask)
    yield assert_equal(dataS.get_size(), np.asarray(dataS).shape[0])
    yield assert_true(dataS.base is None)
    yield 4 is None
    
    #test filld
    slice_one = dataS[:,:,10,:,:,:].filled()
    yield assert_equal(type(slice_one), np.ndarray)
    yield assert_equal(slice_one.shape, mask.shape[:2]+shape_contents)
    
    #test __getitem__
    yield assert_equal(dataS_part.shape, (10, 10, 10)+shape_contents)
    yield assert_equal(dataS_part.get_size(), 
                       np.asarray(dataS_part).shape[0])
    yield assert_array_equal(dataS_part.mask, mask[10:20,10:20,10:20])

    print '1 ',dataS_zero.dtype, dataS_zero.shape#, dataS_zero
    print '2 ',data[0].dtype, data[0].shape#, np.sum(data[0]-dataS_zero)

    print np.subtract(data[0],dataS_zero)
    
    '''
    yield assert_array_equal(dataS_zero, data[0])
    
    yield assert_array_equal(dataS_one, fill_value)
    yield assert_equal(type(dataS_ind), np.ndarray)
    yield assert_equal(dataS_ind.shape, mask.shape)
    yield assert_array_equal(dataS_ind[mask], data[:,0,0,0])


    #test __array_wrap__
    new_view = dataS + np.array(0)
    yield assert_equal(type(new_view), type(dataS))

    '''
