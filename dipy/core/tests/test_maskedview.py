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
    fill_value = np.arange(16)
    fill_value.shape = (2,2,4)
    dataS = maskedview.MaskedView(mask, data, fill_value)
    dataS_part = dataS[10:20,10:20,10:20]
    dataS_zero = dataS[0,0,0]
    dataS_one = dataS[0,0,1]

    #test __init__
    yield assert_equal(dataS.shape, mask.shape)
    yield assert_array_equal(dataS.mask, mask)
    yield assert_equal(dataS.get_size(), np.asarray(dataS).shape[0])
    yield dataS.base is None
    
    #test filld
    slice_one=dataS[:,:,10].filled()
    yield assert_equal(type(slice_one), np.ndarray)
    yield assert_equal(slice_one.shape, mask.shape[:2]+shape_contents)
    #test __getitem__
    yield assert_equal(dataS_part.shape, (10, 10, 10))
    yield assert_equal(dataS_part.get_size(), 
                       np.asarray(dataS_part).shape[0])
    yield assert_array_equal(dataS_part.mask, mask[10:20,10:20,10:20])
    yield assert_array_equal(dataS_zero, data[0])
    yield assert_array_equal(dataS_one, fill_value)

    #test __array_wrap__
    new_view = dataS + np.array(0)
    yield assert_equal(type(new_view), type(dataS))

