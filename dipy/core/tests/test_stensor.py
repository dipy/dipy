""" Testing tensors 

"""

import os
from os.path import join as pjoin
import numpy as np

import dipy.core.stensor as ten

from nose.tools import assert_true, assert_false, \
    assert_equal, assert_raises

from numpy.testing import assert_array_equal, \
    assert_array_almost_equal

from dipy.testing import parametric

from dipy.io import pickles as pkls


@parametric
def test_sltensor():

    fname=pjoin(os.path.dirname(__file__),'data/eg_3voxels.pkl')
    dix=pkls.load_pickle(fname)

    b=dix['bs']
    g=dix['gs']
    data=np.array(dix['data']).T    

    yield assert_equal(b[0],0.)
    slt=ten.SLTensor(b,g)
    yield assert_equal(slt.A.shape[0],len(b)-1)

    slt.fit(data)
    print 'data coeff',slt.coeff
    print 'tensors',slt.tensors
    print 'fa',slt.fa
    print 'adc',slt.adc

    data2=100*np.ones((3,3,3,len(b)))

    slt.fit(data2)
    print 'data2 coeff',slt.coeff
    print 'tensors',slt.tensors
    print 'fa',slt.fa
    print 'adc',slt.adc

    yield assert_array_equal(slt.fa,np.zeros((3,3,3)))

    data2[:,:,:,0]=250
    
    slt.fit(data2)
    print 'data2 coeff bigger S0',slt.coeff
    print 'tensors',slt.tensors
    print 'fa',slt.fa
    print 'adc',slt.adc

    data2[:,:,:,0]=50
    
    slt.fit(data2)
    print 'data2 coeff smaller S0',slt.coeff
    print 'tensors',slt.tensors
    print 'fa',slt.fa
    print 'adc',slt.adc
    

    
    

    

