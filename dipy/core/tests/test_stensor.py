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
    slt=ten.sltensor(b,g)
    yield assert_equal(slt.A.shape[0],len(b)-1)

    print data.shape
    slt.fit(data)
    print slt.coeff

    print 'tensors',slt.tensors
    print 'fa',slt.fa
    print 'adc',slt.adc
    


