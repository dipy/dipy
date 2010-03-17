""" Testing qball 

"""

import os
from os.path import join as pjoin
import numpy as np

import dipy.core.stensor as ten

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric

from dipy.io import pickles as pkls


#@parametric
def test_slvector():

    fname=pjoin(os.path.dirname(__file__),'data/eg_3voxels.pkl')
       
    print fname
    
    #dix=pkls.load_pickle(fname)

    #b=dix['bs']
    #g=dix['gs']
    #data=dix['data']

    #yield assert_true, b[0],0.
    
    
    #real_sh = qball.real_sph_harm(0, 0, 0, 0)
    #yield assert_true(True)
    #yield assert_false(True)


