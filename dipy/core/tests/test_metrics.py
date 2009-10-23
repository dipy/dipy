''' Testing track_metrics module '''

from StringIO import StringIO

import numpy as np

from nose.tools import assert_true, assert_false, assert_equal

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.core import track_metrics as tm


def test_splines():

    #create a helix
    t=np.linspace(0,1.75*2*np.pi,100)

    x = np.sin(t)
    y = np.cos(t)
    z = t

    # add noise
    x+= np.random.normal(scale=0.1, size=x.shape)
    y+= np.random.normal(scale=0.1, size=y.shape)
    z+= np.random.normal(scale=0.1, size=z.shape)
    
    xyz=np.vstack((x,y,z)).T    
    # get the B-splines
    xyzn=tm.spline(xyz,3,2,-1)
    
   




 
