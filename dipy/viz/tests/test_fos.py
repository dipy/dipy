""" Testing 

"""

import numpy as np

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.viz import fos


def test_fos_functions():
    
    # Create a renderer
    r=fos.ren()    
    
    # Create 2 lines with 2 different colors
    lines=[np.random.rand(10,3),np.random.rand(20,3)]    
    colors=np.random.rand(2,3)
    c=fos.line(lines,colors)    
    fos.add(r,c)    

    # Create a volume and return a volumetric actor using volumetric rendering
        
    vol=100*np.random.rand(100,100,100)
    vol=vol.astype('uint8')    
    r = fos.ren()
    v = fos.volume(vol)
    fos.add(r,v)
    
    # Remove all objects
    fos.rm_all(r)
    
    # Put some text
    
    l=fos.label(r,text='Yes Men')
    fos.add(r,l)

    # Show everything
    #fos.show(r)

    