import numpy as np

from dipy.io.dpy import Dpy

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_dpy_indexed():        

    a=2*np.ones((4,3)).astype(np.float32)
    b=3*np.ones((4,3)).astype(np.float32)
    c=4*np.ones((4,3)).astype(np.float32)
    d=5*np.ones((4,3)).astype(np.float32)
                
    T=[a,b,c,d]
    print('>>> Writing file on the fly <<<')
    fname='test.dpy'
    dpw=Dpy(fname,'w')
    for t in T:
        dpw.write(t)    
    dpw.close()        
    print('>>> Reading file one array each time <<<')
    dpr=Dpy(fname,'r')        
    while True:         
        a=dpr.read()
        if a!=None: 
            print(a)
        else:
            break
    dpr.close()    
    print('>>> Reading directly with indexes from the disk <<<')
    dpi=Dpy(fname,'r')
    indices=[0,1,2,0,1,0]
    T2=dpi.read_indexed(indices)
    print T2
    dpi.close()
    assert_array_equal(T[0],T2[0])
    assert_array_equal(T[1],T2[1])
    assert_array_equal(T[2],T2[2])
    assert_array_equal(T[0],T2[5])
    
    
    
    