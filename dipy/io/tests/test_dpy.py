import numpy as np

from dipy.io.dpy import Dpy, Dpy_Old

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

def test_dpy_read_write_indexed():

    dpw = Dpy('test.dpy','w')
    
    A=np.ones((5,3))
    B=2*A.copy()
    C=3*A.copy()
    
    dpw.write_track(A)
    dpw.write_track(B)
    dpw.write_track(C)
    
    dpw.close()
    
    dpr = Dpy('test.dpy','r')
    
    print(dpr.read_track())
    print(dpr.read_track())
    
    T=dpr.read_indexed([0,1,2,0,0,2])
    
    print T
    
    dpr.close()
    assert_array_equal(A,T[0])
    assert_array_equal(C,T[5])
    
def test_dpy_old_indexed():        

    a=2*np.ones((4,3)).astype(np.float32)
    b=3*np.ones((4,3)).astype(np.float32)
    c=4*np.ones((4,3)).astype(np.float32)
    d=5*np.ones((4,3)).astype(np.float32)
    
    T=[a,b,c,d]
    print('>>> Writing file on the fly <<<')
    fname='testold.dpyold'
    dpw=Dpy_Old(fname,'w')
    for t in T:
        dpw.write(t)    
    dpw.close()        
    print('>>> Reading file one array each time <<<')
    dpr=Dpy_Old(fname,'r')        
    while True:         
        a=dpr.read()
        if a!=None: 
            print(a)
        else:
            break
    dpr.close()    
    print('>>> Reading directly with indexes from the disk <<<')
    dpi=Dpy_Old(fname,'r')
    indices=[0,1,2,0,1,0]
    T2=dpi.read_indexed(indices)
    print T2
    dpi.close()
    assert_array_equal(T[0],T2[0])
    assert_array_equal(T[1],T2[1])
    assert_array_equal(T[2],T2[2])
    assert_array_equal(T[0],T2[5])
    
    
    
    