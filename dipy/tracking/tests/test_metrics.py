''' Testing track_metrics module '''

from StringIO import StringIO
import numpy as np
from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal
from dipy.tracking import metrics as tm
from dipy.tracking import distances as pf



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
    # get the B-splines smoothed result
    xyzn=tm.spline(xyz,3,2,-1)
    
    

    

def test_segment_intersection():
    xyz=np.array([[1,1,1],[2,2,2],[2,2,2]])    
    center=[10,4,10]
    radius=1    
    assert_equal(tm.intersect_sphere(xyz,center,radius), False)    
    xyz=np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
    center=[10,10,10]
    radius=2    
    assert_equal( tm.intersect_sphere(xyz,center,radius), False)
    xyz=np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
    center=[2.1,2,2.2]
    radius=2    
    assert_equal( tm.intersect_sphere(xyz,center,radius), True)




def test_normalized_3vec():
    vec = [1, 2, 3]
    l2n = np.sqrt(np.dot(vec, vec))
    assert_array_almost_equal(l2n, pf.norm_3vec(vec))
    nvec = pf.normalized_3vec(vec)
    assert_array_almost_equal( np.array(vec) / l2n, nvec)
    vec = np.array([[1, 2, 3]])
    assert_equal(vec.shape, (1, 3))
    assert_equal(pf.normalized_3vec(vec).shape, (3,))


def test_inner_3vecs():
    vec1 = [1, 2.3, 3]
    vec2 = [2, 3, 4.3]
    assert_array_almost_equal(np.inner(vec1, vec2), pf.inner_3vecs(vec1, vec2))
    vec2 = [2, -3, 4.3]
    assert_array_almost_equal(np.inner(vec1, vec2), pf.inner_3vecs(vec1, vec2))


def test_add_sub_3vecs():
    vec1 = np.array([1, 2.3, 3])
    vec2 = np.array([2, 3, 4.3])
    assert_array_almost_equal( vec1 - vec2, pf.sub_3vecs(vec1, vec2))
    assert_array_almost_equal( vec1 + vec2, pf.add_3vecs(vec1, vec2))
    vec2 = [2, -3, 4.3]
    assert_array_almost_equal( vec1 - vec2, pf.sub_3vecs(vec1, vec2))
    assert_array_almost_equal( vec1 + vec2, pf.add_3vecs(vec1, vec2))
    

    
    
    

