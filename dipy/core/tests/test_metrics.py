''' Testing track_metrics module '''

from StringIO import StringIO

import numpy as np

from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.core import track_metrics as tm

from dipy import performance as pf

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
    pass

def test_segment_intersection():

    xyz=np.array([[1,1,1],[2,2,2],[2,2,2]])    
    center=[10,4,10]
    radius=1
    
    yield assert_equal, tm.any_segment_intersect_sphere(xyz,center,radius), False
    
    xyz=np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
    center=[10,10,10]
    radius=2
    
    yield assert_equal, tm.any_segment_intersect_sphere(xyz,center,radius), False

    xyz=np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
    center=[2.1,2,2.2]
    radius=2
    
    yield assert_equal, tm.any_segment_intersect_sphere(xyz,center,radius), True

def test_most_similar_zhang():

    xyz1 = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]],dtype='float32')
    xyz2 = np.array([[0,1,1],[1,0,1],[2,3,-2]],dtype='float32')
    xyz3 = np.array([[-1,0,0],[2,0,0],[2,3,0],[3,0,0]],dtype='float32')
    
    tracks=[xyz1,xyz2,xyz3]
    
    si,s=tm.most_similar_track_zhang(tracks,metric='avg')
    #pf should be much faster and the results equivalent
    si2,s2=pf.most_similar_track_zhang(tracks,metric='avg')
    
    yield assert_almost_equal, si,si2
    
    
def test_zhang_distances():
    xyz1 = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]])
    xyz2 = np.array([[0,1,1],[1,0,1],[2,3,-2]])
    # dm=array([[ 2,  2, 17], [ 3,  1, 14], [6,  2, 13], [11,  5, 14]])
    # this is the distance matrix between points of xyz1
    # and points of xyz2
    zd = tm.zhang_distances(xyz1,xyz2)
    # {'average_mean_closest_distance': 3.9166666666666665,
    # 'maximum_mean_closest_distance': 5.333333333333333,
    # 'minimum_mean_closest_distance': 2.5}
    yield assert_almost_equal, zd[0], 1.76135602742

    xyz1=xyz1.astype('float32')
    xyz2=xyz2.astype('float32')
    zd2 = pf.zhang_distances(xyz1,xyz2)
    yield assert_almost_equal, zd2[0], 1.76135602742
