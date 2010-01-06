''' Testing track_metrics module '''

from StringIO import StringIO

import numpy as np

from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.core import track_metrics as tm

from dipy.core import performance as pf

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
    
def test_minimum_distance():
    
    xyz1=np.array([[1,0,0],[2,0,0]],dtype='float32')
    xyz2=np.array([[3,0,0],[4,0,0]],dtype='float32')
    yield assert_equal, pf.minimum_closest_distance(xyz1,xyz2), 1.0
    

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
    for metric in ('avg', 'min', 'max'):
        si,s=tm.most_similar_track_zhang(tracks,metric=metric)
        #pf should be much faster and the results equivalent
        si2,s2=pf.most_similar_track_zhang(tracks,metric=metric)
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


def test_approx_traj_part():
    
    t=np.linspace(0,1.75*2*np.pi,1000)

    x = np.sin(t)
    y = np.cos(t)
    z = t
    
    xyz=np.vstack((x,y,z)).T 
    
    xyza1 = tm.approximate_trajectory_partitioning(xyz,alpha=1.)
    xyza2 = tm.approximate_trajectory_partitioning(xyz,alpha=2.)
    
    yield assert_equal, len(xyza1), 12
    yield assert_equal, len(xyza2), 8
    yield assert_array_almost_equal, xyza1, np.array([[  0.00000000e+00,   1.00000000e+00,   0.00000000e+00],
       [  8.48214700e-01,   5.29652549e-01,   1.01260544e+00],
       [  8.98518156e-01,  -4.38936354e-01,   2.02521088e+00],
       [  1.03590164e-01,  -9.94620067e-01,   3.03781632e+00],
       [ -7.88784567e-01,  -6.14669754e-01,   4.05042176e+00],
       [ -9.39153678e-01,   3.43497263e-01,   5.06302720e+00],
       [ -2.06065711e-01,   9.78538156e-01,   6.07563264e+00],
       [  7.20867219e-01,   6.93073194e-01,   7.08823808e+00],
       [  9.69684032e-01,  -2.44362188e-01,   8.10084352e+00],
       [  3.06324020e-01,  -9.51927306e-01,   9.11344896e+00],
       [ -6.45193436e-01,  -7.64019260e-01,   1.01260544e+01],
       [ -1.00000000e+00,  -4.28626380e-16,   1.09955743e+01]])
    yield assert_array_almost_equal, xyza2, np.array([[  0.00000000e+00,   1.00000000e+00,   0.00000000e+00],
       [  9.81955739e-01,  -1.89110883e-01,   1.76105294e+00],
       [ -3.71397034e-01,  -9.28474148e-01,   3.52210588e+00],
       [ -8.41485297e-01,   5.40280015e-01,   5.28315882e+00],
       [  6.89665089e-01,   7.24128487e-01,   7.04421176e+00],
       [  5.80638949e-01,  -8.14161170e-01,   8.80526469e+00],
       [ -9.09275378e-01,  -4.16195011e-01,   1.05663176e+01],
       [ -1.00000000e+00,  -4.28626380e-16,   1.09955743e+01]])


def test_approximate_mdl_traj():
    
    pass

def test_cut_plane():
    dt = np.dtype(np.float32)
    refx = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]],dtype=dt)
    bundlex = [np.array([[0.5,1,0],[1.5,2,0],[2.5,3,0]],dtype=dt), 
               np.array([[0.5,2,0],[1.5,3,0],[2.5,4,0]],dtype=dt),
               np.array([[0.5,1,1],[1.5,2,2],[2.5,3,3]],dtype=dt),
               np.array([[-0.5,2,-1],[-1.5,3,-2],[-2.5,4,-3]],dtype=dt)]
    expected_hit0 = [
        [ 1.        ,  1.5       ,  0.        ,  0.70710683,  0.        ],
        [ 1.        ,  2.5       ,  0.        ,  0.70710677,  1.        ],
        [ 1.        ,  1.5       ,  1.5       ,  0.81649661,  2.        ]]
    expected_hit1 = [
        [ 2.        ,  2.5       ,  0.        ,  0.70710677,  0.        ],
        [ 2.        ,  3.5       ,  0.        ,  0.70710677,  1.        ],
        [ 2.        ,  2.5       ,  2.5       ,  0.81649655,  2.        ]]
    hitx=pf.cut_plane(bundlex,refx)
    yield assert_array_almost_equal, hitx[0], expected_hit0
    yield assert_array_almost_equal, hitx[1], expected_hit1
    # check that algorithm allows types other than float32
    bundlex[0] = np.asarray(bundlex[0], dtype=np.float64)
    hitx=pf.cut_plane(bundlex,refx)
    yield assert_array_almost_equal, hitx[0], expected_hit0
    yield assert_array_almost_equal, hitx[1], expected_hit1
    refx = np.asarray(refx, dtype=np.float64)
    hitx=pf.cut_plane(bundlex,refx)
    yield assert_array_almost_equal, hitx[0], expected_hit0
    yield assert_array_almost_equal, hitx[1], expected_hit1


def test_normalized_3vec():
    vec = [1, 2, 3]
    l2n = np.sqrt(np.dot(vec, vec))
    yield assert_array_almost_equal, l2n, pf.norm_3vec(vec)
    nvec = pf.normalized_3vec(vec)
    yield assert_array_almost_equal, np.array(vec) / l2n, nvec
    vec = np.array([[1, 2, 3]])
    yield assert_equal, vec.shape, (1, 3)
    yield assert_equal, pf.normalized_3vec(vec).shape, (3,)


def test_inner_3vecs():
    vec1 = [1, 2.3, 3]
    vec2 = [2, 3, 4.3]
    yield assert_array_almost_equal, np.inner(vec1, vec2), pf.inner_3vecs(vec1, vec2)
    vec2 = [2, -3, 4.3]
    yield assert_array_almost_equal, np.inner(vec1, vec2), pf.inner_3vecs(vec1, vec2)


def test_add_sub_3vecs():
    vec1 = np.array([1, 2.3, 3])
    vec2 = np.array([2, 3, 4.3])
    yield assert_array_almost_equal, vec1 - vec2, pf.sub_3vecs(vec1, vec2)
    yield assert_array_almost_equal, vec1 + vec2, pf.add_3vecs(vec1, vec2)
    vec2 = [2, -3, 4.3]
    yield assert_array_almost_equal, vec1 - vec2, pf.sub_3vecs(vec1, vec2)
    yield assert_array_almost_equal, vec1 + vec2, pf.add_3vecs(vec1, vec2)
