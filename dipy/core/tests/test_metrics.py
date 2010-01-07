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
       [  8.53992868e-01,   5.20284712e-01,   1.02361202e+00],
       [  8.88638867e-01,  -4.58607637e-01,   2.04722404e+00],
       [  7.06975657e-02,  -9.97497797e-01,   3.07083606e+00],
       [ -8.15073141e-01,  -5.79358071e-01,   4.09444808e+00],
       [ -9.18837755e-01,   3.94635503e-01,   5.11806010e+00],
       [ -1.41041332e-01,   9.90003708e-01,   6.14167212e+00],
       [  7.72074457e-01,   6.35532086e-01,   7.16528415e+00],
       [  9.44438405e-01,  -3.28688452e-01,   8.18889617e+00],
       [  2.10679270e-01,  -9.77555239e-01,   9.21250819e+00],
       [ -7.25211999e-01,  -6.88525640e-01,   1.02361202e+01],
       [ -1.00000000e+00,  -4.28626380e-16,   1.09955743e+01]])

    yield assert_array_almost_equal, xyza2, np.array([[  0.00000000e+00,   1.00000000e+00,   0.00000000e+00],
       [  9.79814838e-01,  -1.99907185e-01,   1.77205952e+00],
       [ -3.91744053e-01,  -9.20074235e-01,   3.54411904e+00],
       [ -8.23189936e-01,   5.67766086e-01,   5.31617856e+00],
       [  7.20867219e-01,   6.93073194e-01,   7.08823808e+00],
       [  5.34976863e-01,  -8.44866709e-01,   8.86029760e+00],
       [ -9.34758657e-01,  -3.55283343e-01,   1.06323571e+01],
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
