''' Testing track_metrics module '''

from StringIO import StringIO

import numpy as np

from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.core import track_metrics as tm

from dipy.core import track_performance as pf

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


def test_most_similar_mam():
    xyz1 = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]],dtype='float32')
    xyz2 = np.array([[0,1,1],[1,0,1],[2,3,-2]],dtype='float32')
    xyz3 = np.array([[-1,0,0],[2,0,0],[2,3,0],[3,0,0]],dtype='float32')
    tracks=[xyz1,xyz2,xyz3]
    for metric in ('avg', 'min', 'max'):
        si,s=tm.most_similar_track_mam(tracks,metric=metric)
        #pf should be much faster and the results equivalent
        si2,s2=pf.most_similar_track_mam(tracks,metric=metric)
        yield assert_almost_equal, si,si2
    
    
def test_bundles_distances_mam():
    xyz1A = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]],dtype='float32')
    xyz2A = np.array([[0,1,1],[1,0,1],[2,3,-2]],dtype='float32')
    xyz1B = np.array([[-1,0,0],[2,0,0],[2,3,0],[3,0,0]],dtype='float32')
    tracksA = [xyz1A, xyz2A]
    tracksB = [xyz1B, xyz1A, xyz2A]
    for metric in ('avg', 'min', 'max'):
        DM1 = tm.bundles_distances_mam(tracksA, tracksB, metric=metric)
        DM2 = pf.bundles_distances_mam(tracksA, tracksB, metric=metric)
        yield assert_array_almost_equal, DM1,DM2
    
    
def test_mam_distances():
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


def test_approx_ei_traj():
    
    segs=100
    
    t=np.linspace(0,1.75*2*np.pi,segs)
    
    x =t 
    y=5*np.sin(5*t)
    z=np.zeros(x.shape)
    
    xyz=np.vstack((x,y,z)).T    
    
    xyza=pf.approx_polygon_track(xyz)
    yield assert_equal, len(xyza), 27

def test_approx_mdl_traj():
    
    t=np.linspace(0,1.75*2*np.pi,100)

    x = np.sin(t)
    y = np.cos(t)
    z = t
    
    xyz=np.vstack((x,y,z)).T 
    
    xyza1 = pf.approximate_mdl_trajectory(xyz,alpha=1.)
    xyza2 = pf.approximate_mdl_trajectory(xyz,alpha=2.)
    
    yield assert_equal, len(xyza1), 10
    yield assert_equal, len(xyza2), 8
    yield assert_array_almost_equal, xyza1, np.array([[  0.00000000e+00,   1.00000000e+00,   0.00000000e+00],
       [  9.39692621e-01,   3.42020143e-01,   1.22173048e+00],
       [  6.42787610e-01,  -7.66044443e-01,   2.44346095e+00],
       [ -5.00000000e-01,  -8.66025404e-01,   3.66519143e+00],
       [ -9.84807753e-01,   1.73648178e-01,   4.88692191e+00],
       [ -1.73648178e-01,   9.84807753e-01,   6.10865238e+00],
       [  8.66025404e-01,   5.00000000e-01,   7.33038286e+00],
       [  7.66044443e-01,  -6.42787610e-01,   8.55211333e+00],
       [ -3.42020143e-01,  -9.39692621e-01,   9.77384381e+00],
       [ -1.00000000e+00,  -4.28626380e-16,   1.09955743e+01]])

    yield assert_array_almost_equal, xyza2, np.array([[  0.00000000e+00,   1.00000000e+00,   0.00000000e+00],
       [  9.95471923e-01,  -9.50560433e-02,   1.66599610e+00],
       [ -1.89251244e-01,  -9.81928697e-01,   3.33199221e+00],
       [ -9.59492974e-01,   2.81732557e-01,   4.99798831e+00],
       [  3.71662456e-01,   9.28367933e-01,   6.66398442e+00],
       [  8.88835449e-01,  -4.58226522e-01,   8.32998052e+00],
       [ -5.40640817e-01,  -8.41253533e-01,   9.99597663e+00],
       [ -1.00000000e+00,  -4.28626380e-16,   1.09955743e+01]])


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
    
def test_point_track_sq_distance():
    
    t=np.array([[0,0,0],[1,1,1],[2,2,2]],dtype='f4')
    p=np.array([-1,-1.,-1],dtype='f4')
    yield assert_equal, pf.point_track_sq_distance_check(t,p,.2**2), False    
    yield pf.point_track_sq_distance_check(t,p,2**2), True
    t=np.array([[0,0,0],[1,0,0],[2,2,0]],dtype='f4')
    p=np.array([.5,0,0],dtype='f4')
    yield assert_equal, pf.point_track_sq_distance_check(t,p,.2**2), True
    p=np.array([.5,1,0],dtype='f4')
    yield assert_equal, pf.point_track_sq_distance_check(t,p,.2**2), False
    
def test_track_roi_intersection_check():    
    roi=np.array([[0,0,0],[1,0,0],[2,0,0]],dtype='f4')    
    t=np.array([[0,0,0],[1,1,1],[2,2,2]],dtype='f4')
    yield assert_equal, pf.track_roi_intersection_check(t,roi,1), True
    t=np.array([[0,0,0],[1,0,0],[2,2,2]],dtype='f4')
    yield assert_equal, pf.track_roi_intersection_check(t,roi,1), True
    t=np.array([[1,1,0],[1,0,0],[1,-1,0]],dtype='f4')
    yield assert_equal, pf.track_roi_intersection_check(t,roi,1), True    
    t=np.array([[4,0,0],[4,1,1],[4,2,0]],dtype='f4')
    yield assert_equal, pf.track_roi_intersection_check(t,roi,1), False
    
    
    

