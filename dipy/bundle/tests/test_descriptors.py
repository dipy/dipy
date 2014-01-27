import numpy as np
from numpy.testing import (assert_equal,                           
                           run_module_suite)
from dipy.data import get_data
from nibabel import trackvis as tv
from dipy.bundle.descriptors import (length_distribution,
                                     avg_streamline,
                                     qb_centroids,
                                     winding_angles,
                                     midpoints,
                                     centers_of_mass,
                                     dragons_hits)
from dipy.bundle.descriptors import show_streamlines
from dipy.viz import fvtk


def fornix_streamlines():
    fname = get_data('fornix')
    streams, hdr = tv.read(fname)
    streamlines = [i[0] for i in streams]
    return streamlines


def test_descriptors_fornix():

    streamlines = fornix_streamlines()

    lengths = length_distribution(streamlines)

    assert_equal(lengths.max() < 100, True)
    assert_equal(lengths.min() > 10, True)

    avg = avg_streamline(streamlines)

    avg_length = length_distribution(avg)

    assert_equal(avg_length < lengths.max(), True)
    assert_equal(avg_length > lengths.min(), True)

    centroids = qb_centroids(streamlines, 10)

    assert_equal(len(centroids), 4)

    winds = winding_angles(centroids)

    assert_equal(np.mean(winds) < 300 and np.mean(winds) > 100, True)
    
    mpoints = midpoints(centroids)

    assert_equal(len(mpoints), 4)

    cpoints = centers_of_mass(centroids)

    assert_equal(len(cpoints), 4)

    hpoints, hangles = dragons_hits(centroids, avg)

    assert_equal(len(hpoints) > len(avg), True)

    ren = show_streamlines(centroids, mpoints)

    fvtk.add(ren, fvtk.point(hpoints, fvtk.colors.red))

    fvtk.add(ren, fvtk.line(avg, fvtk.colors.tomato))

    fvtk.add(ren, fvtk.point(avg, fvtk.colors.yellow))

    fvtk.show(ren)


def simulated_bundles(no_pts=200):
    t = np.linspace(-10, 10, no_pts)
    fibno = 150

    # helix
    bundle = []
    for i in np.linspace(3, 5, fibno):
        pts = 5 * np.vstack((np.cos(t), np.sin(t), t / i)).T  # helix diverging
        bundle.append(pts)

    # parallel waves
    bundle2 = []
    for i in np.linspace(-5, 5, fibno):        
        pts = np.vstack((np.cos(t), t, i * np.ones(t.shape))).T
        bundle2.append(pts)

    # spider - diverging in the ends
    bundle3 = []
    for i in np.linspace(-1, 1, fibno):        
        pts = np.vstack((i ** 3 * t / 2., t, np.cos(t))).T
        bundle3.append(pts)
    
    # diverging in the middle
    bundle4 = []
    for i in np.linspace(-1, 1, fibno):
        pts = 2 * \
            np.vstack((0 * t + 2 * i * np.cos(.2 * t), np.cos(.4 * t), t)).T
        bundle4.append(pts)

    return [bundle, bundle2, bundle3, bundle4]


def test_descriptors_sim_bundles():

    sbundles = simulated_bundles()

    helix, parallel, spider, centerdiv = sbundles

    show_streamlines(sbundles[3])


def parametrize_arclength(streamline):
    n_vertex = len(streamline)
    disp = np.diff(streamline, axis=0)
    L2 = np.sqrt(np.sum(disp ** 2, axis=1))

    arc_length = np.sum(L2)
    cum_len = np.cumsum(L2) / float(arc_length)
    para = np.zeros(n_vertex)
    para[1:] = cum_len
    return para


def cosine_series(streamline, para, k=10):
    n_vertex = len(para)
    para_even = [-para[::-1][:-1], para]
    stream_even = [streamline[::-1][:-1], streamline]
    # Y=zeros(2*n_vertex-1,k+1);
    # para_even=repmat(para_even',1,k+1);
    # pi_factors=repmat([0:k],2*n_vertex-1,1).*pi;
    # Y=cos(para_even.*pi_factors).*sqrt(2);

    # beta=pinv(Y'*Y)*Y'*tract_even';

    # hat= Y*beta;

    # wfs=hat(n_vertex:(n_vertex*2-1),:)';


def test_cosine_series():

    helix = simulated_bundles(10)[0]
    streamline = helix[0]
    para = parametrize_arclength(streamline)
    print(para)
    para = parametrize_arclength(streamline * np.array([[1.2, 1., 1.]]))
    print(para)
    

    1/0


if __name__ == '__main__':
    # run_module_suite()
    # test_descriptors_fornix()
    test_cosine_series()
