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

    #show_streamlines(sbundles[3])

if __name__ == '__main__':
    run_module_suite()
