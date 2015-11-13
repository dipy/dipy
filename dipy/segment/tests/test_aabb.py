import numpy as np
import numpy.testing as npt
from dipy.segment.clustering import QuickBundles
from dipy.segment.clusteringspeed import evaluate_aabbb_checks
from dipy.data import get_data
import nibabel as nib
from dipy.tracking.streamline import set_number_of_points


def test_aabb_checks():
    A, B, res = evaluate_aabbb_checks()
    npt.assert_equal(res, 1)


def show_streamlines(streamlines=None, centroids=None):

    from dipy.viz import actor, window

    ren = window.Renderer()

    if streamlines is not None:
        stream_actor = actor.line(streamlines)
        ren.add(stream_actor)
        window.show(ren)
        ren.clear()

    if centroids is not None:
        stream_actor2 = actor.line(centroids)
        ren.add(stream_actor2)
        window.show(ren)


def test_qbundles_aabb():
    streams, hdr = nib.trackvis.read(get_data('fornix'))
    streamlines = [s[0] for s in streams]

    for i in range(100):

        streamlines += [s[0] + np.array([i * 70, 0, 0]) for s in streams]


    from dipy.tracking.streamline import select_random_set_of_streamlines
    streamlines = select_random_set_of_streamlines(streamlines,
                                                   len(streamlines))

    print(len(streamlines))

    rstreamlines = set_number_of_points(streamlines, 20)

    from time import time

    qb = QuickBundles(2.5, bvh=False)
    t = time()
    clusters = qb.cluster(rstreamlines)
    print('Without BVH {}'.format(time() - t))
    print(len(clusters))

    show_streamlines(rstreamlines, clusters.centroids)

    qb = QuickBundles(2.5, bvh=True)
    t = time()
    clusters = qb.cluster(rstreamlines)
    print('With BVH {}'.format(time() - t))
    print(len(clusters))

    show_streamlines(rstreamlines, clusters.centroids)

    #from ipdb import set_trace
    #set_trace()


#test_qbundles_aabb()

def test_qbundles_full_brain():

    fname = '/home/eleftherios/Data/Test_data_Jasmeen/Elef_Test_RecoBundles/tracts.trk'

    #streams, hdr = nib.trackvis.read(fname)
    obj = nib.streamlines.load(fname)
    streamlines = obj.streamlines

    from dipy.tracking.streamline import select_random_set_of_streamlines
    streamlines = select_random_set_of_streamlines(streamlines,
                                                   len(streamlines))

    print(len(streamlines))

    rstreamlines = set_number_of_points(streamlines, 20)

    del streamlines

    from time import time
    from dipy.segment.metric import AveragePointwiseEuclideanMetric

    threshold = 15

#    qb = QuickBundles(threshold, metric=AveragePointwiseEuclideanMetric(), bvh=False)
#    t = time()
#    clusters1 = qb.cluster(rstreamlines)
#    print('Without BVH {}'.format(time() - t))
#    print(len(clusters1))

    #show_streamlines(None, clusters1.centroids)

    qb = QuickBundles(threshold, metric=AveragePointwiseEuclideanMetric(), bvh=True)
    t = time()
    clusters2 = qb.cluster(rstreamlines)
    print('With BVH {}'.format(time() - t))
    print(len(clusters2))

    show_streamlines(None, clusters2.centroids)

    from ipdb import set_trace
    set_trace()


# 30
# 329 vs 210 1.5X
# 20
# 2006s (1110 clusters) vs 1218s (1103 clusters) 1.6X
# 15
# 8669 (4274 clusters) vs 4657 (4274 clusters) 1.86X
# 15 but with 1/2 padding
# 8669 (4274 clusters) vs 3842 (4314 clusters) 2.2X
# 15 but with No padding

test_qbundles_full_brain()


