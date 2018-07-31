import numpy as np
import nibabel as nib
from numpy.testing import assert_equal, run_module_suite
from dipy.data import get_data
from dipy.segment.bundles import RecoBundles
from dipy.tracking.distances import bundles_distances_mam
from dipy.tracking.streamline import Streamlines
from dipy.segment.clustering import qbx_and_merge


streams, hdr = nib.trackvis.read(get_data('fornix'))
fornix = [s[0] for s in streams]

f = Streamlines(fornix)
f1 = f.copy()

f2 = f1[:20].copy()
f2._data += np.array([50, 0, 0])

f3 = f1[200:].copy()
f3._data += np.array([100, 0, 0])

f.extend(f2)
f.extend(f3)


def test_rb_check_defaults():

    rb = RecoBundles(f, clust_thr=10)
    rec_trans, rec_labels, recognized = rb.recognize(model_bundle=f2,
                                                     model_clust_thr=5.,
                                                     reduction_thr=10)
    D = bundles_distances_mam(f2, recognized)

    # check if the bundle is recognized correctly
    for row in D:
        assert_equal(row.min(), 0)


def test_rb_disable_slr():

    rb = RecoBundles(f, clust_thr=10)

    rec_trans, rec_labels, recognized = rb.recognize(model_bundle=f2,
                                                     model_clust_thr=5.,
                                                     reduction_thr=10,
                                                     slr=False)

    D = bundles_distances_mam(f2, recognized)

    # check if the bundle is recognized correctly
    for row in D:
        assert_equal(row.min(), 0)


def test_rb_no_verbose_and_mam():

    rb = RecoBundles(f, clust_thr=10, verbose=False)

    rec_trans, rec_labels, recognized = rb.recognize(model_bundle=f2,
                                                     model_clust_thr=5.,
                                                     reduction_thr=10,
                                                     slr=True,
                                                     pruning_distance='mam')

    D = bundles_distances_mam(f2, recognized)

    # check if the bundle is recognized correctly
    for row in D:
        assert_equal(row.min(), 0)


def test_rb_clustermap():

    cluster_map = qbx_and_merge(f, thresholds=[40, 25, 20, 10])

    rb = RecoBundles(f, cluster_map=cluster_map, clust_thr=10)
    rec_trans, rec_labels, recognized = rb.recognize(model_bundle=f2,
                                                     model_clust_thr=5.,
                                                     reduction_thr=10)
    D = bundles_distances_mam(f2, recognized)

    # check if the bundle is recognized correctly
    for row in D:
        assert_equal(row.min(), 0)


def test_rb_no_neighb():
    # what if no neighbors are found? No recognition

    b = Streamlines(fornix)
    b1 = b.copy()

    b2 = b1[:20].copy()
    b2._data += np.array([100, 0, 0])

    b3 = b1[:20].copy()
    b3._data += np.array([300, 0, 0])

    b.extend(b3)

    rb = RecoBundles(b, clust_thr=10)
    rec_trans, rec_labels, recognized = rb.recognize(model_bundle=b2,
                                                     model_clust_thr=5.,
                                                     reduction_thr=10)

    assert_equal(len(recognized), 0)
    assert_equal(len(rec_labels), 0)
    assert_equal(len(rec_trans), 0)


def test_rb_reduction_mam():

    rb = RecoBundles(f, clust_thr=10, verbose=True)

    rec_trans, rec_labels, recognized = rb.recognize(model_bundle=f2,
                                                     model_clust_thr=5.,
                                                     reduction_thr=10,
                                                     reduction_distance='mam',
                                                     slr=True,
                                                     slr_metric='asymmetric',
                                                     pruning_distance='mam')

    D = bundles_distances_mam(f2, recognized)

    # check if the bundle is recognized correctly
    for row in D:
        assert_equal(row.min(), 0)


if __name__ == '__main__':

    run_module_suite()
