import sys
import pytest
import warnings

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from dipy.data import get_fnames
from dipy.io.streamline import load_tractogram
from dipy.segment.bundles import RecoBundles
from dipy.tracking.distances import bundles_distances_mam
from dipy.tracking.streamline import Streamlines
from dipy.segment.clustering import qbx_and_merge
from dipy.testing.decorators import set_random_number_generator

is_big_endian = 'big' in sys.byteorder.lower()


def setup_module():
    global f, f1, f2, f3, fornix

    fname = get_fnames('fornix')
    fornix = load_tractogram(fname, 'same',
                             bbox_valid_check=False).streamlines

    f = Streamlines(fornix)
    f1 = f.copy()

    f2 = f1[:20].copy()
    f2._data += np.array([50, 0, 0])

    f3 = f1[200:].copy()
    f3._data += np.array([100, 0, 0])

    f.extend(f2)
    f.extend(f3)


@pytest.mark.skipif(is_big_endian,
                    reason="Little Endian architecture required")
def test_rb_check_defaults():

    rb = RecoBundles(f, greater_than=0, clust_thr=10)

    rec_trans, rec_labels = rb.recognize(model_bundle=f2,
                                         model_clust_thr=5.,
                                         reduction_thr=10)

    msg = "Streamlines do not have the same number of points. *"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=msg, category=UserWarning)
        D = bundles_distances_mam(f2, f[rec_labels])

    # check if the bundle is recognized correctly
    if len(f2) == len(rec_labels):
        for row in D:
            assert_equal(row.min(), 0)

    refine_trans, refine_labels = rb.refine(model_bundle=f2,
                                            pruned_streamlines=rec_trans,
                                            model_clust_thr=5.,
                                            reduction_thr=10)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=msg, category=UserWarning)
        D = bundles_distances_mam(f2, f[refine_labels])

    # check if the bundle is recognized correctly
    for row in D:
        assert_equal(row.min(), 0)


@pytest.mark.skipif(is_big_endian,
                    reason="Little Endian architecture required")
def test_rb_disable_slr():

    rb = RecoBundles(f, greater_than=0, clust_thr=10)

    rec_trans, rec_labels = rb.recognize(model_bundle=f2,
                                         model_clust_thr=5.,
                                         reduction_thr=10,
                                         slr=False)

    msg = "Streamlines do not have the same number of points. *"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=msg, category=UserWarning)
        D = bundles_distances_mam(f2, f[rec_labels])

    # check if the bundle is recognized correctly
    if len(f2) == len(rec_labels):
        for row in D:
            assert_equal(row.min(), 0)

    refine_trans, refine_labels = rb.refine(model_bundle=f2,
                                            pruned_streamlines=rec_trans,
                                            model_clust_thr=5.,
                                            reduction_thr=10)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=msg, category=UserWarning)
        D = bundles_distances_mam(f2, f[refine_labels])

    # check if the bundle is recognized correctly
    for row in D:
        assert_equal(row.min(), 0)


@pytest.mark.skipif(is_big_endian,
                    reason="Little Endian architecture required")
@set_random_number_generator(42)
def test_rb_slr_threads(rng):

    rb_multi = RecoBundles(f, greater_than=0, clust_thr=10,
                           rng=rng)
    rec_trans_multi_threads, _ = rb_multi.recognize(model_bundle=f2,
                                                    model_clust_thr=5.,
                                                    reduction_thr=10,
                                                    slr=True,
                                                    num_threads=None)

    rb_single = RecoBundles(f, greater_than=0, clust_thr=10,
                            rng=np.random.default_rng(42))
    rec_trans_single_thread, _ = rb_single.recognize(model_bundle=f2,
                                                     model_clust_thr=5.,
                                                     reduction_thr=10,
                                                     slr=True,
                                                     num_threads=1)

    msg = "Streamlines do not have the same number of points. *"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=msg, category=UserWarning)
        D = bundles_distances_mam(rec_trans_multi_threads,
                                  rec_trans_single_thread)

    # check if the bundle is recognized correctly
    # multi-threading prevent an exact match
    for row in D:
        assert_almost_equal(row.min(), 0, decimal=4)


@pytest.mark.skipif(is_big_endian,
                    reason="Little Endian architecture required")
def test_rb_no_verbose_and_mam():

    rb = RecoBundles(f, greater_than=0, clust_thr=10, verbose=False)

    rec_trans, rec_labels = rb.recognize(model_bundle=f2,
                                         model_clust_thr=5.,
                                         reduction_thr=10,
                                         slr=True,
                                         pruning_distance='mam')

    msg = "Streamlines do not have the same number of points. *"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=msg, category=UserWarning)
        D = bundles_distances_mam(f2, f[rec_labels])

    # check if the bundle is recognized correctly
    if len(f2) == len(rec_labels):
        for row in D:
            assert_equal(row.min(), 0)

    refine_trans, refine_labels = rb.refine(model_bundle=f2,
                                            pruned_streamlines=rec_trans,
                                            model_clust_thr=5.,
                                            reduction_thr=10)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=msg, category=UserWarning)
        D = bundles_distances_mam(f2, f[refine_labels])

    # check if the bundle is recognized correctly
    for row in D:
        assert_equal(row.min(), 0)


@pytest.mark.skipif(is_big_endian,
                    reason="Little Endian architecture required")
def test_rb_clustermap():

    cluster_map = qbx_and_merge(f, thresholds=[40, 25, 20, 10])

    rb = RecoBundles(f, greater_than=0, less_than=1000000,
                     cluster_map=cluster_map, clust_thr=10)
    rec_trans, rec_labels = rb.recognize(model_bundle=f2,
                                         model_clust_thr=5.,
                                         reduction_thr=10)

    msg = "Streamlines do not have the same number of points. *"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=msg, category=UserWarning)
        D = bundles_distances_mam(f2, f[rec_labels])

    # check if the bundle is recognized correctly
    if len(f2) == len(rec_labels):
        for row in D:
            assert_equal(row.min(), 0)

    refine_trans, refine_labels = rb.refine(model_bundle=f2,
                                            pruned_streamlines=rec_trans,
                                            model_clust_thr=5.,
                                            reduction_thr=10)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=msg, category=UserWarning)
        D = bundles_distances_mam(f2, f[refine_labels])

    # check if the bundle is recognized correctly
    for row in D:
        assert_equal(row.min(), 0)


@pytest.mark.skipif(is_big_endian,
                    reason="Little Endian architecture required")
def test_rb_no_neighb():
    # what if no neighbors are found? No recognition

    b = Streamlines(fornix)
    b1 = b.copy()

    b2 = b1[:20].copy()
    b2._data += np.array([100, 0, 0])

    b3 = b1[:20].copy()
    b3._data += np.array([300, 0, 0])

    b.extend(b3)

    rb = RecoBundles(b, greater_than=0, clust_thr=10)

    rec_trans, rec_labels = rb.recognize(model_bundle=b2,
                                         model_clust_thr=5.,
                                         reduction_thr=10)

    if len(rec_trans) > 0:
        refine_trans, refine_labels = rb.refine(model_bundle=b2,
                                                pruned_streamlines=rec_trans,
                                                model_clust_thr=5.,
                                                reduction_thr=10)

        assert_equal(len(refine_labels), 0)
        assert_equal(len(refine_trans), 0)

    else:
        assert_equal(len(rec_labels), 0)
        assert_equal(len(rec_trans), 0)


@pytest.mark.skipif(is_big_endian,
                    reason="Little Endian architecture required")
def test_rb_reduction_mam():

    rb = RecoBundles(f, greater_than=0, clust_thr=10, verbose=True)

    rec_trans, rec_labels = rb.recognize(model_bundle=f2,
                                         model_clust_thr=5.,
                                         reduction_thr=10,
                                         reduction_distance='mam',
                                         slr=True,
                                         slr_metric='asymmetric',
                                         pruning_distance='mam')

    msg = "Streamlines do not have the same number of points. *"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=msg, category=UserWarning)
        D = bundles_distances_mam(f2, f[rec_labels])

    # check if the bundle is recognized correctly
    if len(f2) == len(rec_labels):
        for row in D:
            assert_equal(row.min(), 0)

    refine_trans, refine_labels = rb.refine(model_bundle=f2,
                                            pruned_streamlines=rec_trans,
                                            model_clust_thr=5.,
                                            reduction_thr=10)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=msg, category=UserWarning)
        D = bundles_distances_mam(f2, f[refine_labels])

    # check if the bundle is recognized correctly
    for row in D:
        assert_equal(row.min(), 0)
