import numpy as np
from numpy.testing import (assert_equal, assert_array_almost_equal,
                           assert_raises)

from dipy.align.streamlinear import (compose_matrix44, decompose_matrix44,
                                     transform_streamlines, whole_brain_slr,
                                     slr_with_qbx)
from dipy.io.streamline import load_tractogram
from dipy.data import get_fnames
from dipy.tracking.streamline import Streamlines
from dipy.tracking.distances import bundles_distances_mam


def test_whole_brain_slr():
    fname = get_fnames('fornix')

    fornix = load_tractogram(fname, 'same',
                             bbox_valid_check=False).streamlines

    f = Streamlines(fornix)
    f1 = f.copy()
    f2 = f.copy()

    # check translation
    f2._data += np.array([50, 0, 0])

    moved, transform, qb_centroids1, qb_centroids2 = whole_brain_slr(
            f1, f2, x0='affine', verbose=True, rm_small_clusters=2,
            greater_than=0, less_than=np.inf,
            qbx_thr=[5, 2, 1], progressive=False)

    # we can check the quality of registration by comparing the matrices
    # MAM streamline distances before and after SLR
    D12 = bundles_distances_mam(f1, f2)
    D1M = bundles_distances_mam(f1, moved)

    d12_minsum = np.sum(np.min(D12, axis=0))
    d1m_minsum = np.sum(np.min(D1M, axis=0))

    print("distances= ", d12_minsum, " ", d1m_minsum)

    assert_equal(d1m_minsum < d12_minsum, True)

    assert_array_almost_equal(transform[:3, 3], [-50, -0, -0], 2)

    # check rotation

    mat = compose_matrix44([0, 0, 0, 15, 0, 0])

    f3 = f.copy()
    f3 = transform_streamlines(f3, mat)

    moved, transform, qb_centroids1, qb_centroids2 = slr_with_qbx(
            f1, f3, verbose=False, rm_small_clusters=1, greater_than=20,
            less_than=np.inf, qbx_thr=[2],
            progressive=True)

    # we can also check the quality by looking at the decomposed transform

    assert_array_almost_equal(decompose_matrix44(transform)[3], -15, 2)

    moved, transform, qb_centroids1, qb_centroids2 = slr_with_qbx(
            f1, f3, verbose=False, rm_small_clusters=1, select_random=400,
            greater_than=20, less_than=np.inf, qbx_thr=[2],
            progressive=True)

    # we can also check the quality by looking at the decomposed transform

    assert_array_almost_equal(decompose_matrix44(transform)[3], -15, 2)


def test_slr_one_streamline():
    fname = get_fnames('fornix')

    fornix = load_tractogram(fname, 'same',
                             bbox_valid_check=False).streamlines

    f = Streamlines(fornix)
    f1_one = Streamlines([f[0]])
    f2 = f.copy()
    f2._data += np.array([50, 0, 0])

    assert_raises(ValueError, slr_with_qbx, f1_one, f2, verbose=False,
                  rm_small_clusters=50, greater_than=20,
                  less_than=np.inf, qbx_thr=[2], progressive=True)

    assert_raises(ValueError, slr_with_qbx, f2, f1_one, verbose=False,
                  rm_small_clusters=50, greater_than=20,
                  less_than=np.inf, qbx_thr=[2], progressive=True)
