import numpy as np

from dipy.data import get_fnames
from dipy.io.streamline import load_tractogram
from dipy.segment.fss import (FastStreamlineSearch,
                              nearest_from_matrix_row,
                              nearest_from_matrix_col)
from dipy.segment.metric import mean_euclidean_distance

from dipy.testing import (assert_arrays_equal,
                          assert_greater,
                          assert_greater_equal,
                          assert_true)
from numpy.testing import assert_almost_equal, assert_raises


def setup_module():
    global f1, f2
    fname = get_fnames('fornix')
    fornix = load_tractogram(fname, 'same', bbox_valid_check=False)

    # Should work with both StatefulTractogram and streamlines (list of array)
    f1 = fornix[:200]
    f2 = fornix.streamlines[200:]


def test_fss_radius_search():
    r = 4.0
    nb_pts = 24
    # For each "f1" streamlines search all in radius of "f2"
    fss_f1 = FastStreamlineSearch(f1, max_radius=r, nb_mpts=2, bin_size=20.0,
                                  resampling=nb_pts, bidirectional=True)
    rs_f2_in_f1 = fss_f1.radius_search(f2, radius=r, use_negative=True)

    # For each "f2" streamlines search all in radius of "f1"
    fss_f2 = FastStreamlineSearch(f2, max_radius=r, nb_mpts=8, bin_size=10.0,
                                  resampling=nb_pts, bidirectional=True)
    rs_f1_in_f2 = fss_f2.radius_search(f1, radius=r, use_negative=False)

    # Verify there is results
    assert_greater(rs_f2_in_f1.nnz, 0)
    assert_greater(rs_f1_in_f2.nnz, 0)

    # Verify The number of results are the same
    assert_true(rs_f2_in_f1.nnz == rs_f1_in_f2.nnz)

    # Verify that both search results are equivalent (transposed)
    assert_arrays_equal(np.sort(rs_f2_in_f1.row),
                        np.sort(rs_f1_in_f2.col))
    assert_arrays_equal(np.sort(rs_f2_in_f1.col),
                        np.sort(rs_f1_in_f2.row))

    # Verify if resulting distances are the same
    assert_almost_equal(np.sort(np.abs(rs_f2_in_f1.data)),
                        np.sort(rs_f1_in_f2.data))

    # Verify that minimum are the same from f1 to f2
    r1_a, r1_b, r1_d = nearest_from_matrix_row(rs_f2_in_f1)
    r2_a, r2_b, r2_d = nearest_from_matrix_col(rs_f1_in_f2)
    assert_arrays_equal(r1_a, r2_a)
    assert_arrays_equal(r1_b, r2_b)
    assert_arrays_equal(r1_d, r2_d)

    # Verify that minimum are the same from f2 to f1
    r3_a, r3_b, r3_d = nearest_from_matrix_col(rs_f2_in_f1)
    r4_a, r4_b, r4_d = nearest_from_matrix_row(rs_f1_in_f2)
    assert_arrays_equal(r3_a, r4_a)
    assert_arrays_equal(r3_b, r4_b)
    assert_almost_equal(r3_d, r4_d)

    # Test with unidirectional search (bidirectional=False)
    fss_sd = FastStreamlineSearch(f1, max_radius=r, nb_mpts=6, bin_size=80.0,
                                  resampling=nb_pts, bidirectional=False)
    rs_f1_sd = fss_sd.radius_search(f2, radius=r, use_negative=True)

    # Single direction should be a subset of bidirectional
    assert_greater_equal(rs_f1_in_f2.nnz, rs_f1_sd.nnz)
    assert_true(np.all(np.in1d(rs_f1_sd.row, rs_f2_in_f1.row)))
    assert_true(np.all(np.in1d(rs_f1_sd.col, rs_f2_in_f1.col)))


def test_fss_varying_radius():
    # For each "f1" streamlines search all in radius of "f2"
    fss = FastStreamlineSearch(f1, max_radius=10.0, nb_mpts=5, bin_size=20.0,
                               resampling=25, bidirectional=True)
    rs_6 = fss.radius_search(f2, radius=6.0, use_negative=True)
    rs_4 = fss.radius_search(f2, radius=4.0, use_negative=True)
    rs_2 = fss.radius_search(f2, radius=2.0, use_negative=True)

    # smaller radius should be a subset or equal of a bigger radius
    assert_greater_equal(rs_6.nnz, rs_4.nnz)
    assert_true(np.all(np.in1d(rs_4.row, rs_6.row)))
    assert_true(np.all(np.in1d(rs_4.col, rs_6.col)))

    assert_greater_equal(rs_4.nnz, rs_2.nnz)
    assert_true(np.all(np.in1d(rs_2.row, rs_4.row)))
    assert_true(np.all(np.in1d(rs_2.col, rs_4.col)))


def test_fss_single_point_slines():
    slines = [np.array([[1.0, 1.0, 1.0]]),
              np.array([[0.0, 1.0, 2.0]])]
    fss = FastStreamlineSearch(slines, max_radius=4.0, nb_mpts=4, bin_size=20.0,
                               resampling=24, bidirectional=False)
    res = fss.radius_search(slines, radius=4.0)
    # 2x2 matrix with 4 element
    assert_true(res.nnz == 4)
    mat = res.A
    dist = mean_euclidean_distance(slines[0], slines[1])
    assert_almost_equal(mat[0, 0], 0.0)
    assert_almost_equal(mat[1, 1], 0.0)
    assert_almost_equal(mat[1, 0], dist)
    assert_almost_equal(mat[0, 1], dist)


def test_fss_empty_results():
    fss = FastStreamlineSearch(f1, max_radius=2.0, nb_mpts=2, bin_size=20.0, resampling=22, bidirectional=True)
    res = fss.radius_search(f2, radius=0.01, use_negative=True)
    assert_true(res.nnz == 0)


def test_fss_invalid_max_radius():
    assert_raises(ValueError, FastStreamlineSearch, f1, max_radius=0.0)
    assert_raises(ValueError, FastStreamlineSearch, f1, max_radius=-1.0)


def test_fss_invalid_radius():
    fss = FastStreamlineSearch(f1, max_radius=2.0)
    assert_raises(ValueError, fss.radius_search, f2, radius=5.0)


def test_fss_invalid_mpts():
    assert_raises(ValueError, FastStreamlineSearch, f1,  max_radius=4.0,
                  nb_mpts=4, resampling=23)
    assert_raises(ZeroDivisionError, FastStreamlineSearch, f1,  max_radius=4.0,
                  nb_mpts=0, resampling=24)
