import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from dipy.io.bvectxt import orientation_from_string, reorient_vectors, \
    orientation_to_string, reorient_vectors


def test_orientation_from_to_string():
    ras = np.array(((0, 1), (1, 1), (2, 1)))
    lps = np.array(((0, -1), (1, -1), (2, 1)))
    asl = np.array(((1, 1), (2, 1), (0, -1)))
    assert_array_equal(orientation_from_string('ras'), ras)
    assert_array_equal(orientation_from_string('lps'), lps)
    assert_array_equal(orientation_from_string('asl'), asl)
    assert_raises(ValueError, orientation_from_string, 'aasl')

    assert orientation_to_string(ras) == 'ras'
    assert orientation_to_string(lps) == 'lps'
    assert orientation_to_string(asl) == 'asl'


def test_reorient_vectors():
    bvec = np.arange(12).reshape((3, 4))
    assert_array_equal(reorient_vectors(bvec, 'ras', 'ras'), bvec)
    assert_array_equal(reorient_vectors(bvec, 'ras', 'lpi'), -bvec)
    result = bvec[[1, 2, 0]]
    assert_array_equal(reorient_vectors(bvec, 'ras', 'asr'), result)
    bvec = result
    result = bvec[[1, 0, 2]] * [[-1], [1], [-1]]
    assert_array_equal(reorient_vectors(bvec, 'asr', 'ial'), result)
    result = bvec[[1, 0, 2]] * [[-1], [1], [1]]
    assert_array_equal(reorient_vectors(bvec, 'asr', 'iar'), result)
    assert_raises(ValueError, reorient_vectors, bvec, 'ras', 'ra')

    bvec = np.arange(12).reshape((3, 4))
    bvec = bvec.T
    assert_array_equal(reorient_vectors(bvec, 'ras', 'ras', axis=1), bvec)
    assert_array_equal(reorient_vectors(bvec, 'ras', 'lpi', axis=1), -bvec)
    result = bvec[:, [1, 2, 0]]
    assert_array_equal(reorient_vectors(bvec, 'ras', 'asr', axis=1), result)
    bvec = result
    result = bvec[:, [1, 0, 2]] * [-1, 1, -1]
    assert_array_equal(reorient_vectors(bvec, 'asr', 'ial', axis=1), result)
    result = bvec[:, [1, 0, 2]] * [-1, 1, 1]
    assert_array_equal(reorient_vectors(bvec, 'asr', 'iar', axis=1), result)
