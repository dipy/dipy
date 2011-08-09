import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_raises
from dipy.io.bvectxt import orientation_from_string, reorient_bvec

def test_orientation_from_string():
    ras = np.array(((0,1), (1,1), (2,1)))
    lps = np.array(((0,-1), (1,-1), (2,1)))
    asl = np.array(((1,1), (2,1), (0,-1)))
    assert_array_equal(orientation_from_string('ras'), ras)
    assert_array_equal(orientation_from_string('lps'), lps)
    assert_array_equal(orientation_from_string('asl'), asl)
    assert_raises(ValueError, orientation_from_string, 'aasl')

def test_reorient_bvec():
    bvec = np.arange(12).reshape((3,4))
    assert_array_equal(reorient_bvec(bvec, 'ras', 'ras'), bvec)
    assert_array_equal(reorient_bvec(bvec, 'ras', 'lpi'), -bvec)
    result = bvec[[1,2,0]]
    assert_array_equal(reorient_bvec(bvec, 'ras', 'asr'), result)
    bvec = result
    result = bvec[[1,0,2]]*[[-1],[1],[-1]]
    assert_array_equal(reorient_bvec(bvec, 'asr', 'ial'), result)
    assert_raises(ValueError, reorient_bvec, bvec, 'ras', 'ra')

