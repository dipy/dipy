import numpy as np
from dipy.io.bvectxt import orientation_from_string
from dipy.tracking.utils import reorder_voxels_affine, move_streamlines
from numpy.testing import assert_array_almost_equal, assert_array_equal

def make_streamlines():
    streamlines = [ np.array([[0, 0, 0],
                              [1, 1, 1],
                              [2, 2, 2],
                              [5, 10, 12]], 'float'),
                    np.array([[1, 2, 3],
                              [3, 2, 0],
                              [5, 20, 33],
                              [40, 80, 120]], 'float') ]
    return streamlines

def test_move_streamlines():
    streamlines = make_streamlines()
    affine = np.eye(4)
    new_streamlines = move_streamlines(streamlines, affine)
    for i, test_sl in enumerate(new_streamlines):
        assert_array_equal(test_sl, streamlines[i])

    affine[:3,3] += (4,5,6)
    new_streamlines = move_streamlines(streamlines, affine)
    for i, test_sl in enumerate(new_streamlines):
        assert_array_equal(test_sl, streamlines[i]+(4, 5, 6))

    affine = np.eye(4)
    affine = affine[[2,1,0,3]]
    new_streamlines = move_streamlines(streamlines, affine)
    for i, test_sl in enumerate(new_streamlines):
        assert_array_equal(test_sl, streamlines[i][:, [2, 1, 0]])

def test_voxel_ornt():
    sh = (40,40,40)
    sz = (1, 2, 3)
    I4 = np.eye(4)

    ras = orientation_from_string('ras')
    sra = orientation_from_string('sra')
    lpi = orientation_from_string('lpi')
    srp = orientation_from_string('srp')

    affine = reorder_voxels_affine(ras, ras, sh, sz)
    assert_array_equal(affine, I4)
    affine = reorder_voxels_affine(sra, sra, sh, sz)
    assert_array_equal(affine, I4)
    affine = reorder_voxels_affine(lpi, lpi, sh, sz)
    assert_array_equal(affine, I4)
    affine = reorder_voxels_affine(srp, srp, sh, sz)
    assert_array_equal(affine, I4)

    streamlines = make_streamlines()
    box = np.array(sh)*sz

    sra_affine = reorder_voxels_affine(ras, sra, sh, sz)
    toras_affine = reorder_voxels_affine(sra, ras, sh, sz)
    assert_array_equal(np.dot(toras_affine, sra_affine), I4)
    expected_sl = (sl[:, [2, 0, 1]] for sl in streamlines)
    test_sl = move_streamlines(streamlines, sra_affine)
    for ii in xrange(len(streamlines)):
        assert_array_equal(test_sl.next(), expected_sl.next())

    lpi_affine = reorder_voxels_affine(ras, lpi, sh, sz)
    toras_affine = reorder_voxels_affine(lpi, ras, sh, sz)
    assert_array_equal(np.dot(toras_affine, lpi_affine), I4)
    expected_sl = (box - sl for sl in streamlines)
    test_sl = move_streamlines(streamlines, lpi_affine)
    for ii in xrange(len(streamlines)):
        assert_array_equal(test_sl.next(), expected_sl.next())

    srp_affine = reorder_voxels_affine(ras, srp, sh, sz)
    toras_affine = reorder_voxels_affine(srp, ras, (40,40,40), (3,1,2))
    assert_array_equal(np.dot(toras_affine, srp_affine), I4)
    expected_sl = [sl.copy() for sl in streamlines]
    for sl in expected_sl:
        sl[:, 1] = box[1] - sl[:, 1]
    expected_sl = (sl[:, [2, 0, 1]] for sl in expected_sl)
    test_sl = move_streamlines(streamlines, srp_affine)
    for ii in xrange(len(streamlines)):
        assert_array_equal(test_sl.next(), expected_sl.next())

