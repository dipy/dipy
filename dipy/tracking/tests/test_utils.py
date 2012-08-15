import numpy as np
import nose
from dipy.io.bvectxt import orientation_from_string
from dipy.tracking.utils import (_rmi, connectivity_matrix, density_map,
                                 move_streamlines, ndbincount, reduce_labels,
                                 reorder_voxels_affine, streamline_mapping)
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_equal, assert_raises, assert_true

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

def test_density_map():
    #One streamline diagonal in volume
    streamlines = [np.array([np.arange(10)]*3).T]
    shape = (10, 10, 10)
    x = np.arange(10)
    expected = np.zeros(shape)
    expected[x, x, x] = 1.
    dm = density_map(streamlines, vol_dims=shape, voxel_size=(1, 1, 1))
    assert_array_equal(dm, expected)

    #add streamline, make voxel_size smaller. Each streamline should only be
    #counted once, even if multiple points lie in a voxel
    streamlines.append(np.ones((5, 3)))
    shape = (5, 5, 5)
    x = np.arange(5)
    expected = np.zeros(shape)
    expected[x, x, x] = 1.
    expected[0, 0, 0] += 1
    dm = density_map(streamlines, vol_dims=shape, voxel_size=(2, 2, 2))
    assert_array_equal(dm, expected)
    #should work with a generator
    streamlines = iter(streamlines)
    dm = density_map(streamlines, vol_dims=shape, voxel_size=(2, 2, 2))
    assert_array_equal(dm, expected)

def test_connectivity_matrix():
    label_volume = np.array([[[3, 0, 0],
                              [0, 0, 0],
                              [0, 0, 4]]])
    streamlines = [np.array([[0,0,0],[0,0,0],[0,2,2]], 'float'),
                   np.array([[0,0,0],[0,1,1],[0,2,2]], 'float'),
                   np.array([[0,2,2],[0,1,1],[0,0,0]], 'float')]
    expected = np.zeros((5, 5), 'int')
    expected[3, 4] = 2
    expected[4, 3] = 1
    # Check basic Case
    matrix = connectivity_matrix(streamlines, label_volume, (1, 1, 1))
    assert_array_equal(matrix, expected)
    # Test mapping
    matrix, mapping = connectivity_matrix(streamlines, label_volume, (1, 1, 1),
                                          return_mapping=True)
    assert_array_equal(matrix, expected)
    assert_equal(mapping[3, 4], [0, 1])
    assert_equal(mapping[4, 3], [2])
    assert_raises(KeyError, mapping.__getitem__, (0, 0))
    # Test mapping and symmetric
    matrix, mapping = connectivity_matrix(streamlines, label_volume, (1, 1, 1),
                                          True, return_mapping=True)
    assert_equal(mapping[3, 4], [0, 1, 2])
    # When symmetric only (3,4) is a key, not (4, 3)
    assert_raises(KeyError, mapping.__getitem__, (4, 3))
    # expected output matrix is symmetric version of expected
    expected = expected + expected.T
    assert_array_equal(matrix, expected)
    # Test mapping_as_streamlines, mapping dict has lists of streamlines
    matrix, mapping = connectivity_matrix(streamlines, label_volume, (1, 1, 1),
                                          return_mapping=True,
                                          mapping_as_streamlines=True)
    assert_true(mapping[3, 4][0] is streamlines[0])
    assert_true(mapping[3, 4][1] is streamlines[1])
    assert_true(mapping[4, 3][0] is streamlines[2])

def test_ndbincount():
    def check(expected):
        assert_equal(bc[0, 0], expected[0])
        assert_equal(bc[0, 1], expected[1])
        assert_equal(bc[1, 0], expected[2])
        assert_equal(bc[2, 2], expected[3])
    x = np.array([[0, 0], [0, 0], [0, 1], [0, 1], [1, 0], [2, 2]]).T
    expected = [2, 2, 1, 1]
    #count occurrences in x
    bc = ndbincount(x)
    assert_equal(bc.shape, (3, 3))
    check(expected)
    #pass in shape
    bc = ndbincount(x, shape=(4, 5))
    assert_equal(bc.shape, (4, 5))
    check(expected)
    #pass in weights
    weights = np.arange(6.)
    weights[-1] = 1.23
    expeceted = [1., 5., 4., 1.23]
    bc = ndbincount(x, weights=weights)
    check(expeceted)
    #raises an error if shape is too small
    assert_raises(ValueError, ndbincount, x, None, (2, 2))

def test_reduce_labels():
    shape = (4, 5, 6)
    #labels from 100 to 220
    labels = np.arange(100, np.prod(shape)+100).reshape(shape)
    #new labels form 0 to 120, and lookup maps range(0,120) to range(100, 220)
    new_labels, lookup = reduce_labels(labels)
    assert_array_equal(new_labels, labels-100)
    assert_array_equal(lookup, labels.ravel())

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

def test_streamline_mapping():
    streamlines = [np.array([[0,0,0],[0,0,0],[0,2,2]], 'float'),
                   np.array([[0,0,0],[0,1,1],[0,2,2]], 'float'),
                   np.array([[0,2,2],[0,1,1],[0,0,0]], 'float')]
    mapping = streamline_mapping(streamlines, (1,1,1))
    expected = {(0,0,0):[0,1,2], (0,2,2):[0,1,2], (0,1,1):[1,2]}
    assert_equal(mapping, expected)

    mapping = streamline_mapping(streamlines, (1,1,1), True)
    expected = dict((k, [streamlines[i] for i in indices])
                    for k, indices in expected.iteritems())
    assert_equal(mapping, expected)

def test_rmi():

    I1 = _rmi([3, 4], [10, 10])
    assert_equal(I1, 34)
    I1 = _rmi([0, 0], [10, 10])
    assert_equal(I1, 0)
    assert_raises(ValueError, _rmi, [10, 0], [10, 10])

    try:
        from numpy import ravel_multi_index
    except ImportError:
        raise nose.SkipTest()

    A, B, C, D = np.random.randint(0, 1000, size=[4, 100])

    I1 = _rmi([A, B], dims=[1000, 1000])
    I2 = ravel_multi_index([A, B], dims=[1000, 1000])
    assert_array_equal(I1, I2)

    I1 = _rmi([A, B, C, D], dims=[1000]*4)
    I2 = ravel_multi_index([A, B, C, D], dims=[1000]*4)
    assert_array_equal(I1, I2)

