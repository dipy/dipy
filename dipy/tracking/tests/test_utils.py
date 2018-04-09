from __future__ import division, print_function, absolute_import

from dipy.utils.six.moves import xrange

import numpy as np
import nose

from dipy.io.bvectxt import orientation_from_string
from dipy.tracking.utils import (affine_for_trackvis, connectivity_matrix,
                                 density_map, length, move_streamlines,
                                 ndbincount, reduce_labels,
                                 reorder_voxels_affine, seeds_from_mask,
                                 random_seeds_from_mask, target,
                                 target_line_based, unique_rows, near_roi,
                                 reduce_rois, path_length, flexi_tvis_affine,
                                 get_flexi_tvis_affine, _min_at)

from dipy.tracking._utils import _to_voxel_coordinates

import dipy.tracking.metrics as metrix

from dipy.tracking.vox2track import streamline_mapping
import numpy.testing as npt
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_equal, assert_raises, assert_true


def make_streamlines():
    streamlines = [np.array([[0, 0, 0],
                             [1, 1, 1],
                             [2, 2, 2],
                             [5, 10, 12]], 'float'),
                   np.array([[1, 2, 3],
                             [3, 2, 0],
                             [5, 20, 33],
                             [40, 80, 120]], 'float')]
    return streamlines


def test_density_map():
    # One streamline diagonal in volume
    streamlines = [np.array([np.arange(10)] * 3).T]
    shape = (10, 10, 10)
    x = np.arange(10)
    expected = np.zeros(shape)
    expected[x, x, x] = 1.
    dm = density_map(streamlines, vol_dims=shape, voxel_size=(1, 1, 1))
    assert_array_equal(dm, expected)

    # add streamline, make voxel_size smaller. Each streamline should only be
    # counted once, even if multiple points lie in a voxel
    streamlines.append(np.ones((5, 3)))
    shape = (5, 5, 5)
    x = np.arange(5)
    expected = np.zeros(shape)
    expected[x, x, x] = 1.
    expected[0, 0, 0] += 1
    dm = density_map(streamlines, vol_dims=shape, voxel_size=(2, 2, 2))
    assert_array_equal(dm, expected)
    # should work with a generator
    dm = density_map(iter(streamlines), vol_dims=shape, voxel_size=(2, 2, 2))
    assert_array_equal(dm, expected)

    # Test passing affine
    affine = np.diag([2, 2, 2, 1.])
    affine[:3, 3] = 1.
    dm = density_map(streamlines, shape, affine=affine)
    assert_array_equal(dm, expected)

    # Shift the image by 2 voxels, ie 4mm
    affine[:3, 3] -= 4.
    expected_old = expected
    new_shape = [i + 2 for i in shape]
    expected = np.zeros(new_shape)
    expected[2:, 2:, 2:] = expected_old
    dm = density_map(streamlines, new_shape, affine=affine)
    assert_array_equal(dm, expected)


def test_to_voxel_coordinates_precision():
    # To simplify tests, use an identity affine. This would be the result of
    # a call to _mapping_to_voxel with another identity affine.
    transfo = np.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]])

    # Offset is computed by _mapping_to_voxel. With a 1x1x1 dataset
    # having no translation, the offset is half the voxel size, i.e. 0.5.
    offset = np.array([0.5, 0.5, 0.5])

    # Without the added tolerance in _to_voxel_coordinates, this streamline
    # should raise an Error in the call to _to_voxel_coordinates.
    failing_strl = [np.array([[-0.5000001, 0.0, 0.0], [0.0, 1.0, 0.0]],
                             dtype=np.float32)]

    indices = _to_voxel_coordinates(failing_strl, transfo, offset)

    expected_indices = np.array([[[0, 0, 0], [0, 1, 0]]])
    assert_array_equal(indices, expected_indices)


def test_connectivity_matrix():
    label_volume = np.array([[[3, 0, 0],
                              [0, 0, 0],
                              [0, 0, 4]]])
    streamlines = [np.array([[0, 0, 0], [0, 0, 0], [0, 2, 2]], 'float'),
                   np.array([[0, 0, 0], [0, 1, 1], [0, 2, 2]], 'float'),
                   np.array([[0, 2, 2], [0, 1, 1], [0, 0, 0]], 'float')]
    expected = np.zeros((5, 5), 'int')
    expected[3, 4] = 2
    expected[4, 3] = 1
    # Check basic Case
    matrix = connectivity_matrix(streamlines, label_volume, (1, 1, 1),
                                 symmetric=False)
    assert_array_equal(matrix, expected)
    # Test mapping
    matrix, mapping = connectivity_matrix(streamlines, label_volume, (1, 1, 1),
                                          symmetric=False, return_mapping=True)
    assert_array_equal(matrix, expected)
    assert_equal(mapping[3, 4], [0, 1])
    assert_equal(mapping[4, 3], [2])
    assert_equal(mapping.get((0, 0)), None)
    # Test mapping and symmetric
    matrix, mapping = connectivity_matrix(streamlines, label_volume, (1, 1, 1),
                                          symmetric=True, return_mapping=True)
    assert_equal(mapping[3, 4], [0, 1, 2])
    # When symmetric only (3,4) is a key, not (4, 3)
    assert_equal(mapping.get((4, 3)), None)
    # expected output matrix is symmetric version of expected
    expected = expected + expected.T
    assert_array_equal(matrix, expected)
    # Test mapping_as_streamlines, mapping dict has lists of streamlines
    matrix, mapping = connectivity_matrix(streamlines, label_volume, (1, 1, 1),
                                          symmetric=False,
                                          return_mapping=True,
                                          mapping_as_streamlines=True)
    assert_true(mapping[3, 4][0] is streamlines[0])
    assert_true(mapping[3, 4][1] is streamlines[1])
    assert_true(mapping[4, 3][0] is streamlines[2])

    # Test passing affine to connectivity_matrix
    affine = np.diag([-1, -1, -1, 1.])
    streamlines = [-i for i in streamlines]
    matrix = connectivity_matrix(streamlines, label_volume, affine=affine)
    # In the symmetrical case, the matrix should be, well, symmetric:
    assert_equal(matrix[4, 3], matrix[4, 3])


def test_ndbincount():
    def check(expected):
        assert_equal(bc[0, 0], expected[0])
        assert_equal(bc[0, 1], expected[1])
        assert_equal(bc[1, 0], expected[2])
        assert_equal(bc[2, 2], expected[3])

    x = np.array([[0, 0], [0, 0], [0, 1], [0, 1], [1, 0], [2, 2]]).T
    expected = [2, 2, 1, 1]
    # count occurrences in x
    bc = ndbincount(x)
    assert_equal(bc.shape, (3, 3))
    check(expected)
    # pass in shape
    bc = ndbincount(x, shape=(4, 5))
    assert_equal(bc.shape, (4, 5))
    check(expected)
    # pass in weights
    weights = np.arange(6.)
    weights[-1] = 1.23
    expeceted = [1., 5., 4., 1.23]
    bc = ndbincount(x, weights=weights)
    check(expeceted)
    # raises an error if shape is too small
    assert_raises(ValueError, ndbincount, x, None, (2, 2))


def test_reduce_labels():
    shape = (4, 5, 6)
    # labels from 100 to 220
    labels = np.arange(100, np.prod(shape) + 100).reshape(shape)
    # new labels form 0 to 120, and lookup maps range(0,120) to range(100, 220)
    new_labels, lookup = reduce_labels(labels)
    assert_array_equal(new_labels, labels - 100)
    assert_array_equal(lookup, labels.ravel())


def test_move_streamlines():
    streamlines = make_streamlines()
    affine = np.eye(4)
    new_streamlines = move_streamlines(streamlines, affine)
    for i, test_sl in enumerate(new_streamlines):
        assert_array_equal(test_sl, streamlines[i])

    affine[:3, 3] += (4, 5, 6)
    new_streamlines = move_streamlines(streamlines, affine)
    for i, test_sl in enumerate(new_streamlines):
        assert_array_equal(test_sl, streamlines[i] + (4, 5, 6))

    affine = np.eye(4)
    affine = affine[[2, 1, 0, 3]]
    new_streamlines = move_streamlines(streamlines, affine)
    for i, test_sl in enumerate(new_streamlines):
        assert_array_equal(test_sl, streamlines[i][:, [2, 1, 0]])

    affine[:3, 3] += (4, 5, 6)
    new_streamlines = move_streamlines(streamlines, affine)
    undo_affine = move_streamlines(new_streamlines, np.eye(4),
                                   input_space=affine)
    for i, test_sl in enumerate(undo_affine):
        assert_array_almost_equal(test_sl, streamlines[i])

    # Test that changing affine does affect moving streamlines
    affineA = affine.copy()
    affineB = affine.copy()
    streamlinesA = move_streamlines(streamlines, affineA)
    streamlinesB = move_streamlines(streamlines, affineB)
    affineB[:] = 0
    for (a, b) in zip(streamlinesA, streamlinesB):
        assert_array_equal(a, b)


def test_target():
    streamlines = [np.array([[0., 0., 0.],
                             [1., 0., 0.],
                             [2., 0., 0.]]),
                   np.array([[0., 0., 0],
                             [0, 1., 1.],
                             [0, 2., 2.]])]
    _target(target, streamlines, (0, 0, 0), (1, 0, 0), True)


def test_target_lb():
    streamlines = [np.array([[0., 1., 1.],
                             [3., 1., 1.]]),
                   np.array([[0., 0., 0.],
                             [2., 2., 2.]]),
                   np.array([[1., 1., 1.]])]  # Single-point streamline
    _target(target_line_based, streamlines, (1, 1, 1), (2, 1, 1), False)


def _target(target_f, streamlines, voxel_both_true, voxel_one_true,
            test_bad_points):
    affine = np.eye(4)
    mask = np.zeros((4, 4, 4), dtype=bool)

    # Both pass though
    mask[voxel_both_true] = True
    new = list(target_f(streamlines, mask, affine=affine))
    assert_equal(len(new), 2)
    new = list(target_f(streamlines, mask, affine=affine, include=False))
    assert_equal(len(new), 0)

    # only first
    mask[:] = False
    mask[voxel_one_true] = True
    new = list(target_f(streamlines, mask, affine=affine))
    assert_equal(len(new), 1)
    assert_true(new[0] is streamlines[0])
    new = list(target_f(streamlines, mask, affine=affine, include=False))
    assert_equal(len(new), 1)
    assert_true(new[0] is streamlines[1])

    # Test that bad points raise a value error
    if test_bad_points:
        bad_sl = streamlines + [np.array([[10.0, 10.0, 10.0]])]
        new = target_f(bad_sl, mask, affine=affine)
        assert_raises(ValueError, list, new)
        bad_sl = streamlines + [-np.array([[10.0, 10.0, 10.0]])]
        new = target_f(bad_sl, mask, affine=affine)
        assert_raises(ValueError, list, new)

    # Test smaller voxels
    affine = np.random.random((4, 4)) - .5
    affine[3] = [0, 0, 0, 1]
    streamlines = list(move_streamlines(streamlines, affine))
    new = list(target_f(streamlines, mask, affine=affine))
    assert_equal(len(new), 1)
    assert_true(new[0] is streamlines[0])
    new = list(target_f(streamlines, mask, affine=affine, include=False))
    assert_equal(len(new), 1)
    assert_true(new[0] is streamlines[1])

    # Test that changing mask or affine does not break target/target_line_based
    include = target_f(streamlines, mask, affine=affine)
    exclude = target_f(streamlines, mask, affine=affine, include=False)
    affine[:] = np.eye(4)
    mask[:] = False
    include = list(include)
    exclude = list(exclude)
    assert_equal(len(include), 1)
    assert_true(include[0] is streamlines[0])
    assert_equal(len(exclude), 1)
    assert_true(exclude[0] is streamlines[1])


def test_near_roi():
    streamlines = [np.array([[0., 0., 0.9],
                             [1.9, 0., 0.],
                             [3, 2., 2.]]),
                   np.array([[0.1, 0., 0],
                             [0, 1., 1.],
                             [0, 2., 2.]]),
                   np.array([[2, 2, 2],
                             [3, 3, 3]])]

    affine = np.eye(4)
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 0, 0] = True

    assert_array_equal(near_roi(streamlines, mask, tol=1),
                       np.array([True, True, False]))
    assert_array_equal(near_roi(streamlines, mask),
                       np.array([False, True, False]))

    # If there is an affine, we need to use it:
    affine[:, 3] = [-1, 100, -20, 1]
    # Transform the streamlines:
    x_streamlines = [sl + affine[:3, 3] for sl in streamlines]
    assert_array_equal(near_roi(x_streamlines, mask, affine=affine, tol=1),
                       np.array([True, True, False]))
    assert_array_equal(near_roi(x_streamlines, mask, affine=affine,
                                tol=None),
                       np.array([False, True, False]))

    # Test for use of the 'all' mode:
    assert_array_equal(near_roi(x_streamlines, mask, affine=affine, tol=None,
                                mode='all'), np.array([False, False, False]))

    mask[0, 1, 1] = True
    mask[0, 2, 2] = True
    # Test for use of the 'all' mode, also testing that setting the tolerance
    # to a very small number gets overridden:
    assert_array_equal(near_roi(x_streamlines, mask, affine=affine, tol=0.1,
                                mode='all'), np.array([False, True, False]))

    mask[2, 2, 2] = True
    mask[3, 3, 3] = True
    assert_array_equal(near_roi(x_streamlines, mask, affine=affine,
                                tol=None,
                                mode='all'),
                       np.array([False, True, True]))

    # Test for use of endpoints as selection criteria:
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[0, 1, 1] = True
    mask[3, 2, 2] = True

    assert_array_equal(near_roi(streamlines, mask, tol=0.87,
                                mode="either_end"),
                       np.array([True, False, False]))

    assert_array_equal(near_roi(streamlines, mask, tol=0.87,
                                mode="both_end"),
                       np.array([False, False, False]))

    mask[0, 0, 0] = True
    mask[0, 2, 2] = True
    assert_array_equal(near_roi(streamlines, mask, mode="both_end"),
                       np.array([False, True, False]))

    # Test with a generator input:
    def generate_sl(streamlines):
        for sl in streamlines:
            yield sl

    assert_array_equal(near_roi(generate_sl(streamlines),
                                mask, mode="both_end"),
                       np.array([False, True, False]))


def test_voxel_ornt():
    sh = (40, 40, 40)
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
    box = np.array(sh) * sz

    sra_affine = reorder_voxels_affine(ras, sra, sh, sz)
    toras_affine = reorder_voxels_affine(sra, ras, sh, sz)
    assert_array_equal(np.dot(toras_affine, sra_affine), I4)
    expected_sl = (sl[:, [2, 0, 1]] for sl in streamlines)
    test_sl = move_streamlines(streamlines, sra_affine)
    for _ in xrange(len(streamlines)):
        assert_array_equal(next(test_sl), next(expected_sl))

    lpi_affine = reorder_voxels_affine(ras, lpi, sh, sz)
    toras_affine = reorder_voxels_affine(lpi, ras, sh, sz)
    assert_array_equal(np.dot(toras_affine, lpi_affine), I4)
    expected_sl = (box - sl for sl in streamlines)
    test_sl = move_streamlines(streamlines, lpi_affine)
    for _ in xrange(len(streamlines)):
        assert_array_equal(next(test_sl), next(expected_sl))

    srp_affine = reorder_voxels_affine(ras, srp, sh, sz)
    toras_affine = reorder_voxels_affine(srp, ras, (40, 40, 40), (3, 1, 2))
    assert_array_equal(np.dot(toras_affine, srp_affine), I4)
    expected_sl = [sl.copy() for sl in streamlines]
    for sl in expected_sl:
        sl[:, 1] = box[1] - sl[:, 1]
    expected_sl = (sl[:, [2, 0, 1]] for sl in expected_sl)
    test_sl = move_streamlines(streamlines, srp_affine)
    for _ in xrange(len(streamlines)):
        assert_array_equal(next(test_sl), next(expected_sl))


def test_streamline_mapping():
    streamlines = [np.array([[0, 0, 0], [0, 0, 0], [0, 2, 2]], 'float'),
                   np.array([[0, 0, 0], [0, 1, 1], [0, 2, 2]], 'float'),
                   np.array([[0, 2, 2], [0, 1, 1], [0, 0, 0]], 'float')]
    mapping = streamline_mapping(streamlines, (1, 1, 1))
    expected = {(0, 0, 0): [0, 1, 2], (0, 2, 2): [0, 1, 2],
                (0, 1, 1): [1, 2]}
    assert_equal(mapping, expected)

    mapping = streamline_mapping(streamlines, (1, 1, 1),
                                 mapping_as_streamlines=True)
    expected = dict((k, [streamlines[i] for i in indices])
                    for k, indices in expected.items())
    assert_equal(mapping, expected)

    # Test passing affine
    affine = np.eye(4)
    affine[:3, 3] = .5
    mapping = streamline_mapping(streamlines, affine=affine,
                                 mapping_as_streamlines=True)
    assert_equal(mapping, expected)

    # Make the voxel size smaller
    affine = np.diag([.5, .5, .5, 1.])
    affine[:3, 3] = .25
    expected = dict((tuple(i * 2 for i in key), value)
                    for key, value in expected.items())
    mapping = streamline_mapping(streamlines, affine=affine,
                                 mapping_as_streamlines=True)
    assert_equal(mapping, expected)


def test_affine_for_trackvis():
    voxel_size = np.array([1., 2, 3.])
    affine = affine_for_trackvis(voxel_size)
    origin = np.dot(affine, [0, 0, 0, 1])
    assert_array_almost_equal(origin[:3], voxel_size / 2)


def test_length():
    # Generate a simulated bundle of fibers:
    n_streamlines = 50
    n_pts = 100
    t = np.linspace(-10, 10, n_pts)

    bundle = []
    for i in np.linspace(3, 5, n_streamlines):
        pts = np.vstack((np.cos(2 * t / np.pi), np.zeros(t.shape) + i, t)).T
        bundle.append(pts)

    start = np.random.randint(10, 30, n_streamlines)
    end = np.random.randint(60, 100, n_streamlines)

    bundle = [10 * streamline[start[i]:end[i]] for (i, streamline) in
              enumerate(bundle)]

    bundle_lengths = length(bundle)
    for idx, this_length in enumerate(bundle_lengths):
        assert_equal(this_length, metrix.length(bundle[idx]))


def test_seeds_from_mask():
    mask = np.random.random_integers(0, 1, size=(10, 10, 10))
    seeds = seeds_from_mask(mask, density=1)
    assert_equal(mask.sum(), len(seeds))
    assert_array_equal(np.argwhere(mask), seeds)

    mask[:] = False
    mask[3, 3, 3] = True
    seeds = seeds_from_mask(mask, density=[3, 4, 5])
    assert_equal(len(seeds), 3 * 4 * 5)
    assert_true(np.all((seeds > 2.5) & (seeds < 3.5)))

    mask[4, 4, 4] = True
    seeds = seeds_from_mask(mask, density=[3, 4, 5])
    assert_equal(len(seeds), 2 * 3 * 4 * 5)
    assert_true(np.all((seeds > 2.5) & (seeds < 4.5)))
    in_333 = ((seeds > 2.5) & (seeds < 3.5)).all(1)
    assert_equal(in_333.sum(), 3 * 4 * 5)
    in_444 = ((seeds > 3.5) & (seeds < 4.5)).all(1)
    assert_equal(in_444.sum(), 3 * 4 * 5)


def test_random_seeds_from_mask():
    mask = np.random.random_integers(0, 1, size=(4, 6, 3))
    seeds = random_seeds_from_mask(mask,
                                   seeds_count=24,
                                   seed_count_per_voxel=True)
    assert_equal(mask.sum() * 24, len(seeds))
    seeds = random_seeds_from_mask(mask,
                                   seeds_count=0,
                                   seed_count_per_voxel=True)
    assert_equal(0, len(seeds))

    mask[:] = False
    mask[2, 2, 2] = True
    seeds = random_seeds_from_mask(mask,
                                   seeds_count=8,
                                   seed_count_per_voxel=True)
    assert_equal(mask.sum() * 8, len(seeds))
    assert_true(np.all((seeds > 1.5) & (seeds < 2.5)))

    seeds = random_seeds_from_mask(mask,
                                   seeds_count=24,
                                   seed_count_per_voxel=False)
    assert_equal(24, len(seeds))
    seeds = random_seeds_from_mask(mask,
                                   seeds_count=0,
                                   seed_count_per_voxel=False)
    assert_equal(0, len(seeds))

    mask[:] = False
    mask[2, 2, 2] = True
    seeds = random_seeds_from_mask(mask,
                                   seeds_count=100,
                                   seed_count_per_voxel=False)
    assert_equal(100, len(seeds))
    assert_true(np.all((seeds > 1.5) & (seeds < 2.5)))


def test_connectivity_matrix_shape():
    # Labels: z-planes have labels 0,1,2
    labels = np.zeros((3, 3, 3), dtype=int)
    labels[:, :, 1] = 1
    labels[:, :, 2] = 2
    # Streamline set, only moves between first two z-planes.
    streamlines = [np.array([[0., 0., 0.],
                             [0., 0., 0.5],
                             [0., 0., 1.]]),
                   np.array([[0., 1., 1.],
                             [0., 1., 0.5],
                             [0., 1., 0.]])]
    matrix = connectivity_matrix(streamlines, labels, affine=np.eye(4))
    assert_equal(matrix.shape, (3, 3))


def test_unique_rows():
    """
    Testing the function unique_coords
    """
    arr = np.array([[1, 2, 3], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
    arr_w_unique = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    assert_array_equal(unique_rows(arr), arr_w_unique)

    # Should preserve order:
    arr = np.array([[2, 3, 4], [1, 2, 3], [1, 2, 3], [3, 4, 5]])
    arr_w_unique = np.array([[2, 3, 4], [1, 2, 3], [3, 4, 5]])
    assert_array_equal(unique_rows(arr), arr_w_unique)

    # Should work even with longer arrays:
    arr = np.array([[2, 3, 4], [1, 2, 3], [1, 2, 3], [3, 4, 5],
                    [6, 7, 8], [0, 1, 0], [1, 0, 1]])
    arr_w_unique = np.array([[2, 3, 4], [1, 2, 3], [3, 4, 5],
                             [6, 7, 8], [0, 1, 0], [1, 0, 1]])

    assert_array_equal(unique_rows(arr), arr_w_unique)


def test_reduce_rois():
    roi1 = np.zeros((4, 4, 4), dtype=np.bool)
    roi2 = np.zeros((4, 4, 4), dtype=np.bool)
    roi1[1, 1, 1] = 1
    roi2[2, 2, 2] = 1
    include_roi, exclude_roi = reduce_rois([roi1, roi2], [True, True])
    npt.assert_equal(include_roi, roi1 + roi2)
    npt.assert_equal(exclude_roi, np.zeros((4, 4, 4)))
    include_roi, exclude_roi = reduce_rois([roi1, roi2], [True, False])
    npt.assert_equal(include_roi, roi1)
    npt.assert_equal(exclude_roi, roi2)
    # Array input:
    include_roi, exclude_roi = reduce_rois(np.array([roi1, roi2]),
                                           [True, True])
    npt.assert_equal(include_roi, roi1 + roi2)
    npt.assert_equal(exclude_roi, np.zeros((4, 4, 4)))


def test_flexi_tvis_affine():
    sl_vox_order = 'RPI'
    grid_affine = np.array(
        [[-1.08566022e+00, 1.42664334e-03, 2.43463114e-01, 1.34783203e+02],
         [2.43251352e-03, 1.09376717e+00, 1.48301506e-02, -1.07367630e+02],
         [1.33170187e-01, -8.34854878e-03, 1.98454463e+00, -9.98151169e+01],
         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dim = (256, 256, 86)
    voxel_size = np.array([1.09379995, 1.09379995, 1.99947774])
    affine = flexi_tvis_affine(sl_vox_order, grid_affine, dim, voxel_size)

    origin = np.dot(affine, [0, 0, 0, 1])
    assert_array_almost_equal(origin[:3],
                              np.multiply(dim, voxel_size) - voxel_size / 2)


def test_get_flexi_tvis_affine():
    tvis_hdr = {'voxel_order': 'RPI', 'dim': (30, 40, 50),
                'voxel_size': [2, 3, 4]}

    grid_affine = np.array([[-2, 0, 0, 0],
                            [0, 3, 0, 0],
                            [0, 0, 4, 0],
                            [0, 0, 0, 1.]])

    affine = get_flexi_tvis_affine(tvis_hdr, grid_affine)

    origin = np.dot(affine, [0, 0, 0, 1])
    vsz = np.array(tvis_hdr['voxel_size'])
    assert_array_almost_equal(origin[:3],
                              np.multiply(tvis_hdr['dim'], vsz) - vsz / 2)


    # grid_affine =
    tvis_hdr['voxel_order'] = 'ASL'
    vsz = tvis_hdr['voxel_size'] = np.array([3, 4, 2.])
    affine = get_flexi_tvis_affine(tvis_hdr, grid_affine)

    vox_point = np.array([9, 8, 7])
    trk_point = np.dot(affine, np.append(vox_point, 1))

    assert_array_almost_equal(trk_point[:3],
                              (vox_point[[1, 2, 0]] + 0.5) * vsz)


def test_path_length():
    aoi = np.zeros((20, 20, 20), dtype=bool)
    aoi[0, 0, 0] = 1

    # A few tests for basic usage
    x = np.arange(20)
    streamlines = [np.array([x, x, x]).T]
    pl = path_length(streamlines, aoi, affine=np.eye(4))
    expected = x.copy() * np.sqrt(3)
    # expected[0] = np.inf
    npt.assert_array_almost_equal(pl[x, x, x], expected)

    aoi[19, 19, 19] = 1
    pl = path_length(streamlines, aoi, affine=np.eye(4))
    expected = np.minimum(expected, expected[::-1])
    npt.assert_array_almost_equal(pl[x, x, x], expected)

    aoi[19, 19, 19] = 0
    aoi[1, 1, 1] = 1
    pl = path_length(streamlines, aoi, affine=np.eye(4))
    expected = (x - 1) * np.sqrt(3)
    expected[0] = 0
    npt.assert_array_almost_equal(pl[x, x, x], expected)

    z = np.zeros(x.shape, x.dtype)
    streamlines.append(np.array([x, z, z]).T)
    pl = path_length(streamlines, aoi, affine=np.eye(4))
    npt.assert_array_almost_equal(pl[x, x, x], expected)
    npt.assert_array_almost_equal(pl[x, 0, 0], x)

    # Only streamlines that pass through aoi contribute to path length so if
    # all streamlines are duds, plm will be all inf.
    aoi[:] = 0
    aoi[0, 0, 0] = 1
    streamlines = []
    for i in range(1000):
        rando = np.random.random(size=(100, 3)) * 19 + .5
        assert (rando > .5).all()
        assert (rando < 19.5).all()
        streamlines.append(rando)
    pl = path_length(streamlines, aoi, affine=np.eye(4))
    npt.assert_array_almost_equal(pl, -1)

    pl = path_length(streamlines, aoi, affine=np.eye(4), fill_value=-12.)
    npt.assert_array_almost_equal(pl, -12.)


def test_min_at():
    k = np.array([3, 2, 2, 2, 1, 1, 1])
    values = np.array([10., 1, 2, 3, 31, 21, 11])
    i = np.zeros(k.shape, int)
    j = np.zeros(k.shape, int)
    a = np.zeros([1, 1, 4]) + 100.

    _min_at(a, (i, j, k), values)
    npt.assert_array_equal(a, [[[100, 11, 1, 10]]])
