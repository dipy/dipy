import warnings

import numpy as np
import pytest

from dipy.tracking import metrics
from dipy.tracking.streamline import transform_streamlines
from dipy.tracking.utils import (connectivity_matrix, density_map, length,
                                 ndbincount, reduce_labels, seeds_from_mask,
                                 random_seeds_from_mask, target,
                                 target_line_based, unique_rows, near_roi,
                                 reduce_rois, path_length, _min_at,
                                 max_angle_from_curvature,
                                 min_radius_curvature_from_angle)

from dipy.tracking._utils import _to_voxel_coordinates
from dipy.tracking.vox2track import streamline_mapping
import numpy.testing as npt
from dipy.testing import assert_true
from dipy.testing.decorators import set_random_number_generator


def make_streamlines(return_seeds=False):
    streamlines = [np.array([[0, 0, 0],
                             [1, 1, 1],
                             [2, 2, 2],
                             [5, 10, 12]], 'float'),
                   np.array([[1, 2, 3],
                             [3, 2, 0],
                             [5, 20, 33],
                             [40, 80, 120]], 'float')]
    seeds = [np.array([0., 0., 0.], 'float'),
             np.array([1., 2., 3.], 'float')]
    if return_seeds:
        return streamlines, seeds
    else:
        return streamlines


def test_density_map():
    # One streamline diagonal in volume
    streamlines = [np.array([np.arange(10)] * 3).T]
    shape = (10, 10, 10)
    x = np.arange(10)
    expected = np.zeros(shape)
    expected[x, x, x] = 1.
    dm = density_map(streamlines, np.eye(4), shape)
    npt.assert_array_equal(dm, expected)

    # add streamline, make voxel_size smaller. Each streamline should only be
    # counted once, even if multiple points lie in a voxel
    streamlines.append(np.ones((5, 3)))
    shape = (5, 5, 5)
    x = np.arange(5)
    expected = np.zeros(shape)
    expected[x, x, x] = 1.
    expected[0, 0, 0] += 1
    affine = np.eye(4) * 2
    affine[:3, 3] = 0.05
    dm = density_map(streamlines, affine, shape)
    npt.assert_array_equal(dm, expected)
    # should work with a generator
    dm = density_map(iter(streamlines), affine, shape)
    npt.assert_array_equal(dm, expected)

    # Test passing affine
    affine = np.diag([2, 2, 2, 1.])
    affine[: 3, 3] = 1.
    dm = density_map(streamlines, affine, shape)
    npt.assert_array_equal(dm, expected)

    # Shift the image by 2 voxels, ie 4mm
    affine[: 3, 3] -= 4.
    expected_old = expected
    new_shape = [i + 2 for i in shape]
    expected = np.zeros(new_shape)
    expected[2:, 2:, 2:] = expected_old
    dm = density_map(streamlines, affine, new_shape)
    npt.assert_array_equal(dm, expected)


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
    npt.assert_array_equal(indices, expected_indices)


def test_connectivity_matrix():
    label_volume = np.array([[[3, 0, 0],
                              [0, 0, 5],
                              [0, 0, 4]]])
    streamlines = [np.array([[0, 0, 0], [0, 1, 2], [0, 2, 2]], 'float'),
                   np.array([[0, 0, 0], [0, 1, 1], [0, 2, 2]], 'float'),
                   np.array([[0, 2, 2], [0, 1, 1], [0, 0, 0]], 'float')]
    expected = np.zeros((6, 6), 'int')
    expected[3, 4] = 2
    expected[4, 3] = 1
    # Check basic Case
    matrix = connectivity_matrix(streamlines, np.eye(4), label_volume,
                                 symmetric=False)
    npt.assert_array_equal(matrix, expected)
    # Test mapping
    matrix, mapping = connectivity_matrix(streamlines, np.eye(4), label_volume,
                                          symmetric=False,
                                          return_mapping=True)
    npt.assert_array_equal(matrix, expected)
    npt.assert_equal(mapping[3, 4], [0, 1])
    npt.assert_equal(mapping[4, 3], [2])
    npt.assert_equal(mapping.get((0, 0)), None)
    # Test mapping and symmetric
    matrix, mapping = connectivity_matrix(streamlines, np.eye(4), label_volume,
                                          symmetric=True, return_mapping=True)
    npt.assert_equal(mapping[3, 4], [0, 1, 2])
    # When symmetric only (3,4) is a key, not (4, 3)
    npt.assert_equal(mapping.get((4, 3)), None)
    # expected output matrix is symmetric version of expected
    expected = expected + expected.T
    npt.assert_array_equal(matrix, expected)
    # Test mapping_as_streamlines, mapping dict has lists of streamlines
    matrix, mapping = connectivity_matrix(streamlines, np.eye(4), label_volume,
                                          symmetric=False,
                                          return_mapping=True,
                                          mapping_as_streamlines=True)
    assert_true(mapping[3, 4][0] is streamlines[0])
    assert_true(mapping[3, 4][1] is streamlines[1])
    assert_true(mapping[4, 3][0] is streamlines[2])

    # Test Inclusive streamline analysis
    # Check basic Case (inclusive)
    expected = np.zeros((6, 6), 'int')
    expected[3, 4] = 2
    expected[4, 3] = 1
    expected[3, 5] = 1
    expected[5, 4] = 1
    expected[0, 3:5] = 1
    expected[3:5, 0] = 1

    matrix = connectivity_matrix(streamlines, np.eye(4), label_volume,
                                 symmetric=False, inclusive=True)
    npt.assert_array_equal(matrix, expected)

    # Test mapping
    matrix, mapping = connectivity_matrix(streamlines, np.eye(4), label_volume,
                                          inclusive=True, symmetric=False,
                                          return_mapping=True)
    npt.assert_array_equal(matrix, expected)
    npt.assert_equal(mapping[3, 4], [0, 1])
    npt.assert_equal(mapping[4, 3], [2])
    npt.assert_equal(mapping[0, 4], [1])
    npt.assert_equal(mapping[3, 5], [0])
    npt.assert_equal(mapping.get((0, 0)), None)

    # Test mapping and symmetric
    matrix, mapping = connectivity_matrix(streamlines, np.eye(4), label_volume,
                                          inclusive=True, symmetric=True,
                                          return_mapping=True)
    npt.assert_equal(mapping[3, 4], [0, 1, 2])
    npt.assert_equal(mapping[0, 3], [1, 2])
    npt.assert_equal(mapping[0, 4], [1, 2])
    npt.assert_equal(mapping[3, 5], [0])
    npt.assert_equal(mapping[4, 5], [0])

    # When symmetric only (3,4) is a key, not (4,3)
    npt.assert_equal(mapping.get((4, 3)), None)

    # expected output matrix is symmetric version of expected
    expected = expected + expected.T
    npt.assert_array_equal(matrix, expected)
    # Test mapping_as_streamlines, mapping dict has lists of streamlines
    matrix, mapping = connectivity_matrix(streamlines, np.eye(4), label_volume,
                                          symmetric=False, inclusive=True,
                                          return_mapping=True,
                                          mapping_as_streamlines=True)
    assert_true(mapping[0, 3][0] is streamlines[2])
    assert_true(mapping[0, 4][0] is streamlines[1])
    assert_true(mapping[3, 0][0] is streamlines[1])
    assert_true(mapping[3, 4][0] is streamlines[0])
    assert_true(mapping[3, 4][1] is streamlines[1])
    assert_true(mapping[3, 5][0] is streamlines[0])
    assert_true(mapping[4, 0][0] is streamlines[2])
    assert_true(mapping[4, 3][0] is streamlines[2])
    assert_true(mapping[5, 4][0] is streamlines[0])

    # Test passing affine to connectivity_matrix
    affine = np.diag([-1, -1, -1, 1.])
    streamlines = [-i for i in streamlines]
    matrix = connectivity_matrix(streamlines, affine, label_volume)
    # In the symmetrical case, the matrix should be, well, symmetric:
    npt.assert_equal(matrix[4, 3], matrix[4, 3])


def test_ndbincount():
    def check(expected):
        npt.assert_equal(bc[0, 0], expected[0])
        npt.assert_equal(bc[0, 1], expected[1])
        npt.assert_equal(bc[1, 0], expected[2])
        npt.assert_equal(bc[2, 2], expected[3])

    x = np.array([[0, 0], [0, 0], [0, 1], [0, 1], [1, 0], [2, 2]]).T
    expected = [2, 2, 1, 1]
    # count occurrences in x
    bc = ndbincount(x)
    npt.assert_equal(bc.shape, (3, 3))
    check(expected)
    # pass in shape
    bc = ndbincount(x, shape=(4, 5))
    npt.assert_equal(bc.shape, (4, 5))
    check(expected)
    # pass in weights
    weights = np.arange(6.)
    weights[-1] = 1.23
    expected = [1., 5., 4., 1.23]
    bc = ndbincount(x, weights=weights)
    check(expected)
    # raises an error if shape is too small
    npt.assert_raises(ValueError, ndbincount, x, None, (2, 2))


def test_reduce_labels():
    shape = (4, 5, 6)
    # labels from 100 to 220
    labels = np.arange(100, np.prod(shape) + 100).reshape(shape)
    # new labels form 0 to 120, and lookup maps range(0,120) to range(100, 220)
    new_labels, lookup = reduce_labels(labels)
    npt.assert_array_equal(new_labels, labels - 100)
    npt.assert_array_equal(lookup, labels.ravel())


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
    new = list(target_f(streamlines, affine, mask))
    npt.assert_equal(len(new), 2)
    new = list(target_f(streamlines, affine, mask, include=False))
    npt.assert_equal(len(new), 0)

    # only first
    mask[:] = False
    mask[voxel_one_true] = True
    new = list(target_f(streamlines, affine, mask))
    npt.assert_equal(len(new), 1)
    assert_true(new[0] is streamlines[0])
    new = list(target_f(streamlines, affine, mask, include=False))
    npt.assert_equal(len(new), 1)
    assert_true(new[0] is streamlines[1])

    # Test that bad points raise a value error
    if test_bad_points:
        bad_sl = streamlines + [np.array([[10.0, 10.0, 10.0]])]
        new = target_f(bad_sl, affine, mask)
        npt.assert_raises(ValueError, list, new)
        bad_sl = streamlines + [-np.array([[10.0, 10.0, 10.0]])]
        new = target_f(bad_sl, affine, mask)
        npt.assert_raises(ValueError, list, new)

    # Test smaller voxels
    affine = np.array([[.3, 0, 0, 0],
                       [0, .2, 0, 0],
                       [0, 0, .4, 0],
                       [0, 0, 0, 1]])
    streamlines = transform_streamlines(streamlines, affine)
    new = list(target_f(streamlines, affine, mask))
    npt.assert_equal(len(new), 1)
    assert_true(new[0] is streamlines[0])
    new = list(target_f(streamlines, affine, mask, include=False))
    npt.assert_equal(len(new), 1)
    assert_true(new[0] is streamlines[1])

    # Test that changing mask or affine does not break target/target_line_based
    include = target_f(streamlines, affine, mask)
    exclude = target_f(streamlines, affine, mask, include=False)
    affine[:] = np.eye(4)
    mask[:] = False
    include = list(include)
    exclude = list(exclude)
    npt.assert_equal(len(include), 1)
    assert_true(include[0] is streamlines[0])
    npt.assert_equal(len(exclude), 1)
    assert_true(exclude[0] is streamlines[1])


def test_target_line_based_out_of_bounds():
    test_cases = [
        (np.array([[-10, 0, 0], [0, 0, 0]]), 0),
        (np.array([[-10, 0, 0], [1, -10, 0]]), 0),
        (np.array([[0, 0, 0], [10, 10, 10]]), 0),
        (np.array([[-2, -0.6, -0.6], [2, -0.6, -0.6]]), 0),
        (np.array([[-10000, 0, 0], [0, 0, 0]]), 0),
        (np.array([[-10, 0, 0], [10, 0, 0]]), 1),
        (np.array([[1, -10, 0], [1, 10, 0]]), 1),
    ]
    for streamline, expected_matched in test_cases:
        mask = np.zeros((2, 1, 1), dtype=np.uint8)
        mask[1, 0, 0] = 1
        matched = list(target_line_based([streamline], np.eye(4), mask))
        assert len(matched) == expected_matched


def test_near_roi():

    streamlines = [np.array([[0., 0., 0.9],
                             [1.9, 0., 0.],
                             [3, 2., 2.]]),
                   np.array([[0.1, 0., 0],
                             [0, 1., 1.],
                             [0, 2., 2.]]),
                   np.array([[2, 2, 2],
                             [3, 3, 3]])]

    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 0, 0] = True

    npt.assert_array_equal(near_roi(streamlines, np.eye(4), mask, tol=1),
                           np.array([True, True, False]))
    npt.assert_array_equal(near_roi(streamlines, np.eye(4), mask),
                           np.array([False, True, False]))

    # test for handling of various forms of null streamlines
    # including a streamline from previous test because near_roi / tol
    # can't handle completely empty streamline collections
    streamlines_null = [np.array([[0., 0., 0.9],
                                 [1.9, 0., 0.],
                                 [3, 2., 2.]]),
                        np.array([[],
                                 [],
                                 []]).T,
                        np.array([]),
                        []]
    npt.assert_array_equal(near_roi(streamlines_null, np.eye(4), mask, tol=1),
                           np.array([True, False, False, False]))
    npt.assert_array_equal(near_roi(streamlines_null, np.eye(4), mask),
                           np.array([False, False, False, False]))

    # If there is an affine, we need to use it:
    affine = np.eye(4)
    affine[:, 3] = [-1, 100, -20, 1]
    # Transform the streamlines:
    x_streamlines = [sl + affine[:3, 3] for sl in streamlines]
    npt.assert_array_equal(near_roi(x_streamlines, affine, mask, tol=1),
                           np.array([True, True, False]))
    npt.assert_array_equal(near_roi(x_streamlines, affine, mask,
                                    tol=None),
                           np.array([False, True, False]))

    # Test for use of the 'all' mode:
    npt.assert_array_equal(near_roi(x_streamlines, affine, mask,
                                    tol=None, mode='all'),
                           np.array([False, False, False]))

    mask[0, 1, 1] = True
    mask[0, 2, 2] = True
    # Test for use of the 'all' mode, also testing that setting the tolerance
    # to a very small number gets overridden:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        npt.assert_array_equal(near_roi(x_streamlines, affine, mask,
                                        tol=0.1,
                                        mode='all'),
                               np.array([False, True, False]))

    mask[2, 2, 2] = True
    mask[3, 3, 3] = True
    npt.assert_array_equal(near_roi(x_streamlines, affine, mask,
                                    tol=None, mode='all'),
                           np.array([False, True, True]))

    # Test for use of endpoints as selection criteria:
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[0, 1, 1] = True
    mask[3, 2, 2] = True

    npt.assert_array_equal(near_roi(streamlines, np.eye(4), mask, tol=0.87,
                                    mode="either_end"),
                           np.array([True, False, False]))

    npt.assert_array_equal(near_roi(streamlines, np.eye(4), mask, tol=0.87,
                                    mode="both_end"),
                           np.array([False, False, False]))

    mask[0, 0, 0] = True
    mask[0, 2, 2] = True
    npt.assert_array_equal(near_roi(streamlines, np.eye(4), mask,
                                    mode="both_end"),
                           np.array([False, True, False]))

    # Test with a generator input:
    def generate_sl(streamlines):
        for sl in streamlines:
            yield sl

    npt.assert_array_equal(near_roi(generate_sl(streamlines), np.eye(4),
                                    mask, mode="both_end"),
                           np.array([False, True, False]))


def test_streamline_mapping():
    streamlines = [np.array([[0, 0, 0], [0, 0, 0], [0, 2, 2]], 'float'),
                   np.array([[0, 0, 0], [0, 1, 1], [0, 2, 2]], 'float'),
                   np.array([[0, 2, 2], [0, 1, 1], [0, 0, 0]], 'float')]
    mapping = streamline_mapping(streamlines, affine=np.eye(4))
    expected = {(0, 0, 0): [0, 1, 2], (0, 2, 2): [0, 1, 2],
                (0, 1, 1): [1, 2]}
    npt.assert_equal(mapping, expected)

    mapping = streamline_mapping(streamlines, affine=np.eye(4),
                                 mapping_as_streamlines=True)
    expected = dict((k, [streamlines[i] for i in indices])
                    for k, indices in expected.items())
    npt.assert_equal(mapping, expected)

    # Test passing affine
    affine = np.eye(4)
    affine[:3, 3] = .5
    mapping = streamline_mapping(streamlines, affine=affine,
                                 mapping_as_streamlines=True)
    npt.assert_equal(mapping, expected)

    # Make the voxel size smaller
    affine = np.diag([.5, .5, .5, 1.])
    affine[:3, 3] = .25
    expected = dict((tuple(i * 2 for i in key), value)
                    for key, value in expected.items())
    mapping = streamline_mapping(streamlines, affine=affine,
                                 mapping_as_streamlines=True)
    npt.assert_equal(mapping, expected)


@set_random_number_generator()
def test_length(rng):
    # Generate a simulated bundle of fibers:
    n_streamlines = 50
    n_pts = 100
    t = np.linspace(-10, 10, n_pts)

    bundle = []
    for i in np.linspace(3, 5, n_streamlines):
        pts = np.vstack((np.cos(2 * t / np.pi), np.zeros(t.shape) + i, t)).T
        bundle.append(pts)

    start = rng.integers(10, 30, n_streamlines)
    end = rng.integers(60, 100, n_streamlines)

    bundle = [10 * streamline[start[i]:end[i]] for (i, streamline) in
              enumerate(bundle)]

    bundle_lengths = length(bundle)
    for idx, this_length in enumerate(bundle_lengths):
        npt.assert_equal(this_length, metrics.length(bundle[idx]))


@set_random_number_generator()
def test_seeds_from_mask(rng):
    mask = rng.integers(0, 1, size=(10, 10, 10))
    seeds = seeds_from_mask(mask, np.eye(4), density=1)
    npt.assert_equal(mask.sum(), len(seeds))
    npt.assert_array_equal(np.argwhere(mask), seeds)

    mask[:] = False
    mask[3, 3, 3] = True
    seeds = seeds_from_mask(mask, np.eye(4), density=[3, 4, 5])
    npt.assert_equal(len(seeds), 3 * 4 * 5)
    assert_true(np.all((seeds > 2.5) & (seeds < 3.5)))

    mask[4, 4, 4] = True
    seeds = seeds_from_mask(mask, np.eye(4), density=[3, 4, 5])
    npt.assert_equal(len(seeds), 2 * 3 * 4 * 5)
    assert_true(np.all((seeds > 2.5) & (seeds < 4.5)))
    in_333 = ((seeds > 2.5) & (seeds < 3.5)).all(1)
    npt.assert_equal(in_333.sum(), 3 * 4 * 5)
    in_444 = ((seeds > 3.5) & (seeds < 4.5)).all(1)
    npt.assert_equal(in_444.sum(), 3 * 4 * 5)


@set_random_number_generator()
def test_random_seeds_from_mask(rng):
    mask = rng.integers(0, 1, size=(4, 6, 3))
    seeds = random_seeds_from_mask(mask, np.eye(4),
                                   seeds_count=24,
                                   seed_count_per_voxel=True)
    npt.assert_equal(mask.sum() * 24, len(seeds))
    seeds = random_seeds_from_mask(mask, np.eye(4),
                                   seeds_count=0,
                                   seed_count_per_voxel=True)
    npt.assert_equal(0, len(seeds))

    mask[:] = False
    mask[2, 2, 2] = True
    seeds = random_seeds_from_mask(mask, np.eye(4),
                                   seeds_count=8,
                                   seed_count_per_voxel=True)
    npt.assert_equal(mask.sum() * 8, len(seeds))
    assert_true(np.all((seeds > 1.5) & (seeds < 2.5)))

    seeds = random_seeds_from_mask(mask, np.eye(4),
                                   seeds_count=24,
                                   seed_count_per_voxel=False)
    npt.assert_equal(24, len(seeds))
    seeds = random_seeds_from_mask(mask, np.eye(4),
                                   seeds_count=0,
                                   seed_count_per_voxel=False)
    npt.assert_equal(0, len(seeds))

    mask[:] = False
    mask[2, 2, 2] = True
    seeds = random_seeds_from_mask(mask, np.eye(4),
                                   seeds_count=100,
                                   seed_count_per_voxel=False)
    npt.assert_equal(100, len(seeds))
    assert_true(np.all((seeds > 1.5) & (seeds < 2.5)))

    mask = np.zeros((15, 15, 15))
    mask[2:14, 2:14, 2:14] = 1
    seeds_npv_2 = random_seeds_from_mask(mask, np.eye(4), seeds_count=2,
                                         seed_count_per_voxel=True,
                                         random_seed=0)[:150]
    seeds_npv_3 = random_seeds_from_mask(mask, np.eye(4), seeds_count=3,
                                         seed_count_per_voxel=True,
                                         random_seed=0)[:150]
    assert_true(np.all(seeds_npv_2 == seeds_npv_3))

    seeds_nt_150 = random_seeds_from_mask(mask, np.eye(4), seeds_count=150,
                                          seed_count_per_voxel=False,
                                          random_seed=0)[:150]
    seeds_nt_500 = random_seeds_from_mask(mask, np.eye(4), seeds_count=500,
                                          seed_count_per_voxel=False,
                                          random_seed=0)[:150]
    assert_true(np.all(seeds_nt_150 == seeds_nt_500))


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
    matrix = connectivity_matrix(streamlines, np.eye(4), labels)
    npt.assert_equal(matrix.shape, (3, 3))


def test_unique_rows():
    """
    Testing the function unique_coords
    """
    arr = np.array([[1, 2, 3], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
    arr_w_unique = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    npt.assert_array_equal(unique_rows(arr), arr_w_unique)

    # Should preserve order:
    arr = np.array([[2, 3, 4], [1, 2, 3], [1, 2, 3], [3, 4, 5]])
    arr_w_unique = np.array([[2, 3, 4], [1, 2, 3], [3, 4, 5]])
    npt.assert_array_equal(unique_rows(arr), arr_w_unique)

    # Should work even with longer arrays:
    arr = np.array([[2, 3, 4], [1, 2, 3], [1, 2, 3], [3, 4, 5],
                    [6, 7, 8], [0, 1, 0], [1, 0, 1]])
    arr_w_unique = np.array([[2, 3, 4], [1, 2, 3], [3, 4, 5],
                             [6, 7, 8], [0, 1, 0], [1, 0, 1]])

    npt.assert_array_equal(unique_rows(arr), arr_w_unique)


def test_reduce_rois():
    roi1 = np.zeros((4, 4, 4), dtype=bool)
    roi2 = np.zeros((4, 4, 4), dtype=bool)
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
    # Int and float input
    roi1 = np.zeros((4, 4, 4), dtype=int)
    roi2 = np.zeros((4, 4, 4), dtype=float)
    npt.assert_warns(UserWarning, reduce_rois, [roi1], [True])
    npt.assert_warns(UserWarning, reduce_rois, [roi2], [True])


@set_random_number_generator()
def test_path_length(rng):
    aoi = np.zeros((20, 20, 20), dtype=bool)
    aoi[0, 0, 0] = 1

    # A few tests for basic usage
    x = np.arange(20)
    streamlines = [np.array([x, x, x]).T]
    pl = path_length(streamlines, np.eye(4), aoi)
    expected = x.copy() * np.sqrt(3)
    # expected[0] = np.inf
    npt.assert_array_almost_equal(pl[x, x, x], expected)

    aoi[19, 19, 19] = 1
    pl = path_length(streamlines, np.eye(4), aoi)
    expected = np.minimum(expected, expected[::-1])
    npt.assert_array_almost_equal(pl[x, x, x], expected)

    aoi[19, 19, 19] = 0
    aoi[1, 1, 1] = 1
    pl = path_length(streamlines, np.eye(4), aoi)
    expected = (x - 1) * np.sqrt(3)
    expected[0] = 0
    npt.assert_array_almost_equal(pl[x, x, x], expected)

    z = np.zeros(x.shape, x.dtype)
    streamlines.append(np.array([x, z, z]).T)
    pl = path_length(streamlines, np.eye(4), aoi)
    npt.assert_array_almost_equal(pl[x, x, x], expected)
    npt.assert_array_almost_equal(pl[x, 0, 0], x)

    # Only streamlines that pass through aoi contribute to path length so if
    # all streamlines are duds, plm will be all inf.
    aoi[:] = 0
    aoi[0, 0, 0] = 1
    streamlines = []
    for i in range(1000):
        rando = rng.random(size=(100, 3)) * 19 + .5
        assert (rando > .5).all()
        assert (rando < 19.5).all()
        streamlines.append(rando)
    pl = path_length(streamlines, np.eye(4), aoi)
    npt.assert_array_almost_equal(pl, -1)

    pl = path_length(streamlines, np.eye(4), aoi, fill_value=-12.)
    npt.assert_array_almost_equal(pl, -12.)


def test_min_at():
    k = np.array([3, 2, 2, 2, 1, 1, 1])
    values = np.array([10., 1, 2, 3, 31, 21, 11])
    i = np.zeros(k.shape, int)
    j = np.zeros(k.shape, int)
    a = np.zeros([1, 1, 4]) + 100.

    _min_at(a, (i, j, k), values)
    npt.assert_array_equal(a, [[[100, 11, 1, 10]]])


def test_curvature_angle():
    angle = [0.0000001, np.pi/3, np.pi/2.01]
    step_size = [0.2, 0.5, 1.5]
    curvature = [2000000., 0.5, 1.064829060280437]

    for theta, step, curve in zip(angle, step_size, curvature):
        res_angle = max_angle_from_curvature(curve, step)
        npt.assert_almost_equal(res_angle, theta)

        res_curvature = min_radius_curvature_from_angle(theta, step)
        npt.assert_almost_equal(res_curvature, curve)

    # special case
    with pytest.warns(UserWarning):
        npt.assert_equal(min_radius_curvature_from_angle(0, 1),
                         min_radius_curvature_from_angle(np.pi/2, 1))
