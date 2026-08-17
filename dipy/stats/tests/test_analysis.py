import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose

from dipy.stats.analysis import (
    afq_profile,
    buan_profile,
    compute_robust_centroid,
    create_radial_bins,
    gaussian_weights,
    get_grid_from_atlas,
    parameterize_bundle,
    spectra_assignment_map,
    spectra_profile,
)
from dipy.tracking.streamline import Streamlines


def test_gaussian_weights():
    # Some bogus x,y,z coordinates
    x = np.arange(10).astype(float)
    y = np.arange(10).astype(float)
    z = np.arange(10).astype(float)

    # Create a distribution for which we can predict the weights we would
    # expect to get:
    bundle = Streamlines([np.array([x, y, z]).T + 1, np.array([x, y, z]).T - 1])
    # In this case, all nodes receives an equal weight of 0.5:
    w = gaussian_weights(bundle, n_points=10)
    npt.assert_almost_equal(w, np.ones((len(bundle), 10)) * 0.5)

    # Test when asked to return Mahalanobis, instead of weights
    w = gaussian_weights(bundle, n_points=10, return_mahalnobis=True)
    npt.assert_almost_equal(w, np.ones((len(bundle), 10)))

    # Here, some nodes are twice as far from the mean as others
    bundle = Streamlines(
        [
            np.array([x, y, z]).T + 2,
            np.array([x, y, z]).T + 1,
            np.array([x, y, z]).T - 1,
            np.array([x, y, z]).T - 2,
        ]
    )
    w = gaussian_weights(bundle, n_points=10)

    # And their weights should be halved:
    npt.assert_almost_equal(w[0], w[1] / 2)
    npt.assert_almost_equal(w[-1], w[2] / 2)

    # Test the situation where all the streamlines have an identical node:
    arr1 = np.array([x, y, z]).T + 2
    arr2 = np.array([x, y, z]).T + 1
    arr3 = np.array([x, y, z]).T - 1
    arr4 = np.array([x, y, z]).T - 2

    arr1[0] = np.array([1, 1, 1])
    arr2[0] = np.array([1, 1, 1])
    arr3[0] = np.array([1, 1, 1])
    arr4[0] = np.array([1, 1, 1])

    bundle_w_id_node = Streamlines([arr1, arr2, arr3, arr4])
    w = gaussian_weights(Streamlines(bundle_w_id_node), n_points=10)
    # For this case, the result should be a weight of 1/n_streamlines in that
    # node for all streamlines:
    npt.assert_equal(
        w[:, 0], np.ones(len(bundle_w_id_node)) * 1 / len(bundle_w_id_node)
    )

    # Test the situation where all the streamlines are copies of each other:
    bundle_w_copies = Streamlines([bundle[0], bundle[0], bundle[0], bundle[0]])
    w = gaussian_weights(bundle_w_copies, n_points=10)
    # In this case, the entire array should be equal to 1/n_streamlines:
    npt.assert_equal(w, np.ones(w.shape) * 1 / len(bundle_w_id_node))

    # Test with bundle of length 1:
    bundle_len_1 = Streamlines([bundle[0]])
    w = gaussian_weights(bundle_len_1, n_points=10)
    npt.assert_equal(w, np.ones(w.shape))

    bundle_len_1 = Streamlines([bundle[0]])
    w = gaussian_weights(bundle_len_1, n_points=10, return_mahalnobis=True)
    npt.assert_equal(w, np.ones(w.shape) * np.nan)


def test_afq_profile():
    data = np.ones((10, 10, 10))
    bundle = Streamlines()
    bundle.extend(np.array([[[0, 0.0, 0], [1, 0.0, 0.0], [2, 0.0, 0.0]]]))
    bundle.extend(np.array([[[0, 0.0, 0.0], [1, 0.0, 0], [2, 0, 0.0]]]))

    profile = afq_profile(data, bundle, np.eye(4))
    npt.assert_equal(profile, np.ones(100))

    profile = afq_profile(data, bundle, np.eye(4), n_points=10, weights=None)
    npt.assert_equal(profile, np.ones(10))

    profile = afq_profile(
        data, bundle, np.eye(4), weights=gaussian_weights, stat=np.median
    )

    npt.assert_equal(profile, np.ones(100))

    profile = afq_profile(
        data,
        bundle,
        np.eye(4),
        orient_by=bundle[0],
        weights=gaussian_weights,
        stat=np.median,
    )

    npt.assert_equal(profile, np.ones(100))

    profile = afq_profile(data, bundle, np.eye(4), n_points=10, weights=None)
    npt.assert_equal(profile, np.ones(10))

    profile = afq_profile(
        data, bundle, np.eye(4), n_points=10, weights=np.ones((2, 10)) * 0.5
    )
    npt.assert_equal(profile, np.ones(10))

    profile = afq_profile(data, bundle, np.eye(4), n_points=10, stat=np.median)
    npt.assert_equal(profile, np.ones(10))

    # Disallow setting weights that don't sum to 1 across fibers/nodes:
    npt.assert_raises(
        ValueError,
        afq_profile,
        data,
        bundle,
        np.eye(4),
        n_points=10,
        weights=np.ones((2, 10)) * 0.6,
    )

    # Test using an affine:
    affine = np.eye(4)
    affine[:, 3] = [-1, 100, -20, 1]
    # Transform the streamlines:
    bundle._data = bundle._data + affine[:3, 3]
    profile = afq_profile(data, bundle, affine, n_points=10, weights=None)

    npt.assert_equal(profile, np.ones(10))

    # Test for error-handling:
    empty_bundle = Streamlines([])
    npt.assert_raises(ValueError, afq_profile, data, empty_bundle, np.eye(4))


def test_buan_profile():
    data = np.ones((40, 40, 40), dtype=float)

    # Create 10 streamlines
    n_pts = 20
    x = np.linspace(5, 34, n_pts)
    y0 = 20.0
    z0 = 20.0
    base = np.vstack([x, np.ones(n_pts) * y0, np.ones(n_pts) * z0]).T

    bundle = Streamlines([base + np.array([0, i, 0]) for i in range(10)])
    model_bundle = Streamlines([base + np.array([0, -i, 0]) for i in range(10)])
    orig_bundle = Streamlines([base + np.array([0, i, 0]) for i in range(10)])

    affine = np.eye(4)

    profile = buan_profile(model_bundle, bundle, orig_bundle, data, affine, no_disks=10)
    npt.assert_equal(profile.shape, (10,))
    npt.assert_almost_equal(profile, np.ones(10))

    # Test with a different number of disks/segments
    profile = buan_profile(model_bundle, bundle, orig_bundle, data, affine, no_disks=5)
    npt.assert_equal(profile.shape, (5,))
    npt.assert_almost_equal(profile, np.ones(5))

    # Test NaN handling: if data is all NaNs, output should be all NaNs
    data_nan = np.ones((40, 40, 40), dtype=float) * np.nan
    profile = buan_profile(
        model_bundle, bundle, orig_bundle, data_nan, affine, no_disks=10
    )
    npt.assert_equal(profile.shape, (10,))
    npt.assert_equal(np.all(np.isnan(profile)), True)

    # Test error-handling: empty bundle should raise
    empty_bundle = Streamlines([])
    npt.assert_raises(
        ValueError,
        buan_profile,
        model_bundle,
        empty_bundle,
        empty_bundle,
        data,
        affine,
        no_disks=10,
    )


def test_compute_robust_centroid():
    x = np.linspace(0, 20, 21)
    bundle = Streamlines(
        [
            np.column_stack((x, np.full_like(x, y), np.zeros_like(x)))
            for y in [-2, -1, 0, 1, 2]
        ]
    )

    centroid = compute_robust_centroid(bundle, n_segments=10)

    assert centroid.shape == (10, 3)
    assert_allclose(centroid[:, 1], 0, atol=1e-6)
    assert_allclose(centroid[:, 2], 0, atol=1e-6)


def test_create_radial_bins():
    radial_distance = np.array([-2, -1, 0, 1, 2])

    r_index, n_radial, r_edges = create_radial_bins(radial_distance, n_radial=3)

    assert n_radial == 3
    assert len(r_edges) == 4
    assert np.all(r_index >= 0)
    assert np.all(r_index < n_radial)


def test_get_grid_from_atlas():
    x = np.linspace(0, 20, 21)
    bundle = Streamlines(
        [
            np.column_stack((x, np.full_like(x, y), np.zeros_like(x)))
            for y in [-2, -1, 0, 1, 2]
        ]
    )

    result = get_grid_from_atlas(bundle, n_segments=10, n_radial=3)
    s_index, r_index, centroid, radial_vectors, r_edges, s_len, r_len = result

    assert centroid.shape == (10, 3)
    assert radial_vectors.shape == (10, 3)
    assert len(r_edges) == 4
    assert np.all(s_index >= 0)
    assert np.all(s_index < 10)
    assert np.all(r_index >= 0)
    assert np.all(r_index < 3)
    assert s_len > 0
    assert r_len > 0

    assert_allclose(np.linalg.norm(radial_vectors, axis=1), 1, atol=1e-6)


def test_parameterize_bundle():
    x = np.linspace(0, 20, 21)

    model_bundle = Streamlines(
        [
            np.column_stack((x, np.full_like(x, y), np.zeros_like(x)))
            for y in [-2, -1, 0, 1, 2]
        ]
    )

    target_bundle = Streamlines(
        [np.column_stack((x, np.full_like(x, y), np.zeros_like(x))) for y in [-1, 0, 1]]
    )

    result = get_grid_from_atlas(model_bundle, n_segments=10, n_radial=3)
    _, _, centroid, radial_vectors, r_edges, _, _ = result

    s_index, r_index, dist, valid_mask, counts = parameterize_bundle(
        target_bundle, centroid, radial_vectors, r_edges
    )

    n_points = sum(len(streamline) for streamline in target_bundle)

    assert len(s_index) == n_points
    assert len(r_index) == n_points
    assert len(dist) == n_points
    assert len(valid_mask) == n_points
    assert counts.shape == (10, 3)

    assert np.all(s_index >= 0)
    assert np.all(s_index < 10)
    assert np.all(r_index >= 0)
    assert np.all(r_index < 3)
    assert np.all(dist >= 0)


def test_spectra_assignment_map():
    x = np.linspace(0, 20, 21)

    model_bundle = Streamlines(
        [
            np.column_stack((x, np.full_like(x, y), np.zeros_like(x)))
            for y in [-2, -1, 0, 1, 2]
        ]
    )

    target_bundle = Streamlines(
        [np.column_stack((x, np.full_like(x, y), np.zeros_like(x))) for y in [-1, 0, 1]]
    )

    s_index, r_index, dist, valid_mask, counts = spectra_assignment_map(
        target_bundle, model_bundle, n_segments=10, n_radial=3
    )

    n_points = sum(len(streamline) for streamline in target_bundle)

    assert len(s_index) == n_points
    assert len(r_index) == n_points
    assert len(dist) == n_points
    assert len(valid_mask) == n_points
    assert counts.shape == (10, 3)

    assert np.all(s_index >= 0)
    assert np.all(s_index < 10)
    assert np.all(r_index >= 0)
    assert np.all(r_index < 3)


def test_spectra_radial_assignment():
    x = np.linspace(0, 20, 21)

    model_bundle = Streamlines(
        [
            np.column_stack((x, np.full_like(x, y), np.zeros_like(x)))
            for y in [-2, -1, 0, 1, 2]
        ]
    )

    lower = np.column_stack((x, np.full_like(x, -1.5), np.zeros_like(x)))
    upper = np.column_stack((x, np.full_like(x, 1.5), np.zeros_like(x)))

    target_bundle = Streamlines([lower, upper])

    _, r_index, _, _, _ = spectra_assignment_map(
        target_bundle, model_bundle, n_segments=10, n_radial=3
    )

    n_points = len(x)
    lower_index = r_index[:n_points]
    upper_index = r_index[n_points:]

    assert np.median(lower_index) != np.median(upper_index)


def test_spectra_profile():
    x = np.linspace(2, 22, 101)

    model_bundle = Streamlines(
        [
            np.column_stack((x, np.full_like(x, y), np.full_like(x, 10)))
            for y in np.linspace(8, 12, 61)
        ]
    )

    bundle = Streamlines(
        [
            np.column_stack((x, np.full_like(x, y), np.full_like(x, 10)))
            for y in np.linspace(8.5, 11.5, 61)
        ]
    )

    metric = np.full((30, 30, 30), 2.5)

    profile = spectra_profile(
        model_bundle,
        bundle,
        bundle,
        metric,
        np.eye(4),
        n_segments=10,
        n_radial=3,
    )

    assert profile.shape == (10, 3)

    valid = np.isfinite(profile)

    assert np.any(valid)
    assert_allclose(profile[valid], 2.5)
