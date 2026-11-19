import numpy as np
import numpy.testing as npt
import pytest
from scipy import stats
from scipy.ndimage import gaussian_filter

from dipy.stats.analysis import (
    SpectraGrid,
    _cohen_d,
    _storey_pi0,
    afq_profile,
    buan_profile,
    compute_robust_centroid,
    conditional_pvalue,
    create_radial_bins,
    empirical_correlation,
    estimate_length_scales,
    fdr_correction,
    fit_grid_lmm,
    fit_matern,
    gaussian_weights,
    get_grid_from_atlas,
    hierarchical_fdr,
    matern_correlation,
    parameterize_bundle,
    partition_bundle_into_blocks,
    simes_combine,
    spectra_assignment_map,
    spectra_profile,
    stouffer_combine,
)
from dipy.tracking.streamline import Streamlines
from dipy.utils.optpkg import optional_package


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


def test_afq_profile_gaussian_weights_node_count():
    data = np.ones((10, 10, 10))
    streamline = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    bundle = Streamlines([streamline, streamline.copy()])
    profile = afq_profile(
        data,
        bundle,
        np.eye(4),
        n_points=10,
        weights=gaussian_weights,
        stat=np.median,
    )
    npt.assert_allclose(profile, np.ones(10))


def test_afq_profile_custom_weight_callable():
    data = np.ones((10, 10, 10))
    streamline = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    bundle = Streamlines([streamline, streamline.copy()])

    def weights(streamlines):
        return np.full(len(streamlines), 1 / len(streamlines))

    profile = afq_profile(data, bundle, np.eye(4), n_points=10, weights=weights)
    npt.assert_allclose(profile, np.ones(10))


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
    npt.assert_allclose(centroid[:, 1], 0, atol=1e-6)
    npt.assert_allclose(centroid[:, 2], 0, atol=1e-6)


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

    npt.assert_allclose(np.linalg.norm(radial_vectors, axis=1), 1, atol=1e-6)


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
    npt.assert_allclose(profile[valid], 2.5)


def _spectra_bundle(seed=0, n=60):
    rng = np.random.default_rng(seed)
    lines = [
        np.c_[
            np.linspace(0, 50, 60),
            rng.normal(0, 3, 60),
            rng.normal(0, 3, 60),
        ]
        + 25
        for _ in range(n)
    ]
    return lines


def test_create_radial_bins_empty_bin_labels():
    # An empty interior bin must not shift labels away from the edges
    dist = np.concatenate([[-5] * 2, [-2] * 100, [2] * 100, [4] * 100, [5]]).astype(
        float
    )
    index, n_bins, edges = create_radial_bins(dist, radial_length=2.0)
    npt.assert_array_equal(edges, [-3, -1, 1, 3, 5])
    assert n_bins == 4
    expected = np.clip(np.digitize(dist, edges) - 1, 0, n_bins - 1)
    npt.assert_array_equal(index, expected)
    assert index[dist == 4][0] == 3


def test_create_radial_bins_median_ignores_empty_bins():
    # Most bins are empty: the median of all bins would be 0 and nothing
    # would ever be merged. Non-empty median (100) merges the 2-point bin.
    dist = np.concatenate([[0.0] * 2, [10.0] * 100, [20.0] * 100, [40.0] * 100])
    _, n_bins, edges = create_radial_bins(dist, radial_length=5.0, merge_threshold=0.2)
    assert edges[0] > dist.min()
    # a threshold of 0 never merges
    _, n_none, edges_none = create_radial_bins(
        dist, radial_length=5.0, merge_threshold=0.0
    )
    assert n_none == n_bins + 1
    assert edges_none[0] == dist.min()


def test_create_radial_bins_n_radial_no_merge():
    dist = np.concatenate([[0.0] * 2, [10.0] * 100, [20.0] * 100])
    _, n_bins, edges = create_radial_bins(dist, n_radial=4)
    assert n_bins == 4
    assert len(edges) == 5


def test_get_grid_from_atlas_threshold_and_1d_mode():
    atlas = Streamlines(_spectra_bundle())
    out = get_grid_from_atlas(atlas, n_segments=8, threshold=50.0)
    centroid, edges = out[2], out[4]
    assert len(centroid) == 8
    # n_segments without n_radial gives a single radial bin
    assert len(edges) - 1 == 1
    # radial_length is honoured when n_segments is not set
    out2 = get_grid_from_atlas(atlas, radial_length=2.0)
    assert len(out2[4]) - 1 > 1


def test_parameterize_bundle_count_limits():
    atlas = Streamlines(_spectra_bundle())
    _, _, centroid, rvec, edges, _, _ = get_grid_from_atlas(atlas, n_segments=6)
    # huge min_count masks every cell
    *_, valid, _ = parameterize_bundle(
        atlas, centroid, rvec, edges, min_count=10**9, max_count=10**9
    )
    assert not valid.any()
    # zero limits mask no cell; only points outside the radial edges (e.g.
    # exactly on the last edge or in merged-away bins) remain invalid
    *_, valid, _ = parameterize_bundle(
        atlas, centroid, rvec, edges, mask_threshold=0.0, min_count=0, max_count=0
    )
    assert valid.mean() > 0.95


def test_spectra_accepts_lists_and_ndarrays():
    lines = _spectra_bundle()
    metric = np.ones((90, 60, 60))
    ref = spectra_profile(
        Streamlines(lines), Streamlines(lines), Streamlines(lines), metric, np.eye(4)
    )
    from_lists = spectra_profile(lines, lines, lines, metric, np.eye(4))
    npt.assert_allclose(ref, from_lists, equal_nan=True)
    assign = spectra_assignment_map(lines, lines, n_segments=5)
    assert len(assign[0]) == sum(len(x) for x in lines)


def test_spectra_empty_bundle_raises():
    lines = _spectra_bundle()
    metric = np.ones((90, 60, 60))
    with pytest.raises(ValueError):
        spectra_profile([], lines, lines, metric, np.eye(4))
    with pytest.raises(ValueError):
        SpectraGrid(lines).profile([], lines, metric, np.eye(4))


def test_spectra_grid_class():
    lines = _spectra_bundle()
    metric = np.ones((90, 60, 60))
    grid = SpectraGrid(lines, segment_length=5.0, radial_length=5.0)
    assert grid.centroid.shape == (grid.n_segments, 3)
    assert grid.radial_vectors.shape == (grid.n_segments, 3)
    assert len(grid.radial_edges) == grid.n_radial + 1
    assert grid.segment_length_actual > 0
    assert grid.radial_length_actual > 0

    s_idx, r_idx, s_dist, valid, counts = grid.assign(lines)
    assert counts.shape == (grid.n_segments, grid.n_radial)
    assert len(s_idx) == len(r_idx) == len(s_dist) == len(valid)

    profile = grid.profile(lines, lines, metric, np.eye(4))
    assert profile.shape == (grid.n_segments, grid.n_radial)
    npt.assert_allclose(profile[np.isfinite(profile)], 1.0)

    # functional wrapper gives identical results
    func = spectra_profile(lines, lines, lines, metric, np.eye(4))
    npt.assert_allclose(profile, func, equal_nan=True)


def test_spectra_grid_profile_1d_adaptive_buan():
    lines = _spectra_bundle()
    grid = SpectraGrid(lines, n_segments=10, n_radial=3)

    # A constant metric gives a constant profile. The sparse end segments are
    # masked.
    metric = np.full((90, 60, 60), 0.4)
    profile = grid.profile_1d_adaptive_buan(lines, lines, metric, np.eye(4))
    npt.assert_allclose(profile[1:-1], 0.4)
    assert np.isnan(profile[[0, -1]]).all()

    # Weighted mean of each segment: points closer to the centroid count more
    rng = np.random.default_rng(0)
    metric = rng.random((90, 60, 60))
    profile = grid.profile_1d_adaptive_buan(
        lines, lines, metric, np.eye(4), mask_threshold=0, min_count=0
    )
    s_index, _, s_distance, _, _ = grid.assign(lines)
    values = np.concatenate(
        [metric[tuple(np.round(line).astype(int).T)] for line in lines]
    )
    for segment in (0, 4, 9):
        in_segment = s_index == segment
        weights = 1 / (s_distance[in_segment] + 1e-8)
        expected = np.sum(weights * values[in_segment]) / np.sum(weights)
        npt.assert_allclose(profile[segment], expected, rtol=0.2)
    assert profile.shape == (10,)

    # Segments with too few points are NaN
    profile = grid.profile_1d_adaptive_buan(
        lines, lines, metric, np.eye(4), min_count=10**6, max_count=10**6
    )
    assert np.isnan(profile).all()

    with pytest.raises(ValueError):
        grid.profile_1d_adaptive_buan([], lines, metric, np.eye(4))


def test_create_radial_bins_two_sparse_edge_bins():
    # Both bins fall below the merge threshold: only one may be merged away
    index, n_bins, edges = create_radial_bins(
        [0, 0, 1, 10, 10, 9], radial_length=5, merge_threshold=5
    )
    assert n_bins == 1
    npt.assert_array_equal(index, 0)
    npt.assert_array_equal(edges, [5, 10])


def test_parameterize_bundle_counts_are_raw_occupancy():
    lines = _spectra_bundle()
    atlas = Streamlines(lines)
    _, _, centroid, rvec, edges, _, _ = get_grid_from_atlas(
        atlas, n_segments=6, n_radial=4
    )
    edges = edges[1:-1]
    s_idx, r_idx, _, valid, counts = parameterize_bundle(
        atlas, centroid, rvec, edges, min_count=200, max_count=200
    )
    # some points are masked, but counts still hold every assigned point
    assert not valid.all()
    assert counts.sum() == len(s_idx)
    expected = np.bincount(
        s_idx * counts.shape[1] + r_idx, minlength=counts.size
    ).reshape(counts.shape)
    npt.assert_array_equal(counts, expected)


pd, have_pd, _ = optional_package("pandas")
_, have_smf, _ = optional_package("statsmodels.formula.api")
needs_statsmodels = pytest.mark.skipif(
    not (have_pd and have_smf), reason="Requires pandas and statsmodels"
)


def _correlated_residuals(n_segments, n_radial, n_subjects, length, *, seed=0):
    """Residuals with an exponential correlation of ``length`` cells."""
    rng = np.random.default_rng(seed)
    s = np.arange(n_segments)
    r = np.arange(n_radial)
    chol_s = np.linalg.cholesky(np.exp(-np.abs(s[:, None] - s[None, :]) / length))
    chol_r = np.linalg.cholesky(np.exp(-np.abs(r[:, None] - r[None, :]) / length))
    noise = rng.standard_normal((n_segments, n_radial, n_subjects))
    return np.stack(
        [chol_s @ noise[:, :, i] @ chol_r.T for i in range(n_subjects)], axis=2
    )


def _simulate_study(n_subjects=60, n_segments=12, n_radial=4, *, effect=0.03, seed=0):
    """Profiles of two groups of subjects from three sites."""
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {
            "group": np.repeat(["case", "control"], n_subjects // 2),
            "age": rng.normal(40, 10, n_subjects),
            "site": np.tile(["a", "b", "c"], n_subjects // 3),
        }
    )
    noise = _correlated_residuals(n_segments, n_radial, n_subjects, 2.0, seed=seed)
    profiles = 0.5 + 0.01 * noise.transpose(2, 0, 1)
    # Group difference in the first half of the segments
    is_case = data["group"].to_numpy() == "case"
    profiles[is_case, : n_segments // 2] += effect
    return profiles, data


def test_cohen_d():
    rng = np.random.default_rng(0)
    case = rng.normal(1.0, 1.0, 5000)
    control = rng.normal(0.0, 1.0, 5000)
    npt.assert_allclose(_cohen_d(case, control), 1.0, atol=0.05)
    npt.assert_almost_equal(_cohen_d(case, case), 0.0)
    assert np.isnan(_cohen_d(np.array([1.0]), control))
    assert np.isnan(_cohen_d(np.ones(3), np.ones(3)))


@needs_statsmodels
def test_fit_grid_lmm():
    profiles, data = _simulate_study()
    res = fit_grid_lmm(
        profiles,
        data,
        "group",
        covariates=["age"],
        groups="site",
        continuous=["age"],
        case_group="case",
        ref_group="control",
        fit_reduced=True,
    )

    for name in ("t", "p", "beta", "se", "dof", "ci_lower", "ci_upper", "d_cohen"):
        assert res[name].shape == (12, 4)
    assert res["mask"].all()
    npt.assert_array_equal(res["n_subjects"], 60)
    # 60 subjects, 3 fixed effects (intercept, group, age)
    npt.assert_array_equal(res["dof"], 57)
    assert res["residuals"].shape == (12, 4, 60)
    assert res["residuals_reduced"].shape == (12, 4, 60)

    # The effect is only in the first half of the segments
    assert np.all(res["p"][:6] < 0.01)
    assert np.all(res["t"][:6] > 0)
    assert np.all(res["d_cohen"][:6] > 1)
    assert np.mean(res["p"][6:] < 0.05) < 0.2

    assert np.all(res["ci_lower"] < res["beta"])
    assert np.all(res["beta"] < res["ci_upper"])
    npt.assert_allclose(res["r2_partial"], res["t"] ** 2 / (res["t"] ** 2 + res["dof"]))


@needs_statsmodels
def test_fit_grid_lmm_without_groups_is_ols():
    profiles, data = _simulate_study()
    res = fit_grid_lmm(profiles, data, "age", continuous=["age"])
    assert "d_cohen" not in res
    assert "residuals_reduced" not in res

    # Same slope as a least-squares fit of the standardized data
    y = (profiles[:, 3, 2] - profiles.mean()) / profiles.std()
    x = (data["age"] - data["age"].mean()) / data["age"].std(ddof=0)
    design = np.column_stack([np.ones(60), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0][1]
    npt.assert_allclose(res["beta"][3, 2], beta, rtol=1e-6)


@needs_statsmodels
def test_fit_grid_lmm_drops_missing_values():
    profiles, data = _simulate_study(seed=1)
    profiles[:15, 0, 0] = np.nan
    res = fit_grid_lmm(profiles, data, "age")

    # The missing values are dropped from the cell...
    assert res["n_subjects"][0, 0] == 45
    assert res["dof"][0, 0] == 43
    assert np.isnan(res["residuals"][0, 0, :15]).all()
    assert not np.isnan(res["residuals"][0, 0, 15:]).any()

    # ...which gives the same statistics as fitting the subjects with a value
    alone = fit_grid_lmm(
        profiles[15:, :3, :1],
        data.iloc[15:],
        "age",
        min_subjects=1,
        min_prop=0,
        min_component_size=1,
    )
    npt.assert_allclose(res["t"][0, 0], alone["t"][0, 0])
    npt.assert_allclose(res["p"][0, 0], alone["p"][0, 0])


@needs_statsmodels
def test_fit_grid_lmm_missing_values_are_not_zeros():
    # Treating the missing values as 0 (the bug fixed in SPECTRA) adds
    # outliers to the fit and hides the effect.
    profiles, data = _simulate_study(seed=2)
    rng = np.random.default_rng(3)
    profiles[rng.random(profiles.shape) < 0.25] = np.nan
    kwargs = {"case_group": "case", "ref_group": "control", "min_subjects": 10}

    res = fit_grid_lmm(profiles, data, "group", **kwargs)
    zero_filled = fit_grid_lmm(np.nan_to_num(profiles), data, "group", **kwargs)

    assert np.all(res["p"][:6] < 0.05)
    assert np.mean(zero_filled["t"][:6]) < 0.5 * np.mean(res["t"][:6])


@needs_statsmodels
def test_fit_grid_lmm_cell_selection():
    profiles, data = _simulate_study(seed=4)
    profiles[:, 0] = np.nan
    profiles[:40, 1] = np.nan

    res = fit_grid_lmm(profiles, data, "age", min_prop=0.5)
    assert not res["mask"][:2].any()
    assert res["mask"][2:].all()
    assert np.isnan(res["t"][:2]).all()

    mask = np.zeros((12, 4), dtype=bool)
    mask[5:7] = True
    res = fit_grid_lmm(profiles, data, "age", mask=mask)
    npt.assert_array_equal(res["mask"], mask)


@needs_statsmodels
def test_fit_grid_lmm_cells_like_spectra():
    # As in SPECTRA: small isolated regions of cells are not fitted, and the
    # cells without a fitted neighbor in both directions are not tested
    profiles, data = _simulate_study(n_segments=12, n_radial=4, seed=6)
    profiles[:, :, 1:] = np.nan
    profiles[:, 9, 0] = np.nan  # segments 10 and 11 form a region of 2 cells
    profiles[:, 2:6, 2] = 0.5  # a strip of 4 cells, one cell wide

    res = fit_grid_lmm(profiles, data, "age", min_component_size=3)
    assert not res["mask"][9:].any()
    assert np.isnan(res["residuals"][10:]).all()

    # Radial bin 1 is empty, so no cell has a fitted neighbor across the
    # radial bins and no cell is tested. The fitted cells keep their residuals.
    npt.assert_array_equal(res["mask"], np.zeros((12, 4), dtype=bool))
    assert not np.isnan(res["residuals"][0, 0]).any()

    # With a fitted neighbor across the radial bins, the cells are tested
    profiles[:, :, 1] = profiles[:, :, 0] + 0.01
    res = fit_grid_lmm(profiles, data, "age", min_component_size=3)
    assert res["mask"][:9, :2].all()


@needs_statsmodels
def test_fit_grid_lmm_1d_profiles():
    profiles, data = _simulate_study(n_radial=1)
    res = fit_grid_lmm(
        profiles[:, :, 0], data, "group", case_group="case", ref_group="control"
    )
    assert res["t"].shape == (12, 1)
    assert res["mask"].all()


@needs_statsmodels
def test_fit_grid_lmm_errors():
    profiles, data = _simulate_study()
    with pytest.raises(ValueError):
        fit_grid_lmm(profiles[:-1], data, "age")
    with pytest.raises(ValueError):
        fit_grid_lmm(profiles, data, "weight")
    with pytest.raises(ValueError):
        fit_grid_lmm(profiles, data, "group")
    with pytest.raises(ValueError):
        fit_grid_lmm(profiles, data, "group", case_group="case")
    with pytest.raises(ValueError):
        fit_grid_lmm(profiles, data, "group", case_group="case", ref_group="none")
    with pytest.raises(ValueError):
        fit_grid_lmm(profiles[0], data, "age")


def test_matern_correlation():
    d = np.array([0.0, 1.0, 2.0])
    npt.assert_allclose(matern_correlation(d, 2.0), np.exp(-d / 2.0))
    for nu in (0.5, 1.5, 2.5):
        rho = matern_correlation(d, 3.0, nu=nu)
        npt.assert_almost_equal(rho[0], 1.0)
        assert np.all(np.diff(rho) < 0)
    with pytest.raises(ValueError):
        matern_correlation(d, 1.0, nu=3.0)


def test_empirical_correlation():
    resid = _correlated_residuals(20, 8, 200, 3.0)
    lags, rho, n_pairs = empirical_correlation(resid, axis=0, max_lag=10)
    npt.assert_array_equal(lags, np.arange(11))
    npt.assert_almost_equal(rho[0], 1.0)
    assert n_pairs[0] == resid.size
    assert rho[1] > rho[5] > 0
    # The true correlation at lag 1 is exp(-1 / 3)
    npt.assert_allclose(rho[1], np.exp(-1 / 3), atol=0.05)

    lags, _, _ = empirical_correlation(resid, axis=1, max_lag=50)
    assert len(lags) == 8
    with pytest.raises(ValueError):
        empirical_correlation(resid, axis=2)


def test_empirical_correlation_ignores_nan():
    resid = _correlated_residuals(20, 8, 200, 3.0)
    expected = empirical_correlation(resid[:, :, :150])[1]
    resid[:, :, 150:] = np.nan
    npt.assert_allclose(empirical_correlation(resid)[1], expected)


def test_fit_matern():
    lags = np.arange(12)
    rho = np.exp(-lags * 2.0 / 6.0)
    n_pairs = np.full(12, 500)
    npt.assert_allclose(fit_matern(lags, rho, n_pairs, 2.0), 6.0, rtol=1e-2)
    with pytest.raises(ValueError):
        fit_matern(lags, rho, np.zeros(12), 2.0)


def _stats_grid(n_radial=10):
    """SPECTRA grid with 30 segments of the synthetic test bundle."""
    return SpectraGrid(_spectra_bundle(), n_segments=30, n_radial=n_radial)


def test_estimate_length_scales():
    grid = _stats_grid()
    short = _correlated_residuals(30, 10, 300, 1.5, seed=1)
    long = _correlated_residuals(30, 10, 300, 4.0, seed=1)
    l_s_short, l_r_short = estimate_length_scales(short, grid)
    l_s_long, l_r_long = estimate_length_scales(long, grid)
    assert l_s_short < l_s_long
    assert l_r_short < l_r_long

    # The true length scale of `long` is 4 cells, converted to mm with the
    # spacing of the grid
    npt.assert_allclose(l_s_long, 4 * grid.segment_length_actual, rtol=0.1)
    npt.assert_allclose(l_r_long, 4 * grid.radial_length_actual, rtol=0.1)

    grid_1d = _stats_grid(n_radial=None)
    l_s, l_r = estimate_length_scales(short[:, :1], grid_1d)
    assert l_s > 0
    assert l_r is None

    # 1D profiles along the segments of a 2D grid
    l_s_2d_grid, l_r = estimate_length_scales(short[:, :1], grid)
    npt.assert_allclose(
        l_s_2d_grid, l_s * grid.segment_length_actual / grid_1d.segment_length_actual
    )
    assert l_r is None

    with pytest.raises(ValueError):
        estimate_length_scales(short[:20], grid)


def test_length_scales_with_missing_values():
    # Missing residuals must be NaN. Setting them to 0 (the bug fixed in
    # SPECTRA) underestimates the radial length scale.
    grid = _stats_grid()
    resid = _correlated_residuals(30, 10, 300, 3.0, seed=2)
    missing = np.random.default_rng(3).random(resid.shape) < 0.3

    _, l_r = estimate_length_scales(resid, grid)
    _, l_r_nan = estimate_length_scales(np.where(missing, np.nan, resid), grid)
    _, l_r_zero = estimate_length_scales(np.where(missing, 0.0, resid), grid)

    npt.assert_allclose(l_r_nan, l_r, rtol=0.1)
    assert l_r_zero < 0.8 * l_r


def test_fdr_correction():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    npt.assert_allclose(fdr_correction(p), [0.02, 0.04, 0.04, 0.02])

    # NaN values are ignored and the shape is kept
    p_grid = np.array([[0.01, np.nan], [0.04, 0.03], [0.005, np.nan]])
    p_corr = fdr_correction(p_grid)
    assert p_corr.shape == (3, 2)
    assert np.isnan(p_corr[:, 1][[0, 2]]).all()
    npt.assert_allclose(p_corr[~np.isnan(p_grid)], [0.02, 0.04, 0.04, 0.02])
    assert np.isnan(fdr_correction([np.nan])).all()


def test_fdr_correction_adaptive():
    rng = np.random.default_rng(1)
    p = np.concatenate([rng.uniform(size=300), rng.uniform(0, 1e-3, size=100)])
    p_bh = fdr_correction(p)
    p_adaptive = fdr_correction(p, adaptive=True)
    assert np.sum(p_adaptive < 0.05) >= np.sum(p_bh < 0.05)
    assert np.all(p_adaptive <= p_bh + 1e-12)
    assert np.all(p_adaptive >= p)


def test_storey_pi0():
    rng = np.random.default_rng(2)
    assert 0.8 < _storey_pi0(rng.uniform(size=2000)) <= 1.0
    mixed = np.concatenate([rng.uniform(size=1000), np.zeros(1000)])
    assert _storey_pi0(mixed) < 0.7


def test_simes_combine():
    p = np.array([0.01, 0.04, 0.5, 0.9])
    z = stats.norm.isf(p / 2) * np.array([-1, 1, 1, 1])
    z_block, p_block = simes_combine(p, z)
    npt.assert_almost_equal(p_block, 0.04)
    npt.assert_almost_equal(z_block, -stats.norm.isf(0.02))
    assert simes_combine([0.3], [1.0])[1] == 0.3


def test_stouffer_combine():
    z, p = stouffer_combine([2.0, 2.0, 2.0, 2.0])
    npt.assert_almost_equal(z, 4.0)
    npt.assert_almost_equal(p, 2 * stats.norm.sf(4.0))
    # Fully correlated cells bring no more evidence than a single cell
    z, _ = stouffer_combine([2.0, 2.0, 2.0, 2.0], correlation=np.ones((4, 4)))
    npt.assert_almost_equal(z, 2.0)


def test_conditional_pvalue():
    # A cell independent of its block keeps its p-value
    npt.assert_almost_equal(conditional_pvalue(2.0, 0.05, 0.0), 2 * stats.norm.sf(2))
    # A correlated cell is corrected for the block selection
    p = conditional_pvalue(2.0, 0.05, 0.7)
    assert 2 * stats.norm.sf(2) < p <= 1.0
    npt.assert_almost_equal(conditional_pvalue(-2.0, 0.05, 0.7), p)


def test_partition_bundle_into_blocks():
    grid = _stats_grid()
    mask = np.ones((30, 10), dtype=bool)
    # Length scales of 2 cells give blocks of round(2.355 * 2) = 5 cells
    l_s = 2 * grid.segment_length_actual
    l_r = 2 * grid.radial_length_actual

    blocks = partition_bundle_into_blocks(mask, grid, l_s, l_r=l_r)
    assert len(blocks) == 12
    assert all(len(cells) == 25 for cells in blocks)
    assert sorted(c for cells in blocks for c in cells) == list(zip(*np.nonzero(mask)))

    # Without a radial length scale, a block spans all the radial bins
    blocks = partition_bundle_into_blocks(mask, grid, l_s)
    assert len(blocks) == 6
    assert all(len(cells) == 50 for cells in blocks)

    # Small blocks are merged with a neighbor
    mask = np.zeros((30, 10), dtype=bool)
    mask[:11, 0] = True
    blocks = partition_bundle_into_blocks(mask, grid, l_s)
    assert [len(cells) for cells in blocks] == [5, 6]

    assert partition_bundle_into_blocks(np.zeros((30, 10)), grid, l_s) == []


def _bundle_statistics(grid, effect, *, seed=0):
    """Cell statistics on ``grid``, with length scales of 2 cells."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((grid.n_segments, grid.n_radial))
    if effect:
        z[8:18, 2:6] += 4.5
    return {
        "grid": grid,
        "t": z,
        "p": 2 * stats.norm.sf(np.abs(z)),
        "l_s": 2 * grid.segment_length_actual,
        "l_r": 2 * grid.radial_length_actual,
    }


@pytest.mark.parametrize("method", ["simes", "stouffer"])
def test_hierarchical_fdr(method):
    grid = _stats_grid()
    bundles = {
        "effect": _bundle_statistics(grid, True),
        "null": _bundle_statistics(grid, False, seed=1),
    }
    p_hfdr, blocks = hierarchical_fdr(bundles, combination_method=method)

    # 2 bundles with 6 x 2 blocks of 5 x 5 cells
    assert len(blocks) == 24
    rejected = [block for block in blocks if block["significant"]]
    assert 0 < len(rejected) < 12
    assert all(block["bundle"] == "effect" for block in rejected)
    assert all(block["p_cond"] is None for block in blocks if not block["significant"])

    significant = p_hfdr["effect"] < 0.15
    assert significant[10:16, 3:5].mean() > 0.9
    assert significant[:4].sum() < 5
    assert np.isnan(p_hfdr["null"]).all()


def test_hierarchical_fdr_single_block():
    bundle = _bundle_statistics(_stats_grid(), True)
    bundle["p"][:, :2] = np.nan

    _, blocks = hierarchical_fdr({"g": bundle}, use_resel_blocks=False)
    assert len(blocks) == 1
    assert len(blocks[0]["cells"]) == 30 * 8

    # A bundle without length scales is a single block
    del bundle["l_s"], bundle["l_r"]
    _, blocks = hierarchical_fdr({"g": bundle})
    assert len(blocks) == 1


def test_hierarchical_fdr_1d_profiles():
    # 1D statistics along the segments of a 2D grid
    grid = _stats_grid()
    bundle = _bundle_statistics(grid, False)
    z = np.random.default_rng(4).standard_normal((30, 1))
    z[8:18] += 5
    bundle.update(t=z, p=2 * stats.norm.sf(np.abs(z)), l_r=None)
    p_hfdr, blocks = hierarchical_fdr({"g": bundle})
    assert p_hfdr["g"].shape == (30, 1)
    assert all(len(block["cells"]) == 5 for block in blocks)
    # The blocks of segments 5 to 19 contain the effect
    assert [block["significant"] for block in blocks] == [0, 1, 1, 1, 0, 0]
    assert (p_hfdr["g"][8:18] < 0.15).mean() > 0.7
    assert np.isnan(p_hfdr["g"][20:]).all()


def test_hierarchical_fdr_errors():
    bundle = _bundle_statistics(_stats_grid(), True)
    with pytest.raises(ValueError):
        hierarchical_fdr({"g": bundle}, combination_method="fisher")
    # Statistics computed on a different grid
    with pytest.raises(ValueError):
        hierarchical_fdr({"g": dict(bundle, grid=_stats_grid(n_radial=None))})
    bundle["p"][:] = np.nan
    with pytest.raises(ValueError):
        hierarchical_fdr({"g": bundle})


@needs_statsmodels
def test_spectra_statistics_pipeline():
    # From the SPECTRA grid of a bundle to the significant cells of a group
    # comparison. Each subject has its own metric volume, with smooth noise,
    # and the cases have a lower metric in the first part of the bundle.
    rng = np.random.default_rng(0)
    lines = _spectra_bundle()
    grid = SpectraGrid(lines, n_segments=20, n_radial=5)
    n_subjects = 90
    data = pd.DataFrame(
        {
            "group": np.repeat(["case", "control"], n_subjects // 2),
            "site": np.tile(["a", "b", "c"], n_subjects // 3),
        }
    )

    profiles = []
    for group in data["group"]:
        noise = gaussian_filter(rng.standard_normal((90, 60, 60)), sigma=3)
        metric = 0.5 + 0.02 * noise / noise.std()
        if group == "case":
            metric[:45] -= 0.03
        profiles.append(grid.profile(lines, lines, metric, np.eye(4)))
    profiles = np.array(profiles)

    fit = fit_grid_lmm(
        profiles,
        data,
        "group",
        groups="site",
        case_group="case",
        ref_group="control",
        fit_reduced=True,
    )
    l_s, l_r = estimate_length_scales(fit["residuals_reduced"], grid)
    bundle = {"grid": grid, "t": fit["t"], "p": fit["p"], "l_s": l_s, "l_r": l_r}
    p_hfdr, _ = hierarchical_fdr({"bundle": bundle})

    # The bundle runs from x = 25 to x = 75, the effect is at x < 45
    significant = p_hfdr["bundle"] < 0.15
    assert significant[:6][fit["mask"][:6]].mean() > 0.8
    assert significant[12:][fit["mask"][12:]].mean() < 0.1
