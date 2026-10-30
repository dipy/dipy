from pathlib import Path
import warnings

import numpy as np
from scipy import integrate, stats
from scipy.interpolate import UnivariateSpline, splev, splprep
from scipy.ndimage import gaussian_filter1d, label, map_coordinates
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import mahalanobis

from dipy.io.utils import save_buan_profiles_hdf5
from dipy.segment.clustering import QuickBundles
from dipy.segment.metricspeed import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import (
    Streamlines,
    length,
    orient_by_streamline,
    set_number_of_points,
    transform_streamlines,
    values_from_volume,
)
from dipy.utils.deprecator import warning_for_keywords
from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package

pd, have_pd, _ = optional_package("pandas")
smf, have_smf, _ = optional_package("statsmodels.formula.api")


def peak_values(bundle, peaks, dt, pname, bname, subject, group_id, ind, dir_name):
    """Peak_values function finds the generalized fractional anisotropy (gfa)
        and quantitative anisotropy (qa) values from peaks object (eg: csa) for
        every point on a streamline used while tracking and saves it in hd5
        file.

    Parameters
    ----------
    bundle : string
        Name of bundle being analyzed
    peaks : peaks
        contains peak directions and values
    dt : DataFrame
        DataFrame to be populated
    pname : string
        Name of the dti metric
    bname : string
        Name of bundle being analyzed.
    subject : string
        subject number as a string (e.g. 10001)
    group_id : integer
        which group subject belongs to 1 patient and 0 for control
    ind : integer list
        ind tells which disk number a point belong.
    dir_name : string
        path of output directory

    """

    gfa = peaks.gfa
    anatomical_measures(
        bundle, gfa, dt, pname + "_gfa", bname, subject, group_id, ind, dir_name
    )

    qa = peaks.qa[..., 0]
    anatomical_measures(
        bundle, qa, dt, pname + "_qa", bname, subject, group_id, ind, dir_name
    )


def anatomical_measures(
    bundle, metric, dt, pname, bname, subject, group_id, ind, dir_name
):
    """Calculates dti measure (eg: FA, MD) per point on streamlines and
        save it in hd5 file.

    Parameters
    ----------
    bundle : string
        Name of bundle being analyzed
    metric : matrix of float values
        dti metric e.g. FA, MD
    dt : DataFrame
        DataFrame to be populated
    pname : string
        Name of the dti metric
    bname : string
        Name of bundle being analyzed.
    subject : string
        subject number as a string (e.g. 10001)
    group_id : integer
        which group subject belongs to 1 for patient and 0 control
    ind : integer list
        ind tells which disk number a point belong.
    dir_name : string
        path of output directory

    """
    dt["streamline"] = []
    dt["disk"] = []
    dt["subject"] = []
    dt[pname] = []
    dt["group"] = []

    values = map_coordinates(metric, bundle._data.T, order=1)

    dt["disk"].extend(ind[list(range(len(values)))] + 1)
    dt["subject"].extend([subject] * len(values))
    dt["group"].extend([group_id] * len(values))
    dt[pname].extend(values)

    for st_i in range(len(bundle)):
        st = bundle[st_i]
        dt["streamline"].extend([st_i] * len(st))

    file_name = f"{bname}_{pname}"

    save_buan_profiles_hdf5(Path(dir_name) / file_name, dt)


def assignment_map(target_bundle, model_bundle, no_disks):
    """
    Calculates assignment maps of the target bundle with reference to
    model bundle centroids.

    See :footcite:p:`Chandio2020a` for further details about the method.

    Parameters
    ----------
    target_bundle : Streamlines
        target bundle extracted from subject data in common space
    model_bundle : Streamlines
        atlas bundle used as reference
    no_disks : integer, optional
        Number of disks used for dividing bundle into disks.

    Returns
    -------
    dist : ndarray
        Distance of each target bundle point to its nearest model bundle
        centroid point.
    indx : ndarray
        Assignment map of the target bundle streamline point indices to the
        model bundle centroid points.

    References
    ----------
    .. footbibliography::

    """

    mbundle_streamlines = set_number_of_points(model_bundle, nb_points=no_disks)

    metric = AveragePointwiseEuclideanMetric()
    qb = QuickBundles(threshold=85.0, metric=metric)
    clusters = qb.cluster(mbundle_streamlines)
    centroids = Streamlines(clusters.centroids)

    dist, indx = cKDTree(centroids.get_data(), 1, copy_data=True).query(
        target_bundle.get_data(), k=1
    )

    return dist, indx


def buan_profile(model_bundle, bundle, orig_bundle, metric, affine, *, no_disks=100):
    """
    Create BUAN weighted mean bundle profiles (lite).

    See :footcite:p:`Chandio2020a` and :footcite:p:`chandio2024bundle`
    for further details about the method.

    Parameters
    ----------
    model_bundle : Streamlines
        The atlas/template bundle used as the along-tract reference.
        Must be in the same space as ``bundle`` (common/MNI space).
    bundle : Streamlines
        The subject bundle in common space (e.g., MNI). Used for segment
        assignment against the model centroids.
    orig_bundle : Streamlines
        The same subject bundle in native/world (RAS) space. Used for
        sampling the metric volume. Must correspond point-for-point to
        ``bundle``.
    metric : ndarray
        3-D scalar volume (e.g., FA) in the same voxel space as ``affine``.
    affine : ndarray
        Voxel-to-world affine of the metric volume (as returned by
        ``nib.load(...).affine``). Used to convert ``orig_bundle`` from
        world to voxel coordinates for metric interpolation.
    no_disks : int, optional
        Number of alongtract segments/disks used for dividing bundle into
        segments.

    Returns
    -------
    bundle_profile : ndarray, shape (no_disks,)
        Inverse-distance-weighted mean metric value for each disk segment.
        Disks with no valid data points are set to NaN.

    References
    ----------
    .. footbibliography::

    """

    if len(model_bundle) == 0 or len(bundle) == 0 or len(orig_bundle) == 0:
        raise ValueError("One of the bundles contains no streamlines")

    dist, indx = assignment_map(bundle, model_bundle, no_disks)
    ind = np.array(indx)
    affine_r = np.linalg.inv(affine)
    transformed_orig_bundle = transform_streamlines(orig_bundle, affine_r)
    bundle_profile = np.zeros(no_disks)
    values = map_coordinates(metric, transformed_orig_bundle._data.T, order=1)

    epsilon = 1e-8
    weights = 1 / (dist + epsilon)

    for i in range(no_disks):
        valid_mask = ind == i
        valid_mask &= ~np.isnan(values)

        if np.any(valid_mask):
            vals = values[valid_mask]
            wts = weights[valid_mask]
            wts /= np.sum(wts)
            weighted_mean = np.sum(wts * vals)

        else:
            weighted_mean = np.nan

        bundle_profile[i] = weighted_mean

    return bundle_profile


def _as_streamlines(bundle):
    """Return ``bundle`` as a Streamlines object."""
    if isinstance(bundle, Streamlines):
        return bundle
    return Streamlines(bundle)


def get_centroid(bundle, *, n_points=50, threshold=100.0):
    """Compute a representative QuickBundles centroid for a bundle.

    Parameters
    ----------
    bundle : Streamlines
        Input streamline bundle.
    n_points : int, optional
        Number of points used to resample streamlines before clustering.
    threshold : float, optional
        QuickBundles clustering threshold.

    Returns
    -------
    centroid : ndarray, shape (n_points, 3)
        Representative centroid streamline.

    """
    bundle = _as_streamlines(bundle)
    if len(bundle) == 0:
        raise ValueError("Bundle contains no streamlines")

    resampled = set_number_of_points(bundle, nb_points=n_points)
    metric = AveragePointwiseEuclideanMetric()
    qb = QuickBundles(threshold=threshold, metric=metric)
    clusters = qb.cluster(resampled)
    centroids = Streamlines(clusters.centroids)

    if len(centroids) == 0:
        raise ValueError("QuickBundles did not generate a centroid")

    return centroids[np.argmax(length(centroids))]


def get_n_segment_by_length(bundle, *, segment_length=5.0):
    """Estimate the number of along-tract segments from bundle length.

    Parameters
    ----------
    bundle : Streamlines
        Input streamline bundle.
    segment_length : float, optional
        Desired segment length in millimeters.

    Returns
    -------
    n_segments : int
        Number of along-tract segments.

    """
    if segment_length <= 0:
        raise ValueError("segment_length must be greater than zero")

    bundle_lengths = length(bundle)
    if bundle_lengths.size == 0:
        raise ValueError("Bundle contains no streamlines")

    return max(int(np.round(np.mean(bundle_lengths) / segment_length)), 1)


def _resample_curve_by_arclength(points, n_points):
    """Resample a 3D curve uniformly by arc length."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        raise ValueError("At least two points are required")

    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    total_length = cumulative[-1]

    if total_length == 0:
        return np.repeat(points[:1], n_points, axis=0)

    target = np.linspace(0.0, total_length, n_points)
    return np.column_stack(
        [np.interp(target, cumulative, points[:, dim]) for dim in range(3)]
    )


def compute_robust_centroid(
    bundle,
    *,
    segment_length=5.0,
    n_segments=None,
    threshold=100.0,
    extrapolate_prop=0.3,
    method="linear",
):
    """Compute a centroid that covers the full extent of a bundle.

    Parameters
    ----------
    bundle : Streamlines
        Atlas or model bundle.
    segment_length : float, optional
        Desired along-tract segment length in millimeters.
    n_segments : int, optional
        Number of along-tract segments. If provided, this takes precedence
        over ``segment_length``.
    threshold : float, optional
        QuickBundles threshold used to estimate the initial centroid.
    extrapolate_prop : float, optional
        Maximum fraction of the centroid length used for endpoint extension.
    method : {"linear", "spline"}, optional
        Endpoint extension method.

    Returns
    -------
    centroid : ndarray, shape (n_segments, 3)
        Centroid sampled uniformly by arc length.

    """
    bundle = _as_streamlines(bundle)
    if len(bundle) == 0:
        raise ValueError("Bundle contains no streamlines")
    if method not in {"linear", "spline"}:
        raise ValueError("method must be either 'linear' or 'spline'")

    centroid = get_centroid(bundle, n_points=100, threshold=threshold)
    all_points = bundle.get_data()

    start_tangent = centroid[1] - centroid[0]
    end_tangent = centroid[-1] - centroid[-2]
    start_norm = np.linalg.norm(start_tangent)
    end_norm = np.linalg.norm(end_tangent)

    if start_norm == 0 or end_norm == 0:
        if n_segments is None:
            n_segments = get_n_segment_by_length(bundle, segment_length=segment_length)
        return _resample_curve_by_arclength(centroid, n_segments)

    start_tangent /= start_norm
    end_tangent /= end_norm

    start_projection = (all_points - centroid[0]) @ start_tangent
    end_projection = (all_points - centroid[-1]) @ end_tangent
    start_extension = max(0.0, -np.min(start_projection))
    end_extension = max(0.0, np.max(end_projection))

    centroid_length = np.sum(np.linalg.norm(np.diff(centroid, axis=0), axis=1))
    max_extension = extrapolate_prop * centroid_length
    start_extension = min(start_extension, max_extension)
    end_extension = min(end_extension, max_extension)

    if method == "linear":
        start_point = centroid[0] - start_tangent * start_extension
        end_point = centroid[-1] + end_tangent * end_extension
        extended = np.vstack((start_point, centroid, end_point))
    else:
        tck, _ = splprep(centroid.T, s=0, k=min(3, len(centroid) - 1))
        start_fraction = start_extension / centroid_length
        end_fraction = end_extension / centroid_length
        t_extended = np.linspace(-start_fraction, 1.0 + end_fraction, 500)
        extended = np.asarray(splev(t_extended, tck)).T

    extended_length = np.sum(np.linalg.norm(np.diff(extended, axis=0), axis=1))
    if n_segments is None:
        n_segments = max(int(np.round(extended_length / segment_length)), 1)

    return _resample_curve_by_arclength(extended, n_segments)


def create_radial_bins(
    radial_distance, *, radial_length=5.0, n_radial=None, merge_threshold=0.2
):
    """Create bins from signed radial distances.

    Parameters
    ----------
    radial_distance : ndarray, shape (n_points,)
        Signed radial position of each atlas point.
    radial_length : float, optional
        Desired radial-bin width in millimeters.
    n_radial : int, optional
        Number of radial bins. If provided, automatic edge-bin merging is
        disabled.
    merge_threshold : float, optional
        Edge bins holding fewer points than ``merge_threshold`` times the
        median count of non-empty bins are merged into their neighbor.

    Returns
    -------
    radial_index : ndarray
        Radial-bin index for each point.
    n_radial : int
        Final number of radial bins.
    radial_edges : ndarray
        Radial-bin boundaries.

    """
    radial_distance = np.asarray(radial_distance, dtype=float)
    if radial_distance.size == 0:
        raise ValueError("radial_distance cannot be empty")

    r_min = np.min(radial_distance)
    r_max = np.max(radial_distance)

    if n_radial is not None:
        n_bins = max(int(n_radial), 1)
    else:
        if radial_length <= 0:
            raise ValueError("radial_length must be greater than zero")
        n_bins = max(int(np.ceil((r_max - r_min) / radial_length)), 1)

    radial_edges = np.linspace(r_min, r_max, n_bins + 1)
    radial_index = np.digitize(radial_distance, radial_edges) - 1
    radial_index = np.clip(radial_index, 0, n_bins - 1)

    if n_radial is None and n_bins > 1:
        counts = np.bincount(radial_index, minlength=n_bins)
        nonzero = counts[counts > 0]
        threshold = np.median(nonzero) * merge_threshold
        remove_first = counts[0] < threshold
        remove_last = counts[-1] < threshold

        if n_bins == 2 and remove_first and remove_last:
            # Both bins are edge bins: merge only the sparser one, otherwise
            # no bin would remain.
            remove_last = counts[-1] < counts[0]
            remove_first = not remove_last

        if remove_first:
            radial_index[radial_index == 0] = 1
            radial_edges = radial_edges[1:]

        if remove_last:
            last_index = n_bins - 1
            radial_index[radial_index == last_index] = last_index - 1
            radial_edges = radial_edges[:-1]

        if remove_first or remove_last:
            n_bins = len(radial_edges) - 1
            radial_index = np.digitize(radial_distance, radial_edges) - 1
            radial_index = np.clip(radial_index, 0, n_bins - 1)

    return radial_index, len(radial_edges) - 1, radial_edges


def get_grid_from_atlas(
    atlas_bundle,
    *,
    segment_length=5.0,
    radial_length=5.0,
    n_segments=None,
    n_radial=None,
    threshold=100.0,
    use_robust_centroid=True,
    robust_method="linear",
):
    """Construct a SPECTRA grid from an atlas bundle.

    Parameters
    ----------
    atlas_bundle : Streamlines
        Atlas or template bundle used to define the SPECTRA grid.
    segment_length : float, optional
        Desired along-tract segment length in millimeters.
    radial_length : float, optional
        Desired radial-bin width in millimeters.
    n_segments : int, optional
        Number of along-tract segments.
    n_radial : int, optional
        Number of radial bins. If ``n_segments`` is given without
        ``n_radial``, a single radial bin is used (1D profile).
    threshold : float, optional
        QuickBundles threshold used to estimate the atlas centroid.
    use_robust_centroid : bool, optional
        If True, use an extended centroid that covers the bundle endpoints.
    robust_method : {"linear", "spline"}, optional
        Endpoint extension method used by ``compute_robust_centroid``.

    Returns
    -------
    s_index : ndarray
        Along-tract assignment for each atlas point.
    r_index : ndarray
        Radial assignment for each atlas point.
    centroid : ndarray
        Atlas centroid.
    radial_vectors : ndarray
        Radial direction at each along-tract segment.
    radial_edges : ndarray
        Radial-bin boundaries.
    segment_length_actual : float
        Mean spacing between neighboring centroid points.
    radial_length_actual : float
        Mean radial-bin width.

    """
    atlas_bundle = _as_streamlines(atlas_bundle)
    if len(atlas_bundle) == 0:
        raise ValueError("Atlas bundle contains no streamlines")

    # 1D mode: a fixed number of segments without a radial size means one bin
    if n_segments is not None and n_radial is None:
        n_radial = 1

    if use_robust_centroid:
        centroid = compute_robust_centroid(
            atlas_bundle,
            segment_length=segment_length,
            n_segments=n_segments,
            threshold=threshold,
            method=robust_method,
        )
    else:
        if n_segments is None:
            n_segments = get_n_segment_by_length(
                atlas_bundle, segment_length=segment_length
            )
        centroid = get_centroid(atlas_bundle, n_points=n_segments, threshold=threshold)

    n_segments_actual = len(centroid)
    atlas_points = atlas_bundle.get_data()
    tree = cKDTree(centroid, copy_data=True)
    _, s_index = tree.query(atlas_points, k=1)

    centroid_length = np.sum(np.linalg.norm(np.diff(centroid, axis=0), axis=1))
    segment_length_actual = centroid_length / max(n_segments_actual - 1, 1)

    tangents = np.gradient(centroid, axis=0)
    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangent_norms[tangent_norms == 0] = 1.0
    tangents /= tangent_norms

    radial_vectors = np.zeros((n_segments_actual, 3), dtype=float)
    for segment in range(n_segments_actual):
        points = atlas_points[s_index == segment]
        if len(points) < 2:
            continue

        delta = points - centroid[segment]
        tangent = tangents[segment]
        projected = delta - np.outer(delta @ tangent, tangent)
        covariance = projected.T @ projected
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        radial_vectors[segment] = eigenvectors[:, np.argmax(eigenvalues)]

    radial_norms = np.linalg.norm(radial_vectors, axis=1)
    valid = radial_norms > 0
    if not np.any(valid):
        raise ValueError("Unable to determine radial directions from atlas geometry")

    valid_indices = np.flatnonzero(valid)
    for segment in range(n_segments_actual):
        if valid[segment]:
            continue
        nearest = valid_indices[np.argmin(np.abs(valid_indices - segment))]
        radial_vectors[segment] = radial_vectors[nearest]

    for segment in range(1, n_segments_actual):
        if np.dot(radial_vectors[segment], radial_vectors[segment - 1]) < 0:
            radial_vectors[segment] *= -1

    radial_vectors = gaussian_filter1d(radial_vectors, sigma=1, axis=0)
    radial_norms = np.linalg.norm(radial_vectors, axis=1, keepdims=True)
    radial_norms[radial_norms == 0] = 1.0
    radial_vectors /= radial_norms

    delta = atlas_points - centroid[s_index]
    radial_distance = np.sum(delta * radial_vectors[s_index], axis=1)
    r_index, n_radial_actual, radial_edges = create_radial_bins(
        radial_distance, radial_length=radial_length, n_radial=n_radial
    )
    radial_length_actual = (radial_edges[-1] - radial_edges[0]) / max(
        n_radial_actual, 1
    )

    return (
        s_index,
        r_index,
        centroid,
        radial_vectors,
        radial_edges,
        segment_length_actual,
        radial_length_actual,
    )


def parameterize_bundle(
    bundle,
    centroid,
    radial_vectors,
    radial_edges,
    *,
    mask_threshold=0.2,
    min_count=50,
    max_count=200,
):
    """Map a bundle onto a SPECTRA grid.

    Parameters
    ----------
    bundle : Streamlines
        Subject bundle in the same space as the atlas bundle.
    centroid : ndarray, shape (n_segments, 3)
        Atlas centroid.
    radial_vectors : ndarray, shape (n_segments, 3)
        Atlas radial directions.
    radial_edges : ndarray
        Atlas radial-bin boundaries.
    mask_threshold : float, optional
        Fraction of the median non-empty cell count below which a cell is
        considered sparse.
    min_count : int, optional
        Lower bound of the sparse-cell count cutoff.
    max_count : int, optional
        Upper bound of the sparse-cell count cutoff.

    Returns
    -------
    s_index : ndarray
        Along-tract assignment for each bundle point.
    r_index : ndarray
        Radial assignment for each bundle point.
    s_distance : ndarray
        Distance from each bundle point to the nearest centroid point.
    valid_mask : ndarray
        Boolean mask identifying points retained for profiling.
    counts : ndarray
        Number of points in each SPECTRA grid cell.

    """
    bundle = _as_streamlines(bundle)
    if len(bundle) == 0:
        raise ValueError("Bundle contains no streamlines")

    points = bundle.get_data()
    n_segments = len(centroid)
    n_radial = len(radial_edges) - 1

    tree = cKDTree(centroid, copy_data=True)
    s_distance, s_index = tree.query(points, k=1)

    delta = points - centroid[s_index]
    radial_distance = np.sum(delta * radial_vectors[s_index], axis=1)
    r_index = np.digitize(radial_distance, radial_edges) - 1

    valid_mask = (r_index >= 0) & (r_index < n_radial)
    r_index = np.clip(r_index, 0, n_radial - 1)

    bin_ids = s_index * n_radial + r_index
    counts_flat = np.bincount(bin_ids, minlength=n_segments * n_radial)
    nonzero = counts_flat[counts_flat > 0]

    if nonzero.size:
        sparse_threshold = np.clip(
            np.median(nonzero) * mask_threshold, min_count, max_count
        )
        sparse_bins = np.flatnonzero(counts_flat < sparse_threshold)
        if sparse_bins.size:
            valid_mask[np.isin(bin_ids, sparse_bins)] = False

    counts = counts_flat.reshape(n_segments, n_radial)
    return s_index, r_index, s_distance, valid_mask, counts


class SpectraGrid:
    """SPECTRA grid defined by an atlas bundle.

    See :footcite:p:`Feng2026SPECTRA` for further details about the method.

    The grid is built once from the atlas bundle and can then be applied to
    any number of subject bundles, so users do not have to carry the centroid,
    radial vectors and radial edges between function calls.

    Parameters
    ----------
    atlas_bundle : Streamlines
        Atlas or template bundle used to define the grid.
    segment_length : float, optional
        Desired along-tract segment length in millimeters.
    radial_length : float, optional
        Desired radial-bin width in millimeters.
    n_segments : int, optional
        Number of along-tract segments.
    n_radial : int, optional
        Number of radial bins. If ``n_segments`` is given without
        ``n_radial``, a single radial bin is used (1D profile).
    threshold : float, optional
        QuickBundles threshold used to estimate the atlas centroid.
    use_robust_centroid : bool, optional
        If True, use an extended centroid that covers the bundle endpoints.
    robust_method : {"linear", "spline"}, optional
        Endpoint extension method used by ``compute_robust_centroid``.

    Attributes
    ----------
    centroid : ndarray, shape (n_segments, 3)
        Atlas centroid.
    radial_vectors : ndarray, shape (n_segments, 3)
        Radial direction at each along-tract segment.
    radial_edges : ndarray, shape (n_radial + 1,)
        Radial-bin boundaries.
    n_segments : int
        Number of along-tract segments.
    n_radial : int
        Number of radial bins.
    segment_length_actual : float
        Mean spacing between neighboring centroid points.
    radial_length_actual : float
        Mean radial-bin width.

    References
    ----------
    .. footbibliography::

    """

    def __init__(
        self,
        atlas_bundle,
        *,
        segment_length=5.0,
        radial_length=5.0,
        n_segments=None,
        n_radial=None,
        threshold=100.0,
        use_robust_centroid=True,
        robust_method="linear",
    ):
        (
            _,
            _,
            self.centroid,
            self.radial_vectors,
            self.radial_edges,
            self.segment_length_actual,
            self.radial_length_actual,
        ) = get_grid_from_atlas(
            atlas_bundle,
            segment_length=segment_length,
            radial_length=radial_length,
            n_segments=n_segments,
            n_radial=n_radial,
            threshold=threshold,
            use_robust_centroid=use_robust_centroid,
            robust_method=robust_method,
        )
        self.n_segments = len(self.centroid)
        self.n_radial = len(self.radial_edges) - 1

    def assign(self, bundle, **kwargs):
        """Assign bundle points to the grid.

        Parameters
        ----------
        bundle : Streamlines
            Subject bundle in the same space as the atlas bundle.
        **kwargs : dict, optional
            Sparse-cell masking options passed to ``parameterize_bundle``
            (``mask_threshold``, ``min_count``, ``max_count``).

        Returns
        -------
        s_index, r_index, s_distance, valid_mask, counts
            See ``parameterize_bundle``.

        """
        return parameterize_bundle(
            bundle, self.centroid, self.radial_vectors, self.radial_edges, **kwargs
        )

    def profile(self, bundle, orig_bundle, metric, affine, **kwargs):
        """Create a two-dimensional SPECTRA profile of a subject bundle.

        Parameters
        ----------
        bundle : Streamlines
            Subject bundle in the atlas space. Used for grid assignment.
        orig_bundle : Streamlines
            Corresponding subject bundle in native/world space. Must
            correspond point-for-point to ``bundle``.
        metric : ndarray
            3D scalar volume sampled along ``orig_bundle``.
        affine : ndarray, shape (4, 4)
            Voxel-to-world affine of ``metric``.
        **kwargs : dict, optional
            Sparse-cell masking options passed to ``parameterize_bundle``.

        Returns
        -------
        profile : ndarray, shape (n_segments, n_radial)
            Mean metric value in each grid cell.

        """
        bundle = _as_streamlines(bundle)
        orig_bundle = _as_streamlines(orig_bundle)
        if len(bundle) == 0 or len(orig_bundle) == 0:
            raise ValueError("One of the bundles contains no streamlines")

        s_index, r_index, _, valid_mask, _ = self.assign(bundle, **kwargs)
        return grid_profile(
            orig_bundle,
            s_index,
            r_index,
            valid_mask,
            self.n_segments,
            self.n_radial,
            metric,
            affine,
        )

    def profile_1d_adaptive_buan(
        self,
        bundle,
        orig_bundle,
        metric,
        affine,
        *,
        mask_threshold=0.2,
        min_count=50,
        max_count=200,
    ):
        """Create a one-dimensional profile of a subject bundle, adapted from BUAN.

        The value of a segment is the mean of the metric at its points,
        weighted by the inverse of their distance to the centroid, as in
        :func:`dipy.stats.analysis.buan_profile`. The segments are the ones of
        this grid, so they are the same as in :meth:`profile`, instead of the
        ones that ``buan_profile`` builds from a model bundle. Segments with
        too few points are NaN.

        See :footcite:p:`Feng2026SPECTRA` for further details about the method.

        Parameters
        ----------
        bundle : Streamlines
            Subject bundle in the atlas space. Used for grid assignment.
        orig_bundle : Streamlines
            Corresponding subject bundle in native/world space. Must
            correspond point-for-point to ``bundle``.
        metric : ndarray
            3D scalar volume sampled along ``orig_bundle``.
        affine : ndarray, shape (4, 4)
            Voxel-to-world affine of ``metric``.
        mask_threshold : float, optional
            Fraction of the median non-empty segment count below which a
            segment is masked.
        min_count : int, optional
            Lower bound of the sparse-segment count cutoff.
        max_count : int, optional
            Upper bound of the sparse-segment count cutoff.

        Returns
        -------
        profile : ndarray, shape (n_segments,)
            Weighted mean metric value of each segment, NaN in the masked
            segments.

        References
        ----------
        .. footbibliography::

        """
        bundle = _as_streamlines(bundle)
        orig_bundle = _as_streamlines(orig_bundle)
        if len(bundle) == 0 or len(orig_bundle) == 0:
            raise ValueError("One of the bundles contains no streamlines")

        s_index, _, s_distance, _, counts = self.assign(bundle)

        # Mask the segments with too few points
        segment_counts = counts.sum(axis=1)
        nonzero = segment_counts[segment_counts > 0]
        threshold = 0
        if nonzero.size:
            threshold = np.clip(
                np.median(nonzero) * mask_threshold, min_count, max_count
            )

        affine_r = np.linalg.inv(affine)
        orig_points = Streamlines(transform_streamlines(orig_bundle, affine_r))
        values = map_coordinates(metric, orig_points.get_data().T, order=1)

        valid = np.isfinite(values)
        weights = 1 / (s_distance[valid] + 1e-8)
        weighted_sum = np.bincount(
            s_index[valid], weights=weights * values[valid], minlength=self.n_segments
        )
        weight_total = np.bincount(
            s_index[valid], weights=weights, minlength=self.n_segments
        )

        profile = np.full(self.n_segments, np.nan)
        keep = (segment_counts >= threshold) & (weight_total > 0)
        profile[keep] = weighted_sum[keep] / weight_total[keep]
        return profile


def spectra_assignment_map(
    target_bundle,
    model_bundle,
    *,
    segment_length=5.0,
    radial_length=5.0,
    n_segments=None,
    n_radial=None,
    threshold=100.0,
    use_robust_centroid=True,
    robust_method="linear",
):
    """Assign bundle points to an atlas-defined SPECTRA grid.

    See :footcite:p:`Feng2026SPECTRA` for further details about the method.

    Parameters
    ----------
    target_bundle : Streamlines
        Subject bundle in common space.
    model_bundle : Streamlines
        Atlas bundle used to define the SPECTRA grid.
    segment_length : float, optional
        Desired along-tract segment length in millimeters.
    radial_length : float, optional
        Desired radial-bin width in millimeters.
    n_segments : int, optional
        Number of along-tract segments.
    n_radial : int, optional
        Number of radial bins.
    threshold : float, optional
        QuickBundles threshold used to estimate the atlas centroid.
    use_robust_centroid : bool, optional
        If True, use an extended centroid that covers the bundle endpoints.
    robust_method : {"linear", "spline"}, optional
        Endpoint extension method used by ``compute_robust_centroid``.

    Returns
    -------
    s_index : ndarray
        Along-tract assignment for each target-bundle point.
    r_index : ndarray
        Radial assignment for each target-bundle point.
    s_distance : ndarray
        Distance from each target-bundle point to the nearest centroid point.
    valid_mask : ndarray
        Boolean mask identifying points retained for profiling.
    counts : ndarray
        Number of points in each SPECTRA grid cell.

    References
    ----------
    .. footbibliography::

    """
    grid = SpectraGrid(
        model_bundle,
        segment_length=segment_length,
        radial_length=radial_length,
        n_segments=n_segments,
        n_radial=n_radial,
        threshold=threshold,
        use_robust_centroid=use_robust_centroid,
        robust_method=robust_method,
    )
    return grid.assign(target_bundle)


def grid_profile(
    orig_bundle,
    s_index,
    r_index,
    valid_mask,
    n_segments,
    n_radial,
    metric,
    affine,
):
    """Compute mean metric values within a SPECTRA grid.

    Parameters
    ----------
    orig_bundle : Streamlines
        Subject bundle in the metric volume's world space.
    s_index : ndarray
        Along-tract assignment for each bundle point.
    r_index : ndarray
        Radial assignment for each bundle point.
    valid_mask : ndarray
        Boolean mask identifying points retained for profiling.
    n_segments : int
        Number of along-tract segments.
    n_radial : int
        Number of radial bins.
    metric : ndarray
        3D scalar volume sampled along the bundle.
    affine : ndarray, shape (4, 4)
        Voxel-to-world affine of ``metric``.

    Returns
    -------
    profile : ndarray, shape (n_segments, n_radial)
        Mean metric value in each SPECTRA grid cell.

    """
    orig_bundle = _as_streamlines(orig_bundle)
    affine_r = np.linalg.inv(affine)
    transformed_orig_bundle = Streamlines(transform_streamlines(orig_bundle, affine_r))
    values = map_coordinates(metric, transformed_orig_bundle.get_data().T, order=1)

    valid = valid_mask & np.isfinite(values)
    s_valid = s_index[valid]
    r_valid = r_index[valid]
    values_valid = values[valid]
    bin_ids = s_valid * n_radial + r_valid

    metric_sum = np.bincount(
        bin_ids, weights=values_valid, minlength=n_segments * n_radial
    )
    counts = np.bincount(bin_ids, minlength=n_segments * n_radial)

    profile = np.full(n_segments * n_radial, np.nan, dtype=float)
    occupied = counts > 0
    profile[occupied] = metric_sum[occupied] / counts[occupied]

    return profile.reshape(n_segments, n_radial)


def spectra_profile(
    model_bundle,
    bundle,
    orig_bundle,
    metric,
    affine,
    *,
    segment_length=5.0,
    radial_length=5.0,
    n_segments=None,
    n_radial=None,
    threshold=100.0,
    use_robust_centroid=True,
    robust_method="linear",
):
    """Create a two-dimensional SPECTRA bundle profile.

    See :footcite:p:`Feng2026SPECTRA` for further details about the method.

    Parameters
    ----------
    model_bundle : Streamlines
        Atlas bundle used to define the SPECTRA grid. Must be in the same
        space as ``bundle``.
    bundle : Streamlines
        Subject bundle in common space. Used for SPECTRA grid assignment.
    orig_bundle : Streamlines
        Corresponding subject bundle in native/world space. Must correspond
        point-for-point to ``bundle``.
    metric : ndarray
        3D scalar volume sampled along ``orig_bundle``.
    affine : ndarray, shape (4, 4)
        Voxel-to-world affine of ``metric``.
    segment_length : float, optional
        Desired along-tract segment length in millimeters.
    radial_length : float, optional
        Desired radial-bin width in millimeters.
    n_segments : int, optional
        Number of along-tract segments.
    n_radial : int, optional
        Number of radial bins.
    threshold : float, optional
        QuickBundles threshold used to estimate the atlas centroid.
    use_robust_centroid : bool, optional
        If True, use an extended centroid that covers the bundle endpoints.
    robust_method : {"linear", "spline"}, optional
        Endpoint extension method used by ``compute_robust_centroid``.

    Returns
    -------
    profile : ndarray
        Two-dimensional bundle profile with shape ``(n_segments, n_radial)``.

    References
    ----------
    .. footbibliography::

    """
    model_bundle = _as_streamlines(model_bundle)
    if len(model_bundle) == 0:
        raise ValueError("One of the bundles contains no streamlines")

    grid = SpectraGrid(
        model_bundle,
        segment_length=segment_length,
        radial_length=radial_length,
        n_segments=n_segments,
        n_radial=n_radial,
        threshold=threshold,
        use_robust_centroid=use_robust_centroid,
        robust_method=robust_method,
    )
    return grid.profile(bundle, orig_bundle, metric, affine)


@warning_for_keywords()
def gaussian_weights(bundle, *, n_points=100, return_mahalnobis=False, stat=np.mean):
    """
    Calculate weights for each streamline/node in a bundle, based on a
    Mahalanobis distance from the core the bundle, at that node (mean, per
    default).

    Parameters
    ----------
    bundle : Streamlines
        The streamlines to weight.
    n_points : int, optional
        The number of points to resample to. *If the `bundle` is an array, this
        input is ignored*.
    return_mahalanobis : bool, optional
        Whether to return the Mahalanobis distance instead of the weights.
    stat : callable, optional.
        The statistic used to calculate the central tendency of streamlines in
        each node. Can be one of {`np.mean`, `np.median`} or other functions
        that have similar API.`

    Returns
    -------
    w : array of shape (n_streamlines, n_points)
        Weights for each node in each streamline, calculated as its relative
        inverse of the Mahalanobis distance, relative to the distribution of
        coordinates at that node position across streamlines.

    """
    # Resample to same length for each streamline:
    bundle = set_number_of_points(bundle, nb_points=n_points)

    # This is the output
    w = np.zeros((len(bundle), n_points))

    # If there's only one fiber here, it gets the entire weighting:
    if len(bundle) == 1:
        if return_mahalnobis:
            return np.array([np.nan])
        else:
            return np.array([1])

    for node in range(n_points):
        # This should come back as a 3D covariance matrix with the spatial
        # variance covariance of this node across the different streamlines
        # This is a 3-by-3 array:
        node_coords = bundle._data[node::n_points]
        c = np.cov(node_coords.T, ddof=0)
        # Reorganize as an upper diagonal matrix for expected Mahalanobis
        # input:
        c = np.array(
            [[c[0, 0], c[0, 1], c[0, 2]], [0, c[1, 1], c[1, 2]], [0, 0, c[2, 2]]]
        )
        # Calculate the mean or median of this node as well
        # delta = node_coords - np.mean(node_coords, 0)
        m = stat(node_coords, 0)
        # Weights are the inverse of the Mahalanobis distance
        for fn in range(len(bundle)):
            # In the special case where all the streamlines have the exact same
            # coordinate in this node, the covariance matrix is all zeros, so
            # we can't calculate the Mahalanobis distance, we will instead give
            # each streamline an identical weight, equal to the number of
            # streamlines:
            if np.allclose(c, 0):
                w[:, node] = len(bundle)
                break
            # Otherwise, go ahead and calculate Mahalanobis for node on
            # fiber[fn]:
            w[fn, node] = mahalanobis(node_coords[fn], m, np.linalg.inv(c))
    if return_mahalnobis:
        return w
    # weighting is inverse to the distance (the further you are, the less you
    # should be weighted)
    w = 1 / w
    # Normalize before returning, so that the weights in each node sum to 1:
    return w / np.sum(w, 0)


@warning_for_keywords()
def afq_profile(
    data,
    bundle,
    affine,
    *,
    n_points=100,
    profile_stat=np.average,
    orient_by=None,
    weights=None,
    **weights_kwarg,
):
    """
    Calculates a summarized profile of data for a bundle or tract
    along its length.

    Follows the approach outlined in :footcite:p:`Yeatman2012`.

    Parameters
    ----------
    data : 3D volume
        The statistic to sample with the streamlines.

    bundle : StreamLines class instance
        The collection of streamlines (possibly already resampled into an array
         for each to have the same length) with which we are resampling. See
         Note below about orienting the streamlines.
    affine : array_like (4, 4)
        The mapping from voxel coordinates to streamline points.
        The voxel_to_rasmm matrix, typically from a NIFTI file.
    n_points: int, optional
        The number of points to sample along the bundle. Default: 100.
    orient_by: streamline, optional
        A streamline to use as a standard to orient all of the streamlines in
        the bundle according to.
    weights : 1D array or 2D array or callable, optional
        Weight each streamline (1D) or each node (2D) when calculating the
        tract-profiles. Must sum to 1 across streamlines (in each node if
        relevant). If callable, this is a function that calculates weights.
        When using ``gaussian_weights`` directly, its node count matches
        ``n_points``. Other callables receive the bundle and ``weights_kwarg``.
    profile_stat : callable, optional
        The statistic used to average the profile across streamlines.
        If weights is not None, this must take weights as a keyword argument.
        The default, np.average, is the same as np.mean but takes weights
        as a keyword argument.
    weights_kwarg : key-word arguments
        Additional key-word arguments to pass to the weight-calculating
        function. Only to be used if weights is a callable.

    Returns
    -------
    ndarray : a 1D array with the profile of `data` along the length of
        `bundle`

    Notes
    -----
    Before providing a bundle as input to this function, you will need to make
    sure that the streamlines in the bundle are all oriented in the same
    orientation relative to the bundle (use :func:`orient_by_streamline`).

    References
    ----------
    .. footbibliography::

    """
    if orient_by is not None:
        bundle = orient_by_streamline(bundle, orient_by)
    if affine is None:
        affine = np.eye(4)
    if len(bundle) == 0:
        raise ValueError("The bundle contains no streamlines")

    # Resample each streamline to the same number of points:
    fgarray = set_number_of_points(bundle, nb_points=n_points)

    # Extract the values
    values = np.array(values_from_volume(data, fgarray, affine))

    if weights is not None:
        if callable(weights):
            if weights is gaussian_weights:
                weights_kwarg["n_points"] = n_points
            weights = weights(bundle, **weights_kwarg)
        else:
            # We check that weights *always sum to 1 across streamlines*:
            if not np.allclose(np.sum(weights, 0), np.ones(n_points)):
                raise ValueError(
                    "The sum of weights across streamlines", " must be equal to 1"
                )

        return profile_stat(values, weights=weights, axis=0)
    else:
        return profile_stat(values, axis=0)


# SPECTRA statistics
#
# Cell-wise linear mixed models, spatial correlation of the residuals and
# hierarchical FDR for SPECTRA profiles, following the SPECTRA implementation
# of Feng et al. 2026 (Feng2026SPECTRA in doc/references.bib).


def _standardize(values):
    """Z-score the non-NaN values of an array, keeping NaN values in place."""
    values = np.asarray(values, dtype=float)
    std = np.nanstd(values)
    return (values - np.nanmean(values)) / (std if std > 0 else 1.0)


def _remove_small_regions(mask, min_size):
    """Remove the connected regions of a mask with fewer than min_size cells."""
    labels, _ = label(mask)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labels]


def _has_fitted_neighbors(mask):
    """Cells of a mask with a neighbor along the tract and across the radial bins."""
    padded = np.pad(mask, 1)
    along = padded[:-2, 1:-1] | padded[2:, 1:-1]
    if mask.shape[1] == 1:
        return mask & along
    across = padded[1:-1, :-2] | padded[1:-1, 2:]
    return mask & along & across


def _cohen_d(case_values, control_values):
    """Cohen's d with the pooled standard deviation, ignoring NaN values."""
    case = case_values[~np.isnan(case_values)]
    control = control_values[~np.isnan(control_values)]
    n_case, n_control = len(case), len(control)
    if n_case < 2 or n_control < 2:
        return np.nan

    pooled_var = (
        (n_case - 1) * np.var(case, ddof=1) + (n_control - 1) * np.var(control, ddof=1)
    ) / (n_case + n_control - 2)
    if pooled_var <= 0:
        return np.nan
    return (case.mean() - control.mean()) / np.sqrt(pooled_var)


def _fit_cell_model(df, formula, groups):
    """Fit a mixed model with a random intercept, or OLS without groups.

    Returns None if the model cannot be fitted.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if groups is not None and df[groups].nunique() > 1:
                return smf.mixedlm(formula, df, groups=df[groups]).fit(reml=True)
            return smf.ols(formula, df).fit()
        except Exception:
            # e.g. a singular design in a cell with few subjects
            return None


def fit_grid_lmm(
    profiles,
    data,
    target,
    *,
    covariates=None,
    groups=None,
    continuous=None,
    case_group=None,
    ref_group=None,
    mask=None,
    min_prop=0.3,
    min_subjects=20,
    min_component_size=8,
    fit_reduced=False,
):
    """Fit a linear mixed model in each cell of a SPECTRA grid.

    In each cell, the measure is regressed on the target variable and the
    covariates, with a random intercept for each level of ``groups``, e.g. the
    acquisition site::

        metric ~ target + covariates + (1 | groups)

    Without ``groups``, or if a cell has a single level of ``groups``, the
    model is fitted by ordinary least squares.

    Missing cells must be NaN, as returned by :meth:`SpectraGrid.profile`.
    They are dropped from the fit and from the standardization of the
    measure, and are never used as a measurement. Do not encode them as 0.

    See :footcite:p:`Feng2026SPECTRA` for further details about the method.

    Parameters
    ----------
    profiles : ndarray, shape (n_subjects, n_segments, n_radial)
        SPECTRA profile of each subject, all computed with the same
        :class:`SpectraGrid` (see :meth:`SpectraGrid.profile`). A 1D profile
        with shape ``(n_subjects, n_segments)`` is also accepted.
    data : pandas.DataFrame
        Variables of the subjects, one row per subject, in the same order as
        ``profiles``.
    target : str
        Column of ``data`` with the variable of interest.
    covariates : list of str, optional
        Columns of ``data`` used as covariates.
    groups : str, optional
        Column of ``data`` used for the random intercept.
    continuous : list of str, optional
        Columns among ``target`` and ``covariates`` that are standardized
        before fitting.
    case_group, ref_group : optional
        Values of a categorical ``target`` to compare. The target is coded 1
        for ``case_group`` and 0 for ``ref_group``, and the other subjects are
        excluded. Required if ``target`` is not numeric.
    mask : ndarray, shape (n_segments, n_radial), optional
        Cells to fit. By default all the cells with enough subjects.
    min_prop : float, optional
        Minimum proportion of subjects with a valid measure in a cell.
    min_subjects : int, optional
        Minimum number of subjects with a valid measure in a cell.
    min_component_size : int, optional
        Cells with enough subjects are not fitted if they form a connected
        region with fewer cells than this.
    fit_reduced : bool, optional
        If True, also fit the model without the target. Its residuals are used
        to estimate the spatial correlation with
        :func:`estimate_length_scales`.

    Returns
    -------
    results : dict
        Arrays with shape ``(n_segments, n_radial)``, NaN in the cells that
        are not fitted: ``"t"``, ``"p"`` (two-sided), ``"beta"``, ``"se"``,
        ``"dof"``, ``"ci_lower"``, ``"ci_upper"`` (95% Wald confidence
        interval, as in SPECTRA),
        ``"r2_partial"`` and, if ``case_group`` is given, ``"d_cohen"``. Also:

        - ``"n_subjects"``: number of subjects with a valid measure per cell.
        - ``"mask"``: cells that were fitted and tested.
        - ``"residuals"``: shape ``(n_segments, n_radial, n_subjects)``,
          residuals of the model.
        - ``"residuals_reduced"``: residuals of the model without the target,
          only if ``fit_reduced`` is True.

    Notes
    -----
    As in SPECTRA, a fitted cell is only tested if it has a fitted neighbor
    along the tract and across the radial bins, where the local smoothness of
    the residuals is defined. The statistics of the other fitted cells are
    NaN, but their residuals are kept.

    The degrees of freedom of the mixed models are approximated by the number
    of subjects minus the number of fixed effects.

    References
    ----------
    .. footbibliography::

    """
    if not have_pd or not have_smf:
        raise ImportError("fit_grid_lmm requires pandas and statsmodels")

    profiles = np.array(profiles, dtype=float)
    if profiles.ndim == 2:
        profiles = profiles[:, :, np.newaxis]
    if profiles.ndim != 3:
        raise ValueError("profiles must have shape (n_subjects, n_segments, n_radial)")
    n_subjects, n_segments, n_radial = profiles.shape
    if len(data) != n_subjects:
        raise ValueError("data must have one row per subject in profiles")
    if (case_group is None) != (ref_group is None):
        raise ValueError("case_group and ref_group must be given together")

    covariates = list(covariates or [])
    columns = [target] + covariates + ([groups] if groups else [])
    continuous = list(continuous or [])
    missing_columns = [c for c in columns + continuous if c not in data.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in data: {missing_columns}")

    df = data.reset_index(drop=True).copy()
    for col in continuous:
        df[col] = _standardize(df[col])

    # Subjects used in the fit: those with all the model variables, and only
    # the two compared groups for a categorical target
    use_subject = df[columns].notna().all(axis=1).to_numpy()
    if case_group is not None:
        use_subject = use_subject & df[target].isin([case_group, ref_group]).to_numpy()
        df[target] = (df[target] == case_group).astype(float)
        if df.loc[use_subject, target].nunique() < 2:
            raise ValueError("Both case_group and ref_group must be in data")
    elif not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError("Give case_group and ref_group for a categorical target")
    elif df.loc[use_subject, target].nunique() < 2:
        raise ValueError("The target variable has a single value")

    # Missing cells (NaN) are dropped. Setting them to 0 would bias the
    # statistics and underestimate the radial length scale.
    valid = np.isfinite(profiles) & use_subject[:, np.newaxis, np.newaxis]
    n_valid = valid.sum(axis=0)
    # Cells with enough subjects, without the small isolated regions
    enough = n_valid >= min_prop * use_subject.sum()
    cell_mask = _remove_small_regions(enough, min_component_size)
    cell_mask &= n_valid >= min_subjects
    if mask is not None:
        cell_mask &= np.asarray(mask, dtype=bool).reshape(cell_mask.shape)

    # Standardize the measure with the valid values of the cells to fit
    metric = np.where(valid & cell_mask, profiles, np.nan)
    if cell_mask.any():
        metric = _standardize(metric)

    formula = "metric ~ " + " + ".join([target] + covariates)
    formula_reduced = "metric ~ " + (" + ".join(covariates) if covariates else "1")

    shape = (n_segments, n_radial)
    stat_names = ["t", "p", "beta", "se", "dof", "ci_lower", "ci_upper", "r2_partial"]
    if case_group is not None:
        stat_names.append("d_cohen")
    results = {name: np.full(shape, np.nan) for name in stat_names}
    residuals = np.full(shape + (n_subjects,), np.nan)
    residuals_reduced = np.full(shape + (n_subjects,), np.nan)

    for s, r in zip(*np.nonzero(cell_mask)):
        rows = np.flatnonzero(~np.isnan(metric[:, s, r]))
        df_cell = df.iloc[rows].copy()
        df_cell["metric"] = metric[rows, s, r]

        fit = _fit_cell_model(df_cell, formula, groups)
        if fit is None:
            continue

        beta = fit.params[target]
        se = fit.bse[target]
        dof = len(rows) - fit.model.exog.shape[1]
        if not (np.isfinite(beta) and np.isfinite(se) and se > 0 and dof > 0):
            continue

        t = beta / se
        margin = stats.norm.ppf(0.975) * se
        results["t"][s, r] = t
        results["p"][s, r] = 2 * stats.t.sf(abs(t), dof)
        results["beta"][s, r] = beta
        results["se"][s, r] = se
        results["dof"][s, r] = dof
        results["ci_lower"][s, r] = beta - margin
        results["ci_upper"][s, r] = beta + margin
        results["r2_partial"][s, r] = t**2 / (t**2 + dof)
        residuals[s, r, rows] = np.asarray(fit.resid)

        if case_group is not None:
            is_case = df_cell[target].to_numpy() == 1
            y = df_cell["metric"].to_numpy()
            results["d_cohen"][s, r] = _cohen_d(y[is_case], y[~is_case])

        if fit_reduced:
            fit_reduced_model = _fit_cell_model(df_cell, formula_reduced, groups)
            if fit_reduced_model is not None:
                residuals_reduced[s, r, rows] = np.asarray(fit_reduced_model.resid)

    # Test only the cells with a fitted neighbor in both directions
    tested = _has_fitted_neighbors(~np.isnan(results["t"]))
    for name in stat_names:
        results[name][~tested] = np.nan

    results["n_subjects"] = n_valid
    results["mask"] = tested
    results["residuals"] = residuals
    if fit_reduced:
        results["residuals_reduced"] = residuals_reduced
    return results


def matern_correlation(distance, length_scale, *, nu=0.5):
    """Compute the Matern correlation at the given distances.

    Parameters
    ----------
    distance : array_like
        Distances between locations.
    length_scale : float
        Length scale of the correlation, in the units of ``distance``.
    nu : {0.5, 1.5, 2.5}, optional
        Smoothness of the correlation. ``nu=0.5`` is the exponential
        correlation used by SPECTRA.

    Returns
    -------
    correlation : ndarray
        Correlation at each distance.

    """
    d = np.asarray(distance, dtype=float) / length_scale
    if nu == 0.5:
        return np.exp(-d)
    if nu == 1.5:
        return (1 + np.sqrt(3) * d) * np.exp(-np.sqrt(3) * d)
    if nu == 2.5:
        return (1 + np.sqrt(5) * d + 5 * d**2 / 3) * np.exp(-np.sqrt(5) * d)
    raise ValueError("nu must be 0.5, 1.5 or 2.5")


def empirical_correlation(residuals, *, axis=0, max_lag=20):
    """Compute the spatial correlation of residuals along one grid axis.

    The residuals of each cell are centered across subjects. The correlation
    at lag ``m`` is the mean product of the residuals of cells ``m`` steps
    apart, divided by the mean squared residual. NaN values are ignored.

    Parameters
    ----------
    residuals : ndarray, shape (n_segments, n_radial, n_subjects)
        Residuals of a model fitted in each cell, NaN where missing.
    axis : {0, 1}, optional
        0 for the along-tract direction, 1 for the radial direction.
    max_lag : int, optional
        Largest lag, in grid cells.

    Returns
    -------
    lags : ndarray
        Lags ``0, 1, ..., max_lag``.
    rho : ndarray
        Correlation at each lag.
    n_pairs : ndarray
        Number of pairs of values at each lag.

    """
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")

    resid = np.ma.masked_invalid(residuals)
    resid = resid - resid.mean(axis=2, keepdims=True)
    n_cells = resid.shape[axis]

    max_lag = min(max_lag, n_cells - 1)
    lags = np.arange(max_lag + 1)
    cov = np.full(max_lag + 1, np.nan)
    n_pairs = np.zeros(max_lag + 1, dtype=int)
    for m in lags:
        if axis == 0:
            prod = resid[: n_cells - m] * resid[m:]
        else:
            prod = resid[:, : n_cells - m] * resid[:, m:]
        n_pairs[m] = prod.count()
        if n_pairs[m] > 0:
            cov[m] = prod.mean()

    return lags, cov / cov[0], n_pairs


def fit_matern(
    lags, rho, n_pairs, spacing, *, nu=0.5, rho_thresh=0.2, max_lag=12, min_pairs=20
):
    """Fit a Matern correlation to an empirical spatial correlation.

    Only the first lags are used, up to the lag where the correlation falls
    below ``rho_thresh``, to avoid fitting the noisy tail. Each lag is weighted
    by its number of pairs.

    Parameters
    ----------
    lags, rho, n_pairs : ndarray
        Lags, correlation and number of pairs, as returned by
        :func:`empirical_correlation`.
    spacing : float
        Grid spacing along the axis of the correlation (mm).
    nu : {0.5, 1.5, 2.5}, optional
        Smoothness of the Matern correlation.
    rho_thresh : float, optional
        Correlation below which the lags are not used.
    max_lag : int, optional
        Largest number of lags used.
    min_pairs : int, optional
        Minimum number of pairs for a lag to be used.

    Returns
    -------
    length_scale : float
        Fitted length scale (mm).

    """
    lags = np.asarray(lags)
    rho = np.asarray(rho, dtype=float)
    n_pairs = np.asarray(n_pairs)

    # Use the lags before the correlation first falls below rho_thresh
    below = np.flatnonzero(rho < rho_thresh)
    n_lags = below[0] if len(below) > 0 else np.nanargmin(rho)
    n_lags = int(np.clip(n_lags, 3, max_lag))
    lags, rho, n_pairs = lags[:n_lags], rho[:n_lags], n_pairs[:n_lags]

    keep = (lags >= 1) & (rho > 0) & (n_pairs > min_pairs)
    if not np.any(keep):
        raise ValueError("Not enough lags with a positive correlation to fit")
    distance = lags[keep] * spacing
    rho = rho[keep]
    weights = n_pairs[keep]

    # Starting point: squared exponential exp(-d**2 / (2 * l**2)), fitted by
    # weighted least squares on -2 * log(rho) = d**2 / l**2
    x = distance**2
    y = -2 * np.log(rho)
    slope = np.sum(weights * x * y) / np.sum(weights * x * x)
    length_start = 1 / np.sqrt(slope)

    def weighted_error(params):
        rho_fit = matern_correlation(distance, params[0], nu=nu)
        return np.sum(weights * (rho - rho_fit) ** 2)

    result = minimize(
        weighted_error,
        [length_start],
        method="L-BFGS-B",
        bounds=[(0.1 * length_start, 10 * length_start)],
    )
    if not result.success:
        logger.warning(f"Matern fit did not converge: {result.message}")
    return float(result.x[0])


def estimate_length_scales(residuals, grid, *, nu=0.5, rho_thresh=0.2):
    """Estimate the spatial correlation length scales of residuals.

    A Matern correlation is fitted to the correlation of the residuals along
    the tract and across the radial bins of a SPECTRA grid. Use the residuals
    of a model without the target variable (``"residuals_reduced"`` of
    :func:`fit_grid_lmm`), so that the estimate does not depend on the effect
    being tested.

    See :footcite:p:`Feng2026SPECTRA` for further details about the method.

    Parameters
    ----------
    residuals : ndarray, shape (n_segments, n_radial, n_subjects)
        Residuals of a model fitted in each cell of ``grid``, NaN where
        missing. For 1D profiles (see :meth:`SpectraGrid.profile_1d_adaptive_buan`),
        ``n_radial`` is 1.
    grid : SpectraGrid
        Grid of the profiles. Its ``segment_length_actual`` and
        ``radial_length_actual`` give the length scales in millimeters.
    nu : {0.5, 1.5, 2.5}, optional
        Smoothness of the Matern correlation.
    rho_thresh : float, optional
        Correlation below which the lags are not used in the fit.

    Returns
    -------
    l_s : float
        Length scale along the tract (mm).
    l_r : float or None
        Length scale across the radial bins (mm). None if the grid has a
        single radial bin.

    References
    ----------
    .. footbibliography::

    """
    n_segments, n_radial, _ = residuals.shape
    if n_segments != grid.n_segments or n_radial not in (1, grid.n_radial):
        raise ValueError("residuals do not have the shape of the grid")

    lags, rho, n_pairs = empirical_correlation(
        residuals, axis=0, max_lag=n_segments - 1
    )
    l_s = fit_matern(
        lags, rho, n_pairs, grid.segment_length_actual, nu=nu, rho_thresh=rho_thresh
    )

    l_r = None
    if n_radial > 1:
        lags, rho, n_pairs = empirical_correlation(
            residuals, axis=1, max_lag=n_radial - 1
        )
        l_r = fit_matern(
            lags, rho, n_pairs, grid.radial_length_actual, nu=nu, rho_thresh=rho_thresh
        )
    return l_s, l_r


def _storey_pi0(p_values):
    """Storey's estimate of the proportion of true null hypotheses."""
    lambdas = np.arange(0.05, 0.95, 0.05)
    pi0_lambda = np.array([np.mean(p_values > lam) / (1 - lam) for lam in lambdas])
    # Smooth with a cubic spline and extrapolate to lambda = 1
    pi0 = float(UnivariateSpline(lambdas, pi0_lambda, k=3)(1.0))
    return min(max(np.clip(pi0, 1e-3, 1.0), pi0_lambda.min()), 1.0)


def fdr_correction(p_values, *, adaptive=False):
    """Adjust p-values for the false discovery rate (FDR).

    Benjamini-Hochberg procedure. If ``adaptive`` is True, the adjusted
    p-values are multiplied by Storey's estimate of the proportion of true
    null hypotheses, and are not allowed to be smaller than the uncorrected
    p-values.

    See :footcite:p:`Feng2026SPECTRA` for further details about the method.

    Parameters
    ----------
    p_values : array_like
        Uncorrected p-values. NaN values are ignored.
    adaptive : bool, optional
        If True, use the adaptive procedure.

    Returns
    -------
    p_corr : ndarray
        Adjusted p-values, with the shape of ``p_values``. The hypotheses
        with ``p_corr < alpha`` are rejected at FDR level ``alpha``.

    References
    ----------
    .. footbibliography::

    """
    p_values = np.asarray(p_values, dtype=float)
    p_corr = np.full(p_values.shape, np.nan)
    valid = ~np.isnan(p_values)
    p = p_values[valid]
    m = len(p)
    if m == 0:
        return p_corr

    pi0 = _storey_pi0(p) if adaptive else 1.0

    order = np.argsort(p)
    adjusted = p[order] * m * pi0 / np.arange(1, m + 1)
    # Make the adjusted p-values increase with the uncorrected p-values
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]

    q = np.empty(m)
    q[order] = np.minimum(adjusted, 1.0)
    if adaptive:
        q = np.maximum(q, p)
    p_corr[valid] = q
    return p_corr


def simes_combine(p_values, z_values):
    """Combine the p-values of a block of cells with Simes' test.

    For ``n`` cells the combined p-value is ``min(n * p_(k) / k)`` over the
    sorted p-values ``p_(1) <= ... <= p_(n)``.

    See :footcite:p:`Feng2026SPECTRA` for further details about the method.

    Parameters
    ----------
    p_values : ndarray
        Two-sided uncorrected p-values of the cells.
    z_values : ndarray
        Signed z-scores of the cells. Only used to give the combined z-score
        the sign of the most significant cell.

    Returns
    -------
    z : float
        Signed z-score of the combined p-value.
    p : float
        Combined p-value.

    References
    ----------
    .. footbibliography::

    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    p = min(1.0, float(np.min(n * p_values[order] / np.arange(1, n + 1))))
    sign = 1.0 if z_values[order[0]] >= 0 else -1.0
    return sign * stats.norm.ppf(1 - p / 2), p


def stouffer_combine(z_values, *, correlation=None):
    """Combine the z-scores of a block of cells with Stouffer's method.

    The sum of the z-scores is divided by its standard deviation, which is
    ``sqrt(n)`` for independent cells, or the square root of the sum of the
    correlation matrix for correlated cells.

    Parameters
    ----------
    z_values : ndarray
        Signed z-scores of the cells.
    correlation : ndarray, shape (n, n), optional
        Correlation between the cells. By default the cells are independent.

    Returns
    -------
    z : float
        Combined z-score.
    p : float
        Two-sided p-value of the combined z-score.

    """
    z_values = np.asarray(z_values, dtype=float)
    variance = len(z_values) if correlation is None else np.sum(correlation)
    z = np.sum(z_values) / np.sqrt(variance)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def conditional_pvalue(z, threshold, rho):
    """Compute the conditional p-value of a cell in a rejected block.

    Two-sided p-value of the cell z-score, given that the block z-score
    passed the Stage 1 threshold. The block is assumed to be null, which gives
    the conservative bound of the hierarchical FDR procedure of Benjamini and
    Heller.

    See :footcite:p:`Feng2026SPECTRA` for further details about the method.

    Parameters
    ----------
    z : float
        Signed z-score of the cell.
    threshold : float
        Stage 1 threshold: the largest uncorrected p-value of the rejected
        blocks.
    rho : float
        Correlation between the cell z-score and the block z-score.

    Returns
    -------
    p_cond : float
        Conditional p-value.

    References
    ----------
    .. footbibliography::

    """
    z = abs(z)
    if abs(rho) < 1e-10:
        return float(2 * stats.norm.sf(z))

    z_block = stats.norm.ppf(1 - threshold / 2)
    scale = np.sqrt(max(1 - rho**2, 0.0))

    def tail_given_block(u):
        # P(|cell z| >= z | block z = u) times the density of the block z
        mean = rho * u
        tail = stats.norm.sf((z - mean) / scale) + stats.norm.cdf((-z - mean) / scale)
        return tail * stats.norm.pdf(u)

    upper, _ = integrate.quad(tail_given_block, z_block, np.inf)
    lower, _ = integrate.quad(tail_given_block, -np.inf, -z_block)
    p_block = 2 * stats.norm.sf(z_block)
    if p_block < 1e-15:
        return float(2 * stats.norm.sf(z))
    return float(np.clip((upper + lower) / p_block, 0.0, 1.0))


def partition_bundle_into_blocks(mask, grid, l_s, *, l_r=None, min_cells=3):
    """Split the cells of a SPECTRA grid into blocks of about one RESEL.

    The blocks are rectangles whose sides are the full width at half maximum
    (FWHM) of the spatial correlation, which is the size of a resolution
    element (RESEL). Blocks with fewer than
    ``min_cells`` cells are merged with their smallest neighbor.

    Parameters
    ----------
    mask : ndarray, shape (n_segments, n_radial)
        Cells of ``grid`` to include in the blocks.
    grid : SpectraGrid
        Grid of the bundle, used for the segment length and the radial bin
        width.
    l_s : float
        Correlation length scale along the tract (mm).
    l_r : float, optional
        Correlation length scale across the radial bins (mm). If None, a
        block spans all the radial bins.
    min_cells : int, optional
        Minimum number of cells in a block.

    Returns
    -------
    blocks : list of list of tuple
        ``(segment, radial)`` indices of the cells of each block.

    References
    ----------
    .. footbibliography::

    """
    # Factor between the Matern length scale and the FWHM of the correlation,
    # 2 * sqrt(2 * log(2)) ~= 2.355. The FWHM defines the size of a RESEL. The
    # rounded value matches the SPECTRA implementation of Feng et al. 2026
    # (Feng2026SPECTRA in doc/references.bib).
    fwhm_factor = 2.355

    mask = np.asarray(mask, dtype=bool)
    n_segments, n_radial = mask.shape
    size_s = max(2, round(fwhm_factor * l_s / grid.segment_length_actual))
    if l_r is None:
        size_r = n_radial
    else:
        size_r = max(2, round(fwhm_factor * l_r / grid.radial_length_actual))

    blocks = []
    for s0 in range(0, n_segments, size_s):
        for r0 in range(0, n_radial, size_r):
            cells = [
                (s, r)
                for s in range(s0, min(s0 + size_s, n_segments))
                for r in range(r0, min(r0 + size_r, n_radial))
                if mask[s, r]
            ]
            if cells:
                blocks.append(cells)

    # Merge the smallest block with its smallest neighbor until all the blocks
    # are large enough
    while len(blocks) > 1:
        sizes = [len(block) for block in blocks]
        i = int(np.argmin(sizes))
        if sizes[i] >= min_cells:
            break
        neighbors = [j for j in (i - 1, i + 1) if 0 <= j < len(blocks)]
        j = min(neighbors, key=lambda k: sizes[k])
        first, second = min(i, j), max(i, j)
        blocks[first] = blocks[first] + blocks[second]
        del blocks[second]

    return blocks


def _block_correlation(cells, grid, l_s, l_r):
    """Exponential correlation between the cells of a block."""
    cells = np.asarray(cells)
    ds = np.abs(cells[:, np.newaxis, 0] - cells[np.newaxis, :, 0])
    correlation = np.exp(-ds * grid.segment_length_actual / l_s)
    if l_r is not None:
        dr = np.abs(cells[:, np.newaxis, 1] - cells[np.newaxis, :, 1])
        correlation *= np.exp(-dr * grid.radial_length_actual / l_r)
    return correlation


def hierarchical_fdr(
    bundles,
    *,
    alpha_block=0.05,
    use_resel_blocks=True,
    combination_method="simes",
):
    """Correct SPECTRA statistics with the hierarchical FDR procedure.

    Two-stage procedure of Benjamini and Heller for spatial signals, as used
    in SPECTRA. See :footcite:p:`Feng2026SPECTRA` for further details about
    the method.

    Stage 1 splits each bundle into blocks of about one RESEL (see
    :func:`partition_bundle_into_blocks`) and combines the cells of each block
    into a block p-value, with Simes' test or Stouffer's method. The
    Benjamini-Hochberg procedure is applied to the blocks of all the bundles at
    level ``alpha_block``. Stage 2 computes a conditional p-value for each cell
    of the rejected blocks, and applies the Benjamini-Hochberg procedure within
    each rejected block.

    Parameters
    ----------
    bundles : dict
        For each bundle name, a dictionary with:

        - ``"grid"``: SpectraGrid of the bundle.
        - ``"t"``: ndarray, shape (n_segments, n_radial), test statistic of
          each cell of the grid, e.g. ``"t"`` of :func:`fit_grid_lmm`.
          ``n_radial`` is 1 for 1D profiles.
        - ``"p"``: ndarray, shape (n_segments, n_radial), two-sided
          uncorrected p-value of each cell. Cells with a NaN p-value are not
          tested.
        - ``"l_s"``, ``"l_r"``: float, optional, correlation length scales
          along the tract and across the radial bins, see
          :func:`estimate_length_scales`.
        - ``"z"``: ndarray, optional, signed z-score of each cell. By default
          it is computed from ``"t"`` and ``"p"``.
    alpha_block : float, optional
        FDR level of Stage 1.
    use_resel_blocks : bool, optional
        If True, split each bundle into RESEL blocks. If False, or if a bundle
        has no ``"l_s"``, the whole bundle is a single block.
    combination_method : {"simes", "stouffer"}, optional
        Method used to combine the cells of a block.

    Returns
    -------
    p_hfdr : dict
        For each bundle, Stage 2 adjusted p-value of each cell, NaN outside
        the rejected blocks. The cells with ``p_hfdr < alpha_cell`` are
        significant. SPECTRA uses ``alpha_cell=0.15`` by default.
    blocks : list of dict
        Stage 1 blocks, with keys ``"bundle"``, ``"cells"`` (list of
        ``(segment, radial)`` indices), ``"z"``, ``"p"``, ``"p_adj"``,
        ``"significant"`` and ``"p_cond"``, the conditional p-values of the
        cells (None if the block is not rejected).

    References
    ----------
    .. footbibliography::

    """
    if combination_method not in ("simes", "stouffer"):
        raise ValueError("combination_method must be 'simes' or 'stouffer'")

    # Stage 1: combine the cells of each block
    blocks = []
    cell_z_rho = []
    for name, bundle in bundles.items():
        grid = bundle["grid"]
        p = np.asarray(bundle["p"], dtype=float)
        if p.shape not in ((grid.n_segments, grid.n_radial), (grid.n_segments, 1)):
            raise ValueError(f"The statistics of {name} do not have the grid shape")
        valid = ~np.isnan(p)
        if not valid.any():
            continue
        z = bundle.get("z")
        if z is None:
            # Signed z-scores, clipped to keep very small p-values finite
            z = np.clip(np.sign(bundle["t"]) * stats.norm.ppf(1 - p / 2), -10, 10)
        l_s = bundle.get("l_s")
        l_r = bundle.get("l_r")

        if use_resel_blocks and l_s is not None:
            cell_blocks = partition_bundle_into_blocks(valid, grid, l_s, l_r=l_r)
        else:
            cell_blocks = [[(int(s), int(r)) for s, r in zip(*np.nonzero(valid))]]

        for cells in cell_blocks:
            s_idx, r_idx = np.array(cells).T
            z_cells = z[s_idx, r_idx]

            correlation = None
            if combination_method == "simes":
                z_block, p_block = simes_combine(p[s_idx, r_idx], z_cells)
            else:
                if l_s is not None:
                    correlation = _block_correlation(cells, grid, l_s, l_r)
                z_block, p_block = stouffer_combine(z_cells, correlation=correlation)

            # Correlation between each cell z-score and the block z-score
            if correlation is None:
                rho = np.full(len(cells), 1 / np.sqrt(len(cells)))
            else:
                rho = correlation.sum(axis=1) / np.sqrt(correlation.sum())

            blocks.append({"bundle": name, "cells": cells, "z": z_block, "p": p_block})
            cell_z_rho.append((z_cells, rho))

    if not blocks:
        raise ValueError("No cell has a valid p-value")

    # Stage 1: Benjamini-Hochberg across the blocks of all the bundles
    p_blocks = np.array([block["p"] for block in blocks])
    p_adj = fdr_correction(p_blocks)
    rejected = p_adj < alpha_block
    threshold = p_blocks[rejected].max() if rejected.any() else alpha_block

    # Stage 2: conditional p-values, corrected within each rejected block
    p_hfdr = {name: np.full(np.shape(b["p"]), np.nan) for name, b in bundles.items()}
    for block, block_p_adj, is_rejected, (z_cells, rho) in zip(
        blocks, p_adj, rejected, cell_z_rho
    ):
        block["p_adj"] = float(block_p_adj)
        block["significant"] = bool(is_rejected)
        block["p_cond"] = None
        if not is_rejected:
            continue

        p_cond = np.array(
            [conditional_pvalue(zc, threshold, rc) for zc, rc in zip(z_cells, rho)]
        )
        block["p_cond"] = p_cond
        s_idx, r_idx = np.array(block["cells"]).T
        p_hfdr[block["bundle"]][s_idx, r_idx] = fdr_correction(p_cond)

    return p_hfdr, blocks
