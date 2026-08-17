from pathlib import Path

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.spatial import cKDTree
from scipy.spatial.distance import mahalanobis

from dipy.io.utils import save_buan_profiles_hdf5
from dipy.segment.clustering import QuickBundles
from dipy.segment.metricspeed import AveragePointwiseEuclideanMetric
from dipy.testing.decorators import warning_for_keywords
from dipy.tracking.streamline import (
    Streamlines,
    orient_by_streamline,
    set_number_of_points,
    transform_streamlines,
    values_from_volume,
)
from dipy.tracking.utils import length


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


MERGE_THRESHOLD = 0.2
MASK_THRESHOLD = 0.2


def _as_streamlines(bundle):
    """Return ``bundle`` as a Streamlines object."""
    if isinstance(bundle, Streamlines):
        return bundle
    return Streamlines(bundle)


@warning_for_keywords()
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

    centroid_lengths = np.asarray(list(length(centroids)))
    return np.asarray(centroids[np.argmax(centroid_lengths)])


@warning_for_keywords()
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

    bundle_lengths = np.asarray(list(length(bundle)))
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


@warning_for_keywords()
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
    if n_segments is None and segment_length is None:
        raise ValueError("Either n_segments or segment_length must be provided")

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


@warning_for_keywords()
def create_radial_bins(radial_distance, *, radial_length=5.0, n_radial=None):
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
        threshold = np.median(counts) * MERGE_THRESHOLD
        remove_first = counts[0] < threshold
        remove_last = counts[-1] < threshold

        if remove_first:
            radial_index[radial_index == 0] = 1
            radial_edges = radial_edges[1:]

        if remove_last:
            last_index = n_bins - 1
            radial_index[radial_index == last_index] = last_index - 1
            radial_edges = radial_edges[:-1]

        if remove_first or remove_last:
            _, radial_index = np.unique(radial_index, return_inverse=True)

    return radial_index, len(radial_edges) - 1, radial_edges


@warning_for_keywords()
def get_grid_from_atlas(
    atlas_bundle,
    *,
    segment_length=5.0,
    radial_length=5.0,
    n_segments=None,
    n_radial=None,
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
        Number of radial bins.
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

    if n_segments is not None and n_radial is None:
        n_radial = 1

    if use_robust_centroid:
        centroid = compute_robust_centroid(
            atlas_bundle,
            segment_length=segment_length,
            n_segments=n_segments,
            method=robust_method,
        )
    else:
        if n_segments is None:
            n_segments = get_n_segment_by_length(
                atlas_bundle, segment_length=segment_length
            )
        centroid = get_centroid(atlas_bundle, n_points=n_segments)

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


def parameterize_bundle(bundle, centroid, radial_vectors, radial_edges):
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
        sparse_threshold = np.clip(np.median(nonzero) * MASK_THRESHOLD, 50, 200)
        sparse_bins = np.flatnonzero(counts_flat < sparse_threshold)
        if sparse_bins.size:
            valid_mask[np.isin(bin_ids, sparse_bins)] = False

    counts = counts_flat.reshape(n_segments, n_radial)
    return s_index, r_index, s_distance, valid_mask, counts


@warning_for_keywords()
def spectra_assignment_map(
    target_bundle,
    model_bundle,
    *,
    segment_length=5.0,
    radial_length=5.0,
    n_segments=None,
    n_radial=None,
    use_robust_centroid=True,
    robust_method="linear",
):
    """Assign bundle points to an atlas-defined SPECTRA grid.

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
    _, _, centroid, radial_vectors, radial_edges, _, _ = get_grid_from_atlas(
        model_bundle,
        segment_length=segment_length,
        radial_length=radial_length,
        n_segments=n_segments,
        n_radial=n_radial,
        use_robust_centroid=use_robust_centroid,
        robust_method=robust_method,
    )

    return parameterize_bundle(target_bundle, centroid, radial_vectors, radial_edges)


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
    affine_r = np.linalg.inv(affine)
    transformed_orig_bundle = transform_streamlines(orig_bundle, affine_r)
    values = map_coordinates(metric, transformed_orig_bundle._data.T, order=1)

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


@warning_for_keywords()
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
    use_robust_centroid=True,
    robust_method="linear",
):
    """Create a two-dimensional SPECTRA bundle profile.

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
    if len(model_bundle) == 0 or len(bundle) == 0 or len(orig_bundle) == 0:
        raise ValueError("One of the bundles contains no streamlines")

    _, _, centroid, radial_vectors, radial_edges, _, _ = get_grid_from_atlas(
        model_bundle,
        segment_length=segment_length,
        radial_length=radial_length,
        n_segments=n_segments,
        n_radial=n_radial,
        use_robust_centroid=use_robust_centroid,
        robust_method=robust_method,
    )

    s_index, r_index, _, valid_mask, _ = parameterize_bundle(
        bundle, centroid, radial_vectors, radial_edges
    )

    return grid_profile(
        orig_bundle,
        s_index,
        r_index,
        valid_mask,
        len(centroid),
        len(radial_edges) - 1,
        metric,
        affine,
    )


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
