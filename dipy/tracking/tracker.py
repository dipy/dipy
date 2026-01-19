import os

from nibabel.affines import voxel_sizes
import numpy as np

from dipy.data import default_sphere
from dipy.direction import (
    BootDirectionGetter,
    ClosestPeakDirectionGetter,
    PeakDirectionGen,
    ProbabilisticDirectionGetter,
)
from dipy.direction.peaks import peaks_from_positions
from dipy.direction.pmf import SHCoeffPmfGen, SimplePeakGen, SimplePmfGen
from dipy.tracking.local_tracking import LocalTracking, ParticleFilteringTracking
from dipy.tracking.tracker_parameters import generate_tracking_parameters
from dipy.tracking.tractogen import generate_tractogram
from dipy.tracking.utils import seeds_directions_pairs


def generic_tracking(
    seed_positions,
    seed_directions,
    sc,
    params,
    *,
    affine=None,
    sh=None,
    peaks=None,
    sf=None,
    sphere=None,
    basis_type=None,
    legacy=True,
    max_cross=None,
    nbr_threads=0,
    seed_buffer_fraction=1.0,
    save_seeds=False,
):
    affine = affine if affine is not None else np.eye(4)

    pmf_type = [
        {"name": "sh", "value": sh, "cls": SHCoeffPmfGen},
        {"name": "peaks", "value": peaks, "cls": SimplePmfGen},
        {"name": "sf", "value": sf, "cls": SimplePmfGen},
    ]

    initialized_pmf = [
        d_selected for d_selected in pmf_type if d_selected["value"] is not None
    ]
    if len(initialized_pmf) > 1:
        selected_pmf = ", ".join([p["name"] for p in initialized_pmf])
        raise ValueError(
            "Only one pmf type should be initialized. "
            f"Variables initialized: {', '.join(selected_pmf)}"
        )
    if len(initialized_pmf) == 0:
        available_pmf = ", ".join([d["name"] for d in pmf_type])
        raise ValueError(
            f"No PMF found. One of this variable ({available_pmf}) should be"
            " initialized."
        )

    selected_pmf = initialized_pmf[0]

    if selected_pmf["name"] == "sf" and sphere is None:
        raise ValueError("A sphere should be defined when using SF (an ODF).")

    sphere = sphere or default_sphere

    if selected_pmf["name"] == "peaks":
        pam = peaks
        if not hasattr(pam, "peak_indices") or not hasattr(pam, "peak_values"):
            raise ValueError(
                "peaks must be a PeaksAndMetrics object with "
                "peak_indices and peak_values attributes"
            )

        if hasattr(pam, "odf_vertices") and pam.odf_vertices is not None:
            odf_vertices = pam.odf_vertices
        else:
            odf_vertices = sphere.vertices

        peak_gen = PeakDirectionGen(
            np.asarray(pam.peak_indices, dtype=float, order="C"),
            np.asarray(pam.peak_values, dtype=float, order="C"),
            np.asarray(odf_vertices, dtype=float, order="C"),
        )

        pmf_gen = SimplePeakGen(peak_gen, sphere)
    else:
        kwargs = {}
        if selected_pmf["name"] == "sh":
            kwargs = {"basis_type": basis_type, "legacy": legacy}

        pmf_gen = selected_pmf["cls"](
            np.asarray(selected_pmf["value"], dtype=float), sphere, **kwargs
        )

    inv_affine = np.linalg.inv(affine)
    seed_voxels = np.dot(seed_positions, inv_affine[:3, :3].T)
    seed_voxels += inv_affine[:3, 3]
    seed_voxels = np.round(seed_voxels).astype(int)

    if seed_directions is not None:
        if not isinstance(seed_directions, (np.ndarray, list)):
            raise ValueError("seed_directions should be a numpy array or a list.")
        elif isinstance(seed_directions, list):
            seed_directions = np.array(seed_directions)

        if not np.array_equal(seed_directions.shape, seed_positions.shape):
            raise ValueError(
                "seed_directions and seed_positions should have the same shape."
            )
    else:
        if selected_pmf["name"] == "peaks":
            pam = selected_pmf["value"]

            if hasattr(pam, "odf_vertices") and pam.odf_vertices is not None:
                odf_verts = pam.odf_vertices
            else:
                odf_verts = sphere.vertices

            seed_voxels[:, 0] = np.clip(
                seed_voxels[:, 0], 0, pam.peak_indices.shape[0] - 1
            )
            seed_voxels[:, 1] = np.clip(
                seed_voxels[:, 1], 0, pam.peak_indices.shape[1] - 1
            )
            seed_voxels[:, 2] = np.clip(
                seed_voxels[:, 2], 0, pam.peak_indices.shape[2] - 1
            )

            seed_peak_indices = pam.peak_indices[
                seed_voxels[:, 0], seed_voxels[:, 1], seed_voxels[:, 2]
            ].astype(int)

            seed_peak_values = pam.peak_values[
                seed_voxels[:, 0], seed_voxels[:, 1], seed_voxels[:, 2]
            ]

            seed_peak_indices = np.clip(seed_peak_indices, 0, len(odf_verts) - 1)
            peak_dirs_at_seeds = odf_verts[seed_peak_indices]

            seed_positions, seed_directions = seeds_directions_pairs(
                seed_positions,
                peak_dirs_at_seeds,
                max_cross=max_cross,
                peak_values=seed_peak_values,
            )
        else:
            peaks_obj = peaks_from_positions(
                seed_positions, None, None, npeaks=1, affine=affine, pmf_gen=pmf_gen
            )
            seed_positions, seed_directions = seeds_directions_pairs(
                seed_positions, peaks_obj, max_cross=max_cross
            )

    return generate_tractogram(
        seed_positions,
        seed_directions,
        sc,
        params,
        pmf_gen,
        affine=affine,
        nbr_threads=nbr_threads,
        buffer_frac=seed_buffer_fraction,
        save_seeds=save_seeds,
    )


def probabilistic_tracking(
    seed_positions,
    sc,
    affine,
    *,
    seed_directions=None,
    sh=None,
    peaks=None,
    sf=None,
    min_len=2,
    max_len=500,
    step_size=0.2,
    voxel_size=None,
    max_angle=20,
    pmf_threshold=0.1,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
    save_seeds=False,
):
    """Probabilistic tracking algorithm.

    Parameters
    ----------
    seed_positions : ndarray
        Seed positions in world space.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seed_directions : ndarray, optional
        Seed directions.
    sh : ndarray, optional
       Spherical Harmonics (SH).
    peaks : ndarray, optional
        Peaks array.
    sf : ndarray, optional
        Spherical Function (SF).
    min_len : int, optional
        Minimum length (mm) of the streamlines.
    max_len : int, optional
        Maximum length (mm) of the streamlines.
    step_size : float, optional
        Step size of the tracking.
    voxel_size : ndarray, optional
        Voxel size.
    max_angle : float, optional
        Maximum angle.
    pmf_threshold : float, optional
        PMF threshold.
    sphere : Sphere, optional
        Sphere.
    basis_type : name of basis
        The basis that ``shcoeff`` are associated with.
        ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    nbr_threads: int, optional
        Number of threads to use for the processing. By default, all available threads
        will be used.
    random_seed: int, optional
        Seed for the random number generator, must be >= 0. A value of greater than 0
        will all produce the same streamline trajectory for a given seed coordinate.
        A value of 0 may produces various streamline tracjectories for a given seed
        coordinate.
    seed_buffer_fraction: float, optional
        Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
        buffer. A value of 0.5 will use half of the seed buffer then the other half.
        a way to reduce memory usage.
    return_all: bool, optional
        True to return all the streamlines, False to return only the streamlines that
        reached the stopping criterion.
    save_seeds: bool, optional
        True to return the seeds with the associated streamline.

    Returns
    -------
    Tractogram

    """
    voxel_size = voxel_size if voxel_size is not None else voxel_sizes(affine)

    params = generate_tracking_parameters(
        "prob",
        min_len=min_len,
        max_len=max_len,
        step_size=step_size,
        voxel_size=voxel_size,
        max_angle=max_angle,
        pmf_threshold=pmf_threshold,
        random_seed=random_seed,
        return_all=return_all,
    )

    return generic_tracking(
        seed_positions,
        seed_directions,
        sc,
        params,
        affine=affine,
        sh=sh,
        peaks=peaks,
        sf=sf,
        sphere=sphere,
        basis_type=basis_type,
        legacy=legacy,
        nbr_threads=nbr_threads,
        seed_buffer_fraction=seed_buffer_fraction,
        save_seeds=save_seeds,
    )


def deterministic_tracking(
    seed_positions,
    sc,
    affine,
    *,
    seed_directions=None,
    sh=None,
    peaks=None,
    sf=None,
    min_len=2,
    max_len=500,
    step_size=0.2,
    voxel_size=None,
    max_angle=20,
    pmf_threshold=0.1,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
    save_seeds=False,
):
    """Deterministic tracking algorithm.

    Parameters
    ----------
    seed_positions : ndarray
        Seed positions in world space.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seed_directions : ndarray, optional
        Seed directions.
    sh : ndarray, optional
        Spherical Harmonics (SH).
    peaks : ndarray, optional
        Peaks array.
    sf : ndarray, optional
        Spherical Function (SF).
    min_len : int, optional
        Minimum length (mm) of the streamlines.
    max_len : int, optional
        Maximum length (mm) of the streamlines.
    step_size : float, optional
        Step size of the tracking.
    voxel_size : ndarray, optional
        Voxel size.
    max_angle : float, optional
        Maximum angle.
    pmf_threshold : float, optional
        PMF threshold.
    sphere : Sphere, optional
        Sphere.
    basis_type : name of basis
        The basis that ``shcoeff`` are associated with.
        ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    nbr_threads: int, optional
        Number of threads to use for the processing. By default, all available threads
        will be used.
    random_seed: int, optional
        Seed for the random number generator, must be >= 0. A value of greater than 0
        will all produce the same streamline trajectory for a given seed coordinate.
        A value of 0 may produces various streamline tracjectories for a given seed
        coordinate.
    seed_buffer_fraction: float, optional
        Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
        buffer. A value of 0.5 will use half of the seed buffer then the other half.
        a way to reduce memory usage.
    return_all: bool, optional
        True to return all the streamlines, False to return only the streamlines that
        reached the stopping criterion.
    save_seeds: bool, optional
        True to return the seeds with the associated streamline.

    Returns
    -------
    Tractogram

    """
    voxel_size = voxel_size if voxel_size is not None else voxel_sizes(affine)

    params = generate_tracking_parameters(
        "det",
        min_len=min_len,
        max_len=max_len,
        step_size=step_size,
        voxel_size=voxel_size,
        max_angle=max_angle,
        pmf_threshold=pmf_threshold,
        random_seed=random_seed,
        return_all=return_all,
    )
    return generic_tracking(
        seed_positions,
        seed_directions,
        sc,
        params,
        affine=affine,
        sh=sh,
        peaks=peaks,
        sf=sf,
        sphere=sphere,
        basis_type=basis_type,
        legacy=legacy,
        nbr_threads=nbr_threads,
        seed_buffer_fraction=seed_buffer_fraction,
        save_seeds=save_seeds,
    )


def ptt_tracking(
    seed_positions,
    sc,
    affine,
    *,
    seed_directions=None,
    sh=None,
    peaks=None,
    sf=None,
    min_len=2,
    max_len=500,
    step_size=0.5,
    voxel_size=None,
    max_angle=10,
    pmf_threshold=0.1,
    probe_length=1.5,
    probe_radius=0,
    probe_quality=7,
    probe_count=1,
    data_support_exponent=1,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
    save_seeds=False,
):
    """Parallel Transport Tractography (PTT) tracking algorithm.

    Parameters
    ----------
    seed_positions : ndarray
        Seed positions in world space.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seed_directions : ndarray, optional
        Seed directions.
    sh : ndarray, optional
        Spherical Harmonics (SH) data.
    peaks : ndarray, optional
        Peaks array
    sf : ndarray, optional
        Spherical Function (SF).
    min_len : int, optional
        Minimum length (mm) of the streamlines.
    max_len : int, optional
        Maximum length (mm) of the streamlines.
    step_size : float, optional
        Step size of the tracking.
    voxel_size : ndarray, optional
        Voxel size.
    max_angle : float, optional
        Maximum angle.
    pmf_threshold : float, optional
        PMF threshold.
    probe_length : float, optional
        Probe length.
    probe_radius : float, optional
        Probe radius.
    probe_quality : int, optional
        Probe quality.
    probe_count : int, optional
        Probe count.
    data_support_exponent : int, optional
        Data support exponent.
    sphere : Sphere, optional
        Sphere.
    basis_type : name of basis
        The basis that ``shcoeff`` are associated with.
        ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    nbr_threads: int, optional
        Number of threads to use for the processing. By default, all available threads
        will be used.
    random_seed: int, optional
        Seed for the random number generator, must be >= 0. A value of greater than 0
        will all produce the same streamline trajectory for a given seed coordinate.
        A value of 0 may produces various streamline tracjectories for a given seed
        coordinate.
    seed_buffer_fraction: float, optional
        Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
        buffer. A value of 0.5 will use half of the seed buffer then the other half.
        a way to reduce memory usage.
    return_all: bool, optional
        True to return all the streamlines, False to return only the streamlines that
        reached the stopping criterion.
    save_seeds: bool, optional
        True to return the seeds with the associated streamline.
    Returns
    -------
    Tractogram

    """
    voxel_size = voxel_size if voxel_size is not None else voxel_sizes(affine)

    params = generate_tracking_parameters(
        "ptt",
        min_len=min_len,
        max_len=max_len,
        step_size=step_size,
        voxel_size=voxel_size,
        max_angle=max_angle,
        pmf_threshold=pmf_threshold,
        random_seed=random_seed,
        probe_length=probe_length,
        probe_radius=probe_radius,
        probe_quality=probe_quality,
        probe_count=probe_count,
        data_support_exponent=data_support_exponent,
        return_all=return_all,
    )
    return generic_tracking(
        seed_positions,
        seed_directions,
        sc,
        params,
        affine=affine,
        sh=sh,
        peaks=peaks,
        sf=sf,
        sphere=sphere,
        basis_type=basis_type,
        legacy=legacy,
        nbr_threads=nbr_threads,
        seed_buffer_fraction=seed_buffer_fraction,
        save_seeds=save_seeds,
    )


def closestpeak_tracking(
    seed_positions,
    sc,
    affine,
    *,
    seed_directions=None,
    sh=None,
    peaks=None,
    sf=None,
    min_len=2,
    max_len=500,
    step_size=0.5,
    voxel_size=None,
    max_angle=60,
    pmf_threshold=0.1,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
    save_seeds=False,
):
    """Closest peak tracking algorithm.

    Parameters
    ----------
    seed_positions : ndarray
        Seed positions in world space.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seed_directions : ndarray, optional
        Seed directions.
    sh : ndarray, optional
        Spherical Harmonics (SH).
    peaks : ndarray, optional
        Peaks array.
    sf : ndarray, optional
        Spherical Function (SF).
    min_len : int, optional
        Minimum length (mm) of the streamlines.
    max_len : int, optional
        Maximum length (mm) of the streamlines.
    step_size : float, optional
        Step size of the tracking.
    voxel_size : ndarray, optional
        Voxel size.
    max_angle : float, optional
        Maximum angle.
    pmf_threshold : float, optional
        PMF threshold.
    sphere : Sphere, optional
        Sphere.
    basis_type : name of basis
        The basis that ``shcoeff`` are associated with.
        ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    nbr_threads: int, optional
        Number of threads to use for the processing. By default, all available threads
        will be used.
    random_seed: int, optional
        Seed for the random number generator, must be >= 0. A value of greater than 0
        will all produce the same streamline trajectory for a given seed coordinate.
        A value of 0 may produces various streamline tracjectories for a given seed
        coordinate.
    seed_buffer_fraction: float, optional
        Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
        buffer. A value of 0.5 will use half of the seed buffer then the other half.
        a way to reduce memory usage.
    return_all: bool, optional
        True to return all the streamlines, False to return only the streamlines that
        reached the stopping criterion.
    save_seeds: bool, optional
        True to return the seeds with the associated streamline.

    Returns
    -------
    Tractogram

    """
    dg = None
    sphere = sphere if sphere is not None else default_sphere
    if sh is not None:
        dg = ClosestPeakDirectionGetter.from_shcoeff(
            sh,
            sphere=sphere,
            max_angle=max_angle,
            pmf_threshold=pmf_threshold,
            basis_type=basis_type,
            legacy=legacy,
        )
    elif sf is not None:
        dg = ClosestPeakDirectionGetter.from_pmf(
            sf, sphere=sphere, max_angle=max_angle, pmf_threshold=pmf_threshold
        )
    else:
        raise ValueError("SH or SF should be defined. Not implemented yet for peaks.")

    # convert length in mm to number of points
    min_len = int(min_len / step_size)
    max_len = int(max_len / step_size)

    return LocalTracking(
        dg,
        sc,
        seed_positions,
        affine,
        step_size=step_size,
        minlen=min_len,
        maxlen=max_len,
        random_seed=random_seed,
        return_all=return_all,
        initial_directions=seed_directions,
        save_seeds=save_seeds,
    )


def bootstrap_tracking(
    seed_positions,
    sc,
    affine,
    *,
    seed_directions=None,
    data=None,
    model=None,
    sh=None,
    peaks=None,
    sf=None,
    min_len=2,
    max_len=500,
    step_size=0.5,
    voxel_size=None,
    max_angle=60,
    pmf_threshold=0.1,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
    save_seeds=False,
):
    """Bootstrap tracking algorithm.

    seed_positions : ndarray
        Seed positions in world space.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seed_directions : ndarray, optional
        Seed directions.
    data : ndarray, optional
        Diffusion data.
    model : Model, optional
        Reconstruction model.
    sh : ndarray, optional
        Spherical Harmonics (SH).
    peaks : ndarray, optional
        Peaks array.
    sf : ndarray, optional
        Spherical Function (SF).
    min_len : int, optional
        Minimum length (mm) of the streamlines.
    max_len : int, optional
        Maximum length (mm) of the streamlines.
    step_size : float, optional
        Step size of the tracking.
    voxel_size : ndarray, optional
        Voxel size.
    max_angle : float, optional
        Maximum angle.
    pmf_threshold : float, optional
        PMF threshold.
    sphere : Sphere, optional
        Sphere.
    basis_type : name of basis
        The basis that ``shcoeff`` are associated with.
        ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    nbr_threads: int, optional
        Number of threads to use for the processing. By default, all available threads
        will be used.
    random_seed: int, optional
        Seed for the random number generator, must be >= 0. A value of greater than 0
        will all produce the same streamline trajectory for a given seed coordinate.
        A value of 0 may produces various streamline tracjectories for a given seed
        coordinate.
    seed_buffer_fraction: float, optional
        Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
        buffer. A value of 0.5 will use half of the seed buffer then the other half.
        a way to reduce memory usage.
    return_all: bool, optional
        True to return all the streamlines, False to return only the streamlines that
        reached the stopping criterion.
    save_seeds: bool, optional
        True to return the seeds with the associated streamline.

    Returns
    -------
    Tractogram

    """
    sphere = sphere if sphere is not None else default_sphere
    if data is None or model is None:
        raise ValueError("Data and model should be defined.")

    dg = BootDirectionGetter.from_data(
        data,
        model,
        max_angle=max_angle,
    )

    # convert length in mm to number of points
    min_len = int(min_len / step_size)
    max_len = int(max_len / step_size)

    return LocalTracking(
        dg,
        sc,
        seed_positions,
        affine,
        step_size=step_size,
        minlen=min_len,
        maxlen=max_len,
        random_seed=random_seed,
        return_all=return_all,
        initial_directions=seed_directions,
        save_seeds=save_seeds,
    )


def eudx_tracking(
    seed_positions,
    sc,
    affine,
    *,
    seed_directions=None,
    sh=None,
    peaks=None,
    sf=None,
    pam=None,
    max_cross=None,
    min_len=2,
    max_len=500,
    step_size=0.5,
    voxel_size=None,
    max_angle=60,
    pmf_threshold=0.0239,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=None,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
    save_seeds=False,
):
    """EuDX tracking algorithm.

    seed_positions : ndarray
        Seed positions in world space.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seed_directions : ndarray, optional
        Seed directions.
    sh : ndarray, optional
        Spherical Harmonics (SH).
    peaks : ndarray, optional
        Peaks array.
    sf : ndarray, optional
        Spherical Function (SF).
    pam : PeakAndMetrics, optional
        Peaks and Metrics object
    max_cross : int or None, optional
        The maximum number of directions to track from each seed in crossing
        voxels. By default (None), all peak directions are tracked.
    min_len : int, optional
        Minimum length (mm) of the streamlines.
    max_len : int, optional
        Maximum length (mm) of the streamlines.
    step_size : float, optional
        Step size of the tracking.
    voxel_size : ndarray, optional
        Voxel size.
    max_angle : float, optional
        Maximum angle.
    pmf_threshold : float, optional
        PMF threshold.
    sphere : Sphere, optional
        Sphere.
    basis_type : name of basis
        The basis that ``shcoeff`` are associated with.
        ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    nbr_threads: int or None, optional
        Number of threads to use for parallel processing. If None (default),
        uses min(8, cpu_count) threads which is optimal for most systems due
        to memory bandwidth limitations. Set to 0 to use all available cores.
    random_seed: int, optional
        Seed for the random number generator, must be >= 0. A value of greater than 0
        will all produce the same streamline trajectory for a given seed coordinate.
        A value of 0 may produces various streamline tracjectories for a given seed
        coordinate.
    seed_buffer_fraction: float, optional
        Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
        buffer. A value of 0.5 will use half of the seed buffer then the other half.
        a way to reduce memory usage.
    return_all: bool, optional
        True to return all the streamlines, False to return only the streamlines that
        reached the stopping criterion.
    save_seeds: bool, optional
        True to return the seeds with the associated streamline.

    Returns
    -------
    Tractogram

    """
    sphere = sphere if sphere is not None else default_sphere
    if pam is None:
        raise ValueError("PAM should be defined.")

    # Handle thread count for parallel processing
    if nbr_threads is None:
        # Default to min(8, cpu_count) for optimal memory bandwidth utilization
        cpu_count = os.cpu_count() or 1
        nbr_threads = min(8, cpu_count)

    # Get voxel size
    voxel_size = voxel_size if voxel_size is not None else voxel_sizes(affine)

    params = generate_tracking_parameters(
        "eudx",
        min_len=min_len,
        max_len=max_len,
        step_size=step_size,
        voxel_size=voxel_size,
        max_angle=max_angle,
        qa_thr=pmf_threshold,
        ang_thr=max_angle,
        total_weight=0.5,
        random_seed=random_seed,
        return_all=return_all,
    )

    return generic_tracking(
        seed_positions,
        seed_directions,
        sc,
        params,
        affine=affine,
        peaks=pam,
        sphere=sphere,
        max_cross=max_cross,
        nbr_threads=nbr_threads,
        seed_buffer_fraction=seed_buffer_fraction,
        save_seeds=save_seeds,
    )


def pft_tracking(
    seed_positions,
    sc,
    affine,
    *,
    seed_directions=None,
    sh=None,
    peaks=None,
    sf=None,
    pam=None,
    max_cross=None,
    min_len=2,
    max_len=500,
    step_size=0.2,
    voxel_size=None,
    max_angle=20,
    pmf_threshold=0.1,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
    pft_back_tracking_dist=2,
    pft_front_tracking_dist=1,
    pft_max_trial=20,
    particle_count=15,
    save_seeds=False,
    min_wm_pve_before_stopping=0,
    unidirectional=False,
    randomize_forward_direction=False,
):
    """Particle Filtering Tracking (PFT) tracking algorithm.

    seed_positions : ndarray
        Seed positions in world space.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seed_directions : ndarray, optional
        Seed directions.
    sh : ndarray, optional
        Spherical Harmonics (SH).
    peaks : ndarray, optional
        Peaks array.
    sf : ndarray, optional
        Spherical Function (SF).
    pam : PeakAndMetrics, optional
        Peaks and Metrics object.
    max_cross : int, optional
        Maximum number of crossing fibers.
    min_len : int, optional
        Minimum length (mm) of the streamlines.
    max_len : int, optional
        Maximum length (mm) of the streamlines.
    step_size : float, optional
        Step size of the tracking.
    voxel_size : ndarray, optional
        Voxel size.
    max_angle : float, optional
        Maximum angle.
    pmf_threshold : float, optional
        PMF threshold.
    sphere : Sphere, optional
        Sphere.
    basis_type : name of basis
        The basis that ``shcoeff`` are associated with.
        ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    nbr_threads: int, optional
        Number of threads to use for the processing. By default, all available threads
        will be used.
    random_seed: int, optional
        Seed for the random number generator, must be >= 0. A value of greater than 0
        will all produce the same streamline trajectory for a given seed coordinate.
        A value of 0 may produces various streamline tracjectories for a given seed
        coordinate.
    seed_buffer_fraction: float, optional
        Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
        buffer. A value of 0.5 will use half of the seed buffer then the other half.
        a way to reduce memory usage.
    return_all: bool, optional
        True to return all the streamlines, False to return only the streamlines that
        reached the stopping criterion.
    pft_back_tracking_dist : float, optional
        Back tracking distance.
    pft_front_tracking_dist : float, optional
        Front tracking distance.
    pft_max_trial : int, optional
        Maximum number of trials.
    particle_count : int, optional
        Number of particles.
    save_seeds: bool, optional
        True to return the seeds with the associated streamline.
    min_wm_pve_before_stopping : float, optional
        Minimum white matter partial volume estimation before stopping.
    unidirectional : bool, optional
        True to use unidirectional tracking.
    randomize_forward_direction : bool, optional
        True to randomize forward direction

    Returns
    -------
    Tractogram

    """
    sphere = sphere if sphere is not None else default_sphere

    dg = None
    if sh is not None:
        dg = ProbabilisticDirectionGetter.from_shcoeff(
            sh,
            max_angle=max_angle,
            sphere=sphere,
            sh_to_pmf=True,
            pmf_threshold=pmf_threshold,
            basis_type=basis_type,
            legacy=legacy,
        )
    elif sf is not None:
        dg = ProbabilisticDirectionGetter.from_pmf(
            sf, max_angle=max_angle, sphere=sphere, pmf_threshold=pmf_threshold
        )
    elif pam is not None and sh is None:
        sh = pam.shm_coeff
    else:
        msg = "SH, SF or PAM should be defined. Not implemented yet for peaks."
        raise ValueError(msg)

    # convert length in mm to number of points
    min_len = int(min_len / step_size)
    max_len = int(max_len / step_size)

    return ParticleFilteringTracking(
        dg,
        sc,
        seed_positions,
        affine,
        max_cross=max_cross,
        step_size=step_size,
        minlen=min_len,
        maxlen=max_len,
        pft_back_tracking_dist=pft_back_tracking_dist,
        pft_front_tracking_dist=pft_front_tracking_dist,
        particle_count=particle_count,
        pft_max_trial=pft_max_trial,
        return_all=return_all,
        random_seed=random_seed,
        initial_directions=seed_directions,
        save_seeds=save_seeds,
        min_wm_pve_before_stopping=min_wm_pve_before_stopping,
        unidirectional=unidirectional,
        randomize_forward_direction=randomize_forward_direction,
    )
