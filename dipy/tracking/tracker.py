from nibabel.affines import voxel_sizes
import numpy as np

from dipy.data import SPHERE_FILES, default_sphere, get_sphere
from dipy.direction.peaks import peaks_from_positions
from dipy.direction.pmf import SHCoeffPmfGen, SimplePmfGen
from dipy.tracking.fast_tracking import generate_tractogram
from dipy.tracking.tracker_parameters import generate_tracking_parameters
from dipy.tracking.utils import seeds_directions_pairs


def generic_tracking(
    seeds_positions,
    seeds_directions,
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
    nbr_threads=0,
    seed_buffer_fraction=1.0,
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

    if selected_pmf["name"] == "sf":
        if sphere is None:
            nb_vertices = sf.shape[-1]
            all_spheres = list(SPHERE_FILES.keys())
            all_spheres.remove("symmetric724")
            found = [sph for sph in all_spheres if str(nb_vertices) in sph]
            if len(found) == 1:
                sphere = get_sphere(name=found[0])
            else:
                raise ValueError("A sphere should be defined when using SF.")

    if selected_pmf["name"] == "peaks":
        raise NotImplementedError("Peaks are not yet implemented.")

    sphere = sphere or default_sphere

    kwargs = {}
    if selected_pmf["name"] == "sh":
        kwargs = {"basis_type": basis_type, "legacy": legacy}

    pmf_gen = selected_pmf["cls"](
        np.asarray(selected_pmf["value"], dtype=float), sphere, **kwargs
    )

    if seeds_directions is not None:
        if not isinstance(seeds_directions, (np.ndarray, list)):
            raise ValueError("seeds_directions should be a numpy array or a list.")
        elif isinstance(seeds_directions, list):
            seeds_directions = np.array(seeds_directions)

        if not np.array_equal(seeds_directions.shape, seeds_positions.shape):
            raise ValueError(
                "seeds_directions and seeds_positions should have the same shape."
            )
    else:
        peaks = peaks_from_positions(
            seeds_positions, None, None, npeaks=1, affine=affine, pmf_gen=pmf_gen
        )
        seeds_positions, seeds_directions = seeds_directions_pairs(
            seeds_positions, peaks, max_cross=1
        )

    return generate_tractogram(
        seeds_positions,
        seeds_directions,
        sc,
        params,
        pmf_gen,
        affine=affine,
        nbr_threads=nbr_threads,
        buffer_frac=seed_buffer_fraction,
    )


def probabilistic_tracking(
    seeds_positions,
    sc,
    affine,
    *,
    seeds_directions=None,
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
):
    """Probabilistic tracking algorithm.

    Parameters
    ----------
    seeds_positions : ndarray
        Seeds positions.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seeds_directions : ndarray, optional
        Seeds directions.
    sh : ndarray, optional
       Spherical Harmonics (SH).
    peaks : ndarray, optional
        Peaks array.
    sf : ndarray, optional
        Spherical Function (SF).
    min_len : int, optional
        Minimum length (Nb points) of the streamlines.
    max_len : int, optional
        Maximum length (Nb points) of the streamlines.
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
        seeds_positions,
        seeds_directions,
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
    )


def deterministic_tracking(
    seeds_positions,
    sc,
    affine,
    *,
    seeds_directions=None,
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
):
    """Deterministic tracking algorithm.

    Parameters
    ----------
    seeds_positions : ndarray
        Seeds positions.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seeds_directions : ndarray, optional
        Seeds directions.
    sh : ndarray, optional
        Spherical Harmonics (SH).
    peaks : ndarray, optional
        Peaks array.
    sf : ndarray, optional
        Spherical Function (SF).
    min_len : int, optional
        Minimum length (nb points) of the streamlines.
    max_len : int, optional
        Maximum length (nb points) of the streamlines.
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
        seeds_positions,
        seeds_directions,
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
    )


def ptt_tracking(
    seeds_positions,
    sc,
    affine,
    *,
    seeds_directions=None,
    sh=None,
    peaks=None,
    sf=None,
    min_len=2,
    max_len=500,
    step_size=0.2,
    voxel_size=None,
    max_angle=15,
    pmf_threshold=0.1,
    probe_length=0.5,
    probe_radius=0,
    probe_quality=3,
    probe_count=1,
    data_support_exponent=1,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
):
    """Probabilistic Particle Tracing (PPT) tracking algorithm.

    Parameters
    ----------
    seeds_positions : ndarray
        Seeds positions.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seeds_directions : ndarray, optional
        Seeds directions.
    sh : ndarray, optional
        Spherical Harmonics (SH) data.
    peaks : ndarray, optional
        Peaks array
    sf : ndarray, optional
        Spherical Function (SF).
    min_len : int, optional
        Minimum length of the streamlines.
    max_len : int, optional
        Maximum length of the streamlines.
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
        seeds_positions,
        seeds_directions,
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
    )
