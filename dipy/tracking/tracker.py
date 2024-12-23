import numpy as np

from dipy.data import default_sphere
from dipy.direction.pmf import SHCoeffPmfGen, SimplePmfGen
from dipy.tracking.fast_tracking import generate_tractogram
from dipy.tracking.tracking_parameters import generate_tracking_parameters
from dipy.tracking.utils import seeds_directions_pairs

# def custom_tracking(seed_positons, seed_directions, *,
#                    tracker_parameters=None, tractogram_func=None,
#                    postprocess_streamline=None, buffer_perc=1.0):
#     tractogram_func = tractogram_func or generate_tractogram
#     if tractogram_func and 'cython' in type(tractogram_func):
#         for streamlines in tractogram_func(seed_positons, seed_directions,
#                                            tracker_parameters,
#                                            buffer_prec=buffer_perc):
#             yield streamlines

#     return generate_tractogram_py(seed_positons, seed_directions, tracker_parameters)


def generic_tracking(
    seeds_positions,
    seeds_directions,
    sc,
    params,
    *,
    sh=None,
    peaks=None,
    sf=None,
    sphere=None,
    basis_type=None,
    legacy=True,
):
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
            f"Variables initialized: {', '.join(initialized_pmf)}"
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
            raise ValueError("A sphere should be defined when using SF.")

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
        # TODO: Get peaks
        # peaks = peaks_from_positions(seeds_positions, GT_ODF, sphere, npeaks=1,
        # affine=affine)
        seeds_directions = seeds_directions_pairs(seeds_positions, peaks, max_cross=1)

    return generate_tractogram(seeds_positions, seeds_directions, sc, params, pmf_gen)


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

    Returns
    -------
    Tractogram

    """
    voxel_size = voxel_size or np.ones(3)
    params = generate_tracking_parameters(
        "prob",
        max_len=max_len,
        step_size=step_size,
        voxel_size=voxel_size,
        max_angle=max_angle,
        pmf_threshold=pmf_threshold,
    )

    return generic_tracking(
        seeds_positions,
        seeds_directions,
        sc,
        params,
        sh=sh,
        peaks=peaks,
        sf=sf,
        sphere=sphere,
        basis_type=basis_type,
        legacy=legacy,
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

    Returns
    -------
    Tractogram

    """
    voxel_size = voxel_size or np.ones(3)
    params = generate_tracking_parameters(
        "prob",
        max_len=max_len,
        step_size=step_size,
        voxel_size=voxel_size,
        max_angle=max_angle,
        pmf_threshold=pmf_threshold,
    )
    return generic_tracking(
        seeds_positions,
        seeds_directions,
        sc,
        params,
        sh=sh,
        peaks=peaks,
        sf=sf,
        sphere=sphere,
        basis_type=basis_type,
        legacy=legacy,
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
    max_angle=20,
    pmf_threshold=0.1,
    probe_length=0.5,
    probe_radius=0,
    probe_quality=3,
    probe_count=1,
    data_support_exponent=1,
    sphere=None,
    basis_type=None,
    legacy=True,
):
    """Probabilistic Particle Tracing (PPT) tracking algorithm.

    Parameters
    ----------
    seeds_positions : ndarray
        Seeds positions.
    sc : StoppingCriterion

    seeds_directions : ndarray, optional
        Seeds directions.
    pmf : ndarray, optional
        Probability Mass Function (PMF).
    peaks : ndarray, optional
        Peaks array
    sf : ndarray, optional
        Spherical Function (SF).
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

    Returns
    -------
    Tractogram

    """
    voxel_size = voxel_size or np.ones(3)
    params = generate_tracking_parameters(
        "ptt",
        max_len=max_len,
        step_size=step_size,
        voxel_size=voxel_size,
        max_angle=max_angle,
        pmf_threshold=pmf_threshold,
        probe_length=probe_length,
        probe_radius=probe_radius,
        probe_quality=probe_quality,
        probe_count=probe_count,
        data_support_exponent=data_support_exponent,
    )
    return generic_tracking(
        seeds_positions,
        seeds_directions,
        sc,
        params,
        sh=sh,
        peaks=peaks,
        sf=sf,
        sphere=sphere,
        basis_type=basis_type,
        legacy=legacy,
    )
