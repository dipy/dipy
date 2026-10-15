"""Interactive prompts for user-guided pipeline configuration."""

from dataclasses import dataclass
import importlib.util
import os

from dipy.utils.logging import logger
from dipy.workflows.templates import PREDEFINED_PIPELINES, list_predefined_pipelines

# =============================================================================
# Data Analysis Functions
# =============================================================================


@dataclass
class DataCharacteristics:
    """Characteristics of diffusion MRI data.

    Attributes
    ----------
    num_bvals : int
        Number of unique b-values.
    unique_bvals : list[float]
        Sorted list of unique b-values.
    is_single_shell : bool
        True if data has only one non-zero b-value shell.
    is_multi_shell : bool
        True if data has multiple non-zero b-value shells.
    supports_dki : bool
        True if data has sufficient b-values for DKI (3+).
    recommended_pipeline : str
        Recommended pipeline name for this data.
    recommended_methods : list[str]
        Recommended reconstruction methods for this data.
    """

    num_bvals: int
    unique_bvals: list
    is_single_shell: bool
    is_multi_shell: bool
    supports_dki: bool
    recommended_pipeline: str
    recommended_methods: list


def analyze_bvals(*, bvals_file, b0_threshold=50):
    """Analyze b-values file to determine data characteristics.

    Parameters
    ----------
    bvals_file : str
        Path to b-values file.
    b0_threshold : float, optional
        Threshold for considering a b-value as b0.

    Returns
    -------
    DataCharacteristics
        Analysis results with data characteristics and recommendations.

    Raises
    ------
    FileNotFoundError
        If bvals_file does not exist.
    ValueError
        If bvals_file cannot be parsed.
    """
    if not os.path.exists(bvals_file):
        raise FileNotFoundError(f"B-values file not found: {bvals_file}")

    try:
        with open(bvals_file) as f:
            content = f.read()
            bvals = [float(x) for x in content.split()]
    except (OSError, ValueError) as e:
        raise ValueError(f"Failed to parse b-values file: {e}") from e

    if not bvals:
        raise ValueError(f"B-values file is empty: {bvals_file}")

    unique_bvals = sorted(set(bvals))
    non_zero_bvals = [b for b in unique_bvals if b > b0_threshold]

    num_bvals = len(unique_bvals)
    is_single_shell = len(non_zero_bvals) == 1
    is_multi_shell = len(non_zero_bvals) > 1
    supports_dki = num_bvals >= 3

    if is_multi_shell and supports_dki:
        recommended_pipeline = "full"
        recommended_methods = ["dti", "dki", "csd", "csa", "gqi"]
    elif is_single_shell and non_zero_bvals and non_zero_bvals[0] >= 1000:
        recommended_pipeline = "tractography"
        recommended_methods = ["dti", "csd"]
    elif is_single_shell:
        recommended_pipeline = "dti_only"
        recommended_methods = ["dti"]
    else:
        recommended_pipeline = "preprocessing"
        recommended_methods = ["dti"]

    return DataCharacteristics(
        num_bvals=num_bvals,
        unique_bvals=unique_bvals,
        is_single_shell=is_single_shell,
        is_multi_shell=is_multi_shell,
        supports_dki=supports_dki,
        recommended_pipeline=recommended_pipeline,
        recommended_methods=recommended_methods,
    )


def suggest_pipeline(*, data_chars):
    """Generate human-readable pipeline suggestion.

    Parameters
    ----------
    data_chars : DataCharacteristics
        Data characteristics from analyze_bvals().

    Returns
    -------
    str
        Human-readable suggestion message.
    """
    bvals_str = str(data_chars.unique_bvals)
    msg = f"Your data has {data_chars.num_bvals} unique b-values: {bvals_str}\n"
    msg += f"Recommended pipeline: '{data_chars.recommended_pipeline}'\n"
    msg += f"Recommended methods: {', '.join(data_chars.recommended_methods)}"
    return msg


def print_data_summary(*, data_chars):
    """Print a formatted summary of data characteristics.

    Parameters
    ----------
    data_chars : DataCharacteristics
        Data characteristics from analyze_bvals().
    """
    logger.info("=" * 60)
    logger.info("Data Characteristics")
    logger.info("=" * 60)
    logger.info(f"Number of unique b-values: {data_chars.num_bvals}")
    logger.info(f"B-values: {data_chars.unique_bvals}")

    if data_chars.is_single_shell:
        logger.info("Acquisition type: Single-shell")
    elif data_chars.is_multi_shell:
        logger.info("Acquisition type: Multi-shell")

    supports_dki_str = "Yes" if data_chars.supports_dki else "No"
    logger.info(f"Supports DKI: {supports_dki_str}")

    logger.info(f"\nRecommended pipeline: {data_chars.recommended_pipeline}")
    methods_str = ", ".join(data_chars.recommended_methods)
    logger.info(f"Recommended methods: {methods_str}")
    logger.info("=" * 60)


# =============================================================================
# Reconstruction Method Definitions
# =============================================================================


@dataclass
class ReconMethod:
    """Information about a reconstruction method.

    Attributes
    ----------
    code : str
        Method code (e.g., 'dti', 'dki').
    name : str
        Full name of the method.
    description : str
        Brief description.
    min_bvals : int
        Minimum number of b-values required.
    best_for : str
        Description of optimal use case.
    """

    code: str
    name: str
    description: str
    min_bvals: int
    best_for: str


# Module-level constant defining reconstruction methods
RECONSTRUCTION_METHODS = {
    "dti": ReconMethod(
        "dti", "DTI", "Diffusion Tensor Imaging (always runs)", 2, "all data"
    ),
    "dki": ReconMethod(
        "dki", "DKI", "Diffusion Kurtosis Imaging", 3, "multi-shell (3+ b-values)"
    ),
    "csd": ReconMethod(
        "csd", "CSD", "Constrained Spherical Deconvolution", 2, "single-shell HARDI"
    ),
    "csa": ReconMethod("csa", "CSA", "Constant Solid Angle", 2, "q-ball imaging"),
    "gqi": ReconMethod(
        "gqi", "GQI", "Generalized Q-Sampling Imaging", 2, "multi-shell"
    ),
    "mapmri": ReconMethod(
        "mapmri", "MAPMRI", "Mean Apparent Propagator MRI", 2, "multi-shell"
    ),
    "force": ReconMethod(
        "force",
        "FORCE",
        "FORward modeling for Complex microstructure Estimation ",
        2,
        "all data",
    ),
}


def check_method_compatibility(*, method_code, data_chars):
    """Check if a reconstruction method is compatible with the data.

    Parameters
    ----------
    method_code : str
        Method code (e.g., 'dki', 'csd').
    data_chars : DataCharacteristics
        Data characteristics from analyze_bvals().

    Returns
    -------
    tuple[bool, str]
        (is_compatible, warning_message)
    """
    if method_code not in RECONSTRUCTION_METHODS:
        return False, f"Unknown method: {method_code}"

    method = RECONSTRUCTION_METHODS[method_code]

    if method.code == "dki" and not data_chars.supports_dki:
        msg = (
            f"DKI requires {method.min_bvals}+ b-values, "
            f"but data has {data_chars.num_bvals}"
        )
        return False, msg

    return True, ""


def interactive_preprocessing_selection():
    """Interactive prompt for preprocessing step selection.

    Returns
    -------
    dict
        Dictionary mapping step names to boolean (selected or not).
    """
    print("\n" + "=" * 60)
    print("Preprocessing Steps Selection")
    print("=" * 60)

    preprocessing_steps = {
        "denoise_nlmeans": "Denoising (NLMeans)",
        "denoise_mppca": "Denoising (MPPCA)",
        "denoise_lpca": "Denoising (LPCA)",
        "denoise_patch2self": "Denoising (Patch2Self)",
        "reslice": "Reslice to isotropic voxels",
        "gibbs": "Gibbs ringing removal",
        "motion_correction": "Motion correction (eddy)",
        "bias_correction": "Bias field correction",
    }
    print(
        "\nSelect preprocessing steps to include (Brain Mask Extraction always runs):"
    )
    for i, (_step_name, desc) in enumerate(preprocessing_steps.items(), 1):
        print(f"  {i}. {desc}")

    print(
        "\nEnter numbers (comma-separated, e.g., 4,5,6,7,8) or press Enter for "
        "recommended defaults:"
    )
    choice = input("Steps: ").strip()

    # Denoising method keys
    denoising_methods = {
        "denoise_nlmeans",
        "denoise_mppca",
        "denoise_lpca",
        "denoise_patch2self",
    }

    # Default recommended pipeline
    default_steps = {
        "reslice",
        "denoise_patch2self",
        "gibbs",
        "motion_correction",
        "bias_correction",
        "mask",
    }

    selected = {}
    if not choice:
        # Default selection
        print(
            "⚠ Using recommended defaults: Reslice, Patch2Self denoising, "
            "Gibbs removal, Motion correction, Bias correction, and Brain masking"
        )
        for step_name in preprocessing_steps:
            selected[step_name] = step_name in default_steps
    else:
        # Parse user selection
        while True:
            try:
                step_list = list(preprocessing_steps.keys())
                indices = [int(x.strip()) for x in choice.split(",") if x.strip()]
                for step_name in preprocessing_steps:
                    selected[step_name] = False
                for idx in indices:
                    if 0 < idx <= len(step_list):
                        selected[step_list[idx - 1]] = True

                # Check if multiple denoising methods are selected
                selected_denoise = [
                    method
                    for method in denoising_methods
                    if selected.get(method, False)
                ]

                if len(selected_denoise) > 1:
                    print(
                        f"\n⚠ Error: You selected {len(selected_denoise)} denoising "
                        "methods, but only one can be used."
                    )
                    print("Selected denoising methods:")
                    for method in selected_denoise:
                        method_desc = preprocessing_steps[method]
                        print(f"  - {method_desc}")

                    print("\nPlease choose only ONE denoising method:")
                    for i, method in enumerate(selected_denoise, 1):
                        print(f"  {i}. {preprocessing_steps[method]}")

                    denoise_choice = input(
                        f"Enter choice [1-{len(selected_denoise)}]: "
                    ).strip()
                    try:
                        denoise_idx = int(denoise_choice)
                        if 1 <= denoise_idx <= len(selected_denoise):
                            # Deselect all denoising methods except the chosen one
                            chosen_method = selected_denoise[denoise_idx - 1]
                            for method in denoising_methods:
                                selected[method] = method == chosen_method
                            break
                        else:
                            print("Invalid choice")
                    except ValueError:
                        print("Invalid input")
                else:
                    # Valid selection (0 or 1 denoising method)
                    break

            except (ValueError, IndexError):
                print("Invalid input format. Please enter comma-separated numbers.")
                choice = input("Steps: ").strip()
                if not choice:
                    # If they just press Enter, use default
                    print(
                        "⚠ Using recommended defaults: Reslice, Patch2Self denoising, "
                        "Gibbs removal, Motion correction, Bias correction, "
                        "and Brain masking"
                    )
                    for step_name in preprocessing_steps:
                        selected[step_name] = step_name in default_steps
                    break

    selected["mask"] = True
    return selected


def interactive_method_selection(*, data_chars):
    """Interactive prompt for reconstruction method selection.

    Parameters
    ----------
    data_chars : DataCharacteristics
        Data characteristics for compatibility checking.

    Returns
    -------
    list[str]
        List of selected method codes (DTI always included).
    """
    logger.info("\nSelect reconstruction methods (DTI always runs):")

    method_list = ["force", "dki", "csd", "csa", "gqi", "mapmri"]
    for i, method_code in enumerate(method_list, 1):
        method = RECONSTRUCTION_METHODS[method_code]
        compatible, warning = check_method_compatibility(
            method_code=method_code, data_chars=data_chars
        )
        status = "✓ compatible" if compatible else f"✗ {warning}"
        logger.info(f"  {i}. {method.name} - {method.description} ({status})")

    logger.info("\nEnter numbers (comma-separated, e.g., 1,2) or press Enter for all:")
    choice = input("Methods: ").strip()

    if not choice:
        selected = method_list.copy()
    elif choice == "*":
        selected = []
    else:
        try:
            indices = [int(x.strip()) for x in choice.split(",") if x.strip()]
            selected = [
                method_list[i - 1] for i in indices if 0 < i <= len(method_list)
            ]
        except (ValueError, IndexError):
            print("Invalid input, selecting all methods")
            selected = method_list.copy()

    final_selected = []
    for method_code in selected:
        compatible, warning = check_method_compatibility(
            method_code=method_code, data_chars=data_chars
        )
        if compatible:
            final_selected.append(method_code)
        else:
            print(f"Skipping {method_code}: {warning}")

    return ["dti"] + final_selected


def interactive_tracking_selection(*, available_methods):
    """Interactive prompt for tracking method selection.

    Parameters
    ----------
    available_methods : list[str]
        List of available reconstruction methods with peaks/PAM.

    Returns
    -------
    list[str]
        List of selected methods for tracking.
    """
    if len(available_methods) <= 1:
        return available_methods

    print("\nMultiple reconstruction methods available for tracking:")
    for i, method in enumerate(available_methods, 1):
        print(f"  {i}. {method}")

    print("Enter numbers (comma-separated, e.g., 1,2) or press Enter for all:")
    choice = input("Tracking methods: ").strip()

    if not choice:
        return available_methods

    try:
        indices = [int(x.strip()) for x in choice.split(",") if x.strip()]
        selected = [
            available_methods[i - 1] for i in indices if 0 < i <= len(available_methods)
        ]
        return selected or available_methods
    except (ValueError, IndexError):
        print("Invalid input, using all methods")
        return available_methods


def interactive_registration_selection(*, available_tractograms):
    """Interactive prompt for registration method selection.

    Parameters
    ----------
    available_tractograms : list[str]
        List of available tractogram methods.

    Returns
    -------
    list[str]
        List of selected methods for registration.
    """
    if len(available_tractograms) <= 1:
        return available_tractograms

    print("\nMultiple tracking outputs available for registration:")
    for i, method in enumerate(available_tractograms, 1):
        print(f"  {i}. {method}")

    print("Enter numbers (comma-separated, e.g., 1,2) or press Enter for all:")
    choice = input("Registration methods: ").strip()

    if not choice:
        return available_tractograms

    try:
        indices = [int(x.strip()) for x in choice.split(",") if x.strip()]
        selected = [
            available_tractograms[i - 1]
            for i in indices
            if 0 < i <= len(available_tractograms)
        ]
        return selected or available_tractograms
    except (ValueError, IndexError):
        print("Invalid input, using all methods")
        return available_tractograms


def interactive_segmentation_selection(*, available_registered):
    """Interactive prompt for segmentation method selection.

    Parameters
    ----------
    available_registered : list[str]
        List of available registered tractogram methods.

    Returns
    -------
    list[str]
        List of selected methods for segmentation.
    """
    if len(available_registered) <= 1:
        return available_registered

    print("\nMultiple registration outputs available for segmentation:")
    for i, method in enumerate(available_registered, 1):
        print(f"  {i}. {method}")

    print("Enter numbers (comma-separated, e.g., 1,2) or press Enter for all:")
    choice = input("Segmentation methods: ").strip()

    if not choice:
        return available_registered

    try:
        indices = [int(x.strip()) for x in choice.split(",") if x.strip()]
        selected = [
            available_registered[i - 1]
            for i in indices
            if 0 < i <= len(available_registered)
        ]
        return selected or available_registered
    except (ValueError, IndexError):
        print("Invalid input, using all methods")
        return available_registered


def check_synthseg_available():
    """Check if SynthSeg is available on the system.

    PyTorch is looked up without being imported: loading its ``libomp`` in
    this process is what running SynthSeg in an isolated subprocess avoids.

    Returns
    -------
    bool
        True if PyTorch is installed.
    """
    return importlib.util.find_spec("torch") is not None


def interactive_brain_extraction_method():
    """Interactive prompt for brain extraction method.

    Returns
    -------
    str
        Selected method ('synthseg' or 'median_otsu').
    """
    has_synthseg = check_synthseg_available()

    if not has_synthseg:
        print("\nSynthSeg not found, using median_otsu")
        return "median_otsu"

    print("\nChoose brain extraction method:")
    print("  1. SynthSeg - More accurate, requires pytorch installed")
    print("  2. median_otsu - Good accuracy, no extra dependencies")

    choice = input("Enter choice [1-2]: ").strip()

    if choice == "1":
        return "synthseg"
    elif choice == "2":
        return "median_otsu"
    else:
        print("Invalid choice, defaulting to median_otsu")
        return "median_otsu"


def interactive_mapmri_timing():
    """Interactive prompt for the diffusion timing MAPMRI needs.

    Returns
    -------
    tuple of float
        ``(small_delta, big_delta)``, the gradient pulse duration and the
        time between the two pulses, in seconds.
    """
    print("\nMAPMRI needs the diffusion timing of the acquisition, in seconds")
    print("(e.g., small delta 0.0157 and big delta 0.0365).")
    timing = []
    for label in ("Small delta (gradient duration)", "Big delta (pulse separation)"):
        while True:
            answer = input(f"{label} [s]: ").strip()
            try:
                value = float(answer)
            except ValueError:
                print("Invalid input. Please enter a positive number of seconds.")
                continue
            if value > 0:
                timing.append(value)
                break
            print("Invalid input. Please enter a positive number of seconds.")
    return tuple(timing)


def _require_mask(*, method, mask_output):
    """Check that a reconstruction needing a brain mask gets one.

    Parameters
    ----------
    method : str
        Display name of the reconstruction method.
    mask_output : str or None
        Reference to the brain mask, None when no mask stage is built.

    Raises
    ------
    ValueError
        If ``mask_output`` is empty.
    """
    if not mask_output:
        raise ValueError(
            f"{method} fitting requires a brain mask. Please include the "
            "'Brain Mask Extraction' preprocessing step to extract a brain mask."
        )


_MASKED_FITS = {
    "dti": ("DTI", "dipy_fit_dti"),
    "dki": ("DKI", "dipy_fit_dki"),
    "csd": ("CSD", "dipy_fit_csd"),
    "csa": ("CSA", "dipy_fit_csa"),
    "gqi": ("GQI", "dipy_fit_gqi"),
    "force": ("FORCE", "dipy_fit_force"),
}


def pipeline_config_from_choices(
    *,
    preprocessing,
    recon_methods,
    brain_extraction_method,
    tracking_methods,
    mapmri_timing=None,
):
    """Build a pipeline configuration from the answers of the user.

    Preprocessing stages follow the order of the predefined pipelines:
    denoising, Gibbs unringing, motion correction, bias field correction,
    reslicing, then brain masking. Once motion correction ran, every later
    stage reads its rotated b-vectors.

    Parameters
    ----------
    preprocessing : dict
        Step name to whether it was selected, from
        :func:`interactive_preprocessing_selection`.
    recon_methods : list of str
        Reconstruction method codes, from :func:`interactive_method_selection`.
    brain_extraction_method : str
        Method of the ``dipy_brain_mask`` stage.
    tracking_methods : list of str
        Reconstruction method codes whose peaks are tracked.
    mapmri_timing : tuple of float, optional
        ``(small_delta, big_delta)`` in seconds, required when
        ``recon_methods`` holds ``mapmri``.

    Returns
    -------
    dict
        Pipeline configuration dictionary with [[pipeline]] sections.

    Raises
    ------
    ValueError
        If a reconstruction needs a brain mask and none is computed, if
        MAPMRI is selected without its timing, or if tracking is requested
        without DTI, whose FA map stops the tracking.
    """
    config = {
        "General": {
            "name": "interactive_pipeline",
            "description": "Interactively configured pipeline",
            "version": "1.0.0",
            "author": "DIPY User",
        },
        "io": {
            "dwi": "",
            "bvals": "",
            "bvecs": "",
            "t1w": "",
            "bids_folder": "",
            "out_dir": ".",
            "out_report": "reports.toml",
        },
        "pipeline": [],
    }
    stages = config["pipeline"]
    current_input = "${io.dwi}"
    current_bvecs = "${io.bvecs}"

    denoisers = (
        ("denoise_nlmeans", "dipy_denoise_nlmeans", {}),
        ("denoise_mppca", "dipy_denoise_mppca", {}),
        (
            "denoise_lpca",
            "dipy_denoise_lpca",
            {"bvalues_files": "${io.bvals}", "bvectors_files": "${io.bvecs}"},
        ),
        (
            "denoise_patch2self",
            "dipy_denoise_patch2self",
            {"bval_files": "${io.bvals}"},
        ),
    )
    for name, cli, extra in denoisers:
        if preprocessing.get(name, False):
            stages.append(
                {"name": name, "cli": cli, "input_files": current_input, **extra}
            )
            current_input = f"${{{name}.out_denoised}}"

    if preprocessing.get("gibbs", False):
        stages.append(
            {"name": "gibbs", "cli": "dipy_gibbs_ringing", "input_files": current_input}
        )
        current_input = "${gibbs.out_unring}"

    if preprocessing.get("motion_correction", False):
        stages.append(
            {
                "name": "motion",
                "cli": "dipy_correct_motion",
                "input_files": current_input,
                "bvalues_files": "${io.bvals}",
                "bvectors_files": current_bvecs,
            }
        )
        current_input = "${motion.out_moved}"
        current_bvecs = "${motion.out_bvecs}"

    if preprocessing.get("bias_correction", False):
        stages.append(
            {
                "name": "bias",
                "cli": "dipy_correct_biasfield",
                "input_files": current_input,
                "bval": "${io.bvals}",
                "bvec": current_bvecs,
            }
        )
        current_input = "${bias.out_corrected}"

    if preprocessing.get("reslice", False):
        stages.append(
            {"name": "reslice", "cli": "dipy_reslice", "input_files": current_input}
        )
        current_input = "${reslice.out_resliced}"

    mask_output = None
    if preprocessing.get("mask", False):
        stages.append(
            {
                "name": "mask",
                "cli": "dipy_brain_mask",
                "input_files": current_input,
                "bvalues_files": "${io.bvals}",
                "method": brain_extraction_method,
            }
        )
        mask_output = "${mask.out_mask}"

    out_fa = None
    for method in recon_methods:
        if method in _MASKED_FITS:
            label, cli = _MASKED_FITS[method]
            _require_mask(method=label, mask_output=mask_output)
            stages.append(
                {
                    "name": f"{method}_fit",
                    "cli": cli,
                    "input_files": current_input,
                    "bvalues_files": "${io.bvals}",
                    "bvectors_files": current_bvecs,
                    "mask_files": mask_output,
                }
            )
            if method == "dti":
                out_fa = "${dti_fit.out_fa}"

        elif method == "mapmri":
            if mapmri_timing is None:
                raise ValueError(
                    "MAPMRI fitting requires the small and big delta of the "
                    "acquisition."
                )
            small_delta, big_delta = mapmri_timing
            stages.append(
                {
                    "name": "mapmri_fit",
                    "cli": "dipy_fit_mapmri",
                    "data_files": current_input,
                    "bvals_files": "${io.bvals}",
                    "bvecs_files": current_bvecs,
                    "small_delta": small_delta,
                    "big_delta": big_delta,
                }
            )

    if tracking_methods and out_fa is None:
        raise ValueError(
            "Fiber tracking stops on the DTI FA map. Please include DTI in the "
            "reconstruction methods."
        )

    for method in tracking_methods:
        stages.append(
            {
                "name": f"track_{method}",
                "cli": "dipy_track",
                "pam_files": f"${{{method}_fit.out_pam}}",
                "stopping_files": out_fa,
                "seeding_files": mask_output,
                "tracking_method": "eudx",
            }
        )

    return config


def build_interactive_pipeline_config(*, data_chars):
    """Build pipeline configuration interactively.

    Parameters
    ----------
    data_chars : DataCharacteristics
        Data characteristics for compatibility checking.

    Returns
    -------
    dict
        Pipeline configuration dictionary with [[pipeline]] sections.
    """
    print("\n" + "=" * 60)
    print("Interactive Pipeline Builder")
    print("=" * 60)
    print(suggest_pipeline(data_chars=data_chars))

    preprocessing = interactive_preprocessing_selection()

    print("\n" + "=" * 60)
    print("Reconstruction Methods Selection")
    print("=" * 60)
    recon_methods = interactive_method_selection(data_chars=data_chars)
    mapmri_timing = interactive_mapmri_timing() if "mapmri" in recon_methods else None

    print("\n" + "=" * 60)
    print("Fiber Tracking")
    print("=" * 60)
    do_tracking = input("Include fiber tracking? [Y/n]: ").strip().lower()
    include_tracking = do_tracking != "n"

    brain_extraction_method = (
        interactive_brain_extraction_method()
        if preprocessing.get("mask", False)
        else None
    )

    tracking_methods = []
    if include_tracking and recon_methods:
        tracking_methods = interactive_tracking_selection(
            available_methods=recon_methods
        )

    return pipeline_config_from_choices(
        preprocessing=preprocessing,
        recon_methods=recon_methods,
        brain_extraction_method=brain_extraction_method,
        tracking_methods=tracking_methods,
        mapmri_timing=mapmri_timing,
    )


def pipeline_menu(*, data_chars, log_level):
    """List the predefined pipelines offered to the user.

    Parameters
    ----------
    data_chars : DataCharacteristics
        Data characteristics holding the recommended pipeline.
    log_level : int
        Logging level, which limits the menu outside of DEBUG, see
        :func:`dipy.workflows.templates.list_predefined_pipelines`.

    Returns
    -------
    list of str
        Pipeline names, always including the recommended one.
    """
    pipelines = list_predefined_pipelines(log_level=log_level)
    recommended = data_chars.recommended_pipeline
    if recommended in PREDEFINED_PIPELINES and recommended not in pipelines:
        pipelines = [recommended, *pipelines]
    return pipelines


def interactive_pipeline_selection(*, data_chars):
    """Interactive prompt for selecting a predefined pipeline.

    Parameters
    ----------
    data_chars : DataCharacteristics
        Data characteristics for showing recommendation.

    Returns
    -------
    str or None
        Selected pipeline name, or None for custom interactive mode.
    """
    print("\n" + "=" * 60)
    print("Pipeline Selection")
    print("=" * 60)

    print(suggest_pipeline(data_chars=data_chars))

    pipelines = pipeline_menu(
        data_chars=data_chars, log_level=logger.getEffectiveLevel()
    )

    while True:
        print("\nAvailable predefined pipelines:")
        print("  0. Custom (interactive) - Build your own pipeline step-by-step")
        for i, name in enumerate(pipelines, 1):
            desc = PREDEFINED_PIPELINES[name]["description"]
            print(f"  {i}. {name:<15} - {desc}")

        choice = input(
            f"\nEnter choice [0-{len(pipelines)}] (or press Enter for 'full'): "
        ).strip()

        # Default to 'full' if no input
        if not choice:
            return "full"

        # Interactive mode
        if choice == "0":
            return None

        # Try to parse as number
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(pipelines):
                return pipelines[idx]
            else:
                print(
                    f"Invalid choice. Please enter a number "
                    f"between 0 and {len(pipelines)}."
                )
        except ValueError:
            print(
                f"Invalid input. Please enter a number between 0 and {len(pipelines)}."
            )
