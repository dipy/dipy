"""Interactive prompts for user-guided pipeline configuration."""

from dataclasses import dataclass
import os
import shutil

from dipy.utils.logging import logger

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
        Threshold for considering a b-value as b0 (default: 50).

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
        with open(bvals_file, "r") as f:
            content = f.read()
            bvals = [float(x) for x in content.split()]
    except (ValueError, IOError) as e:
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
        recommended_pipeline = "multi_shell"
        recommended_methods = ["dti", "dki", "csd", "csa", "gqi"]
    elif is_single_shell and non_zero_bvals and non_zero_bvals[0] >= 1000:
        recommended_pipeline = "tractography"
        recommended_methods = ["dti", "csd"]
    elif is_single_shell:
        recommended_pipeline = "dti_only"
        recommended_methods = ["dti"]
    else:
        recommended_pipeline = "basic"
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
        "mask": "Brain mask extraction",
    }

    print("\nSelect preprocessing steps to include:")
    for i, (_step_name, desc) in enumerate(preprocessing_steps.items(), 1):
        print(f"  {i}. {desc}")

    print(
        "\nEnter numbers (comma-separated, e.g., 4,5,6,7,8,9) or press Enter for "
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
                        "Enter choice [1-{}]: ".format(len(selected_denoise))
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

    method_list = ["dki", "csd", "csa", "gqi", "mapmri"]
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
        return selected if selected else available_methods
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
        return selected if selected else available_tractograms
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
        return selected if selected else available_registered
    except (ValueError, IndexError):
        print("Invalid input, using all methods")
        return available_registered


def check_synthstrip_available():
    """Check if SynthStrip is available on the system.

    Returns
    -------
    bool
        True if mri_synthstrip command is available.
    """
    return shutil.which("mri_synthstrip") is not None


def interactive_brain_extraction_method():
    """Interactive prompt for brain extraction method.

    Returns
    -------
    str
        Selected method ('synthstrip' or 'median_otsu').
    """
    has_synthstrip = check_synthstrip_available()

    if not has_synthstrip:
        print("\nSynthStrip not found, using median_otsu")
        return "median_otsu"

    print("\nChoose brain extraction method:")
    print("  1. SynthStrip (FreeSurfer) - More accurate, requires FreeSurfer")
    print("  2. median_otsu (DIPY) - Good accuracy, no extra dependencies")

    choice = input("Enter choice [1-2]: ").strip()

    if choice == "1":
        return "synthstrip"
    elif choice == "2":
        return "median_otsu"
    else:
        print("Invalid choice, defaulting to median_otsu")
        return "median_otsu"


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

    # 1. Select preprocessing steps
    preprocessing = interactive_preprocessing_selection()

    # 2. Select reconstruction methods
    print("\n" + "=" * 60)
    print("Reconstruction Methods Selection")
    print("=" * 60)
    recon_methods = interactive_method_selection(data_chars=data_chars)

    # 3. Ask about tracking
    print("\n" + "=" * 60)
    print("Fiber Tracking")
    print("=" * 60)
    do_tracking = input("Include fiber tracking? [Y/n]: ").strip().lower()
    include_tracking = do_tracking != "n"

    # 4. Build pipeline configuration
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

    # Build pipeline stages based on selections
    current_input = "${io.dwi}"

    # Denoising - NLMeans
    if preprocessing.get("denoise_nlmeans", False):
        config["pipeline"].append(
            {
                "name": "denoise_nlmeans",
                "cli": "dipy_denoise_nlmeans",
                "input_files": current_input,
            }
        )
        current_input = "${denoise_nlmeans.out_denoised}"

    # Denoising - MPPCA
    if preprocessing.get("denoise_mppca", False):
        config["pipeline"].append(
            {
                "name": "denoise_mppca",
                "cli": "dipy_denoise_mppca",
                "input_files": current_input,
            }
        )
        current_input = "${denoise_mppca.out_denoised}"

    # Denoising - LPCA
    if preprocessing.get("denoise_lpca", False):
        config["pipeline"].append(
            {
                "name": "denoise_lpca",
                "cli": "dipy_denoise_lpca",
                "input_files": current_input,
            }
        )
        current_input = "${denoise_lpca.out_denoised}"

    # Denoising - Patch2Self
    if preprocessing.get("denoise_patch2self", False):
        config["pipeline"].append(
            {
                "name": "denoise_patch2self",
                "cli": "dipy_denoise_patch2self",
                "input_files": current_input,
                "bvals_files": "${io.bvals}",
            }
        )
        current_input = "${denoise_patch2self.out_denoised}"

    # Reslice
    if preprocessing.get("reslice", False):
        config["pipeline"].append(
            {
                "name": "reslice",
                "cli": "dipy_reslice",
                "input_files": current_input,
            }
        )
        current_input = "${reslice.out_resliced}"

    # Gibbs ringing removal
    if preprocessing.get("gibbs", False):
        config["pipeline"].append(
            {
                "name": "gibbs",
                "cli": "dipy_gibbs_ringing",
                "input_files": current_input,
            }
        )
        current_input = "${gibbs.out_unring}"

    # Motion correction
    if preprocessing.get("motion_correction", False):
        config["pipeline"].append(
            {
                "name": "motion",
                "cli": "dipy_correct_motion",
                "input_files": current_input,
                "bvals_files": "${io.bvals}",
                "bvecs_files": "${io.bvecs}",
            }
        )
        current_input = "${motion.out_moved}"

    # Bias correction
    if preprocessing.get("bias_correction", False):
        config["pipeline"].append(
            {
                "name": "bias",
                "cli": "dipy_median_otsu",
                "input_files": current_input,
            }
        )
        current_input = "${bias.out_corrected}"

    mask_output = None
    if preprocessing.get("mask", False):
        brain_method = interactive_brain_extraction_method()
        if brain_method == "synthstrip":
            print(
                "Note: SynthStrip integration requires custom script. "
                "Using median_otsu."
            )
            brain_method = "median_otsu"

        config["pipeline"].append(
            {
                "name": "mask",
                "cli": "dipy_median_otsu",
                "input_files": current_input,
            }
        )
        mask_output = "${mask.out_mask}"

    preprocessed_dwi = current_input
    for method in recon_methods:
        if method == "dti":
            stage_config = {
                "name": "dti_fit",
                "cli": "dipy_fit_dti",
                "input_files": preprocessed_dwi,
                "bvals_files": "${io.bvals}",
                "bvecs_files": "${io.bvecs}",
            }
            if mask_output:
                stage_config["mask_files"] = mask_output
            config["pipeline"].append(stage_config)

        elif method == "dki":
            stage_config = {
                "name": "dki_fit",
                "cli": "dipy_fit_dki",
                "input_files": preprocessed_dwi,
                "bvals_files": "${io.bvals}",
                "bvecs_files": "${io.bvecs}",
            }
            if mask_output:
                stage_config["mask_files"] = mask_output
            config["pipeline"].append(stage_config)

        elif method == "csd":
            stage_config = {
                "name": "csd_fit",
                "cli": "dipy_fit_csd",
                "input_files": preprocessed_dwi,
                "bvals_files": "${io.bvals}",
                "bvecs_files": "${io.bvecs}",
            }
            if mask_output:
                stage_config["mask_files"] = mask_output
            config["pipeline"].append(stage_config)

        elif method == "csa":
            stage_config = {
                "name": "csa_fit",
                "cli": "dipy_fit_csa",
                "input_files": preprocessed_dwi,
                "bvals_files": "${io.bvals}",
                "bvecs_files": "${io.bvecs}",
            }
            if mask_output:
                stage_config["mask_files"] = mask_output
            config["pipeline"].append(stage_config)

        elif method == "gqi":
            stage_config = {
                "name": "gqi_fit",
                "cli": "dipy_fit_gqi",
                "input_files": preprocessed_dwi,
                "bvals_files": "${io.bvals}",
                "bvecs_files": "${io.bvecs}",
            }
            if mask_output:
                stage_config["mask_files"] = mask_output
            config["pipeline"].append(stage_config)

        elif method == "mapmri":
            stage_config = {
                "name": "mapmri_fit",
                "cli": "dipy_fit_mapmri",
                "input_files": preprocessed_dwi,
                "bvals_files": "${io.bvals}",
                "bvecs_files": "${io.bvecs}",
            }
            if mask_output:
                stage_config["mask_files"] = mask_output
            config["pipeline"].append(stage_config)

    if include_tracking and recon_methods:
        if "dti" in recon_methods:
            tracking_methods = interactive_tracking_selection(
                available_methods=["dti"] + [m for m in recon_methods if m != "dti"]
            )
        else:
            tracking_methods = interactive_tracking_selection(
                available_methods=recon_methods
            )

        for method in tracking_methods:
            config["pipeline"].append(
                {
                    "name": f"track_{method}",
                    "cli": "dipy_track",
                    "pam_files": f"${{{method}_fit.out_pam}}",
                }
            )

    return config


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
    from dipy.workflows.templates import (
        PREDEFINED_PIPELINES,
        list_predefined_pipelines,
    )

    print("\n" + "=" * 60)
    print("Pipeline Selection")
    print("=" * 60)

    print(suggest_pipeline(data_chars=data_chars))

    current_log_level = logger.getEffectiveLevel()
    pipelines = list_predefined_pipelines(log_level=current_log_level)

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
                f"Invalid input. Please enter a number "
                f"between 0 and {len(pipelines)}."
            )
