"""Predefined semantic pipeline templates for dipy_auto workflow.

This module provides TOML templates using the new semantic pipeline
specification with [[pipeline]] sections and automatic DAG-based wiring.
"""

# =============================================================================
# Denoise Pipeline - Simple denoising example
# =============================================================================

DENOISE_PIPELINE = """
[General]
name = "denoise_pipeline"
description = "Simple denoising pipeline with masking and info"
version = "1.0.0"
author = "DIPY Developers"

[io]
dwi = ""
bvals = ""
bvecs = ""
t1w = ""
bids_folder = ""
out_dir = "."

[[pipeline]]
name = "denoise"
cli = "dipy_denoise_nlmeans"
input_files = "${io.dwi}"

[[pipeline]]
name = "mask"
cli = "dipy_mask"
input_files = "${denoise.out_denoised}"
lb = 15

[[pipeline]]
name = "info"
cli = "dipy_info"
input_files = "${mask.out_mask}"
"""

# =============================================================================
# Basic Pipeline - Minimal preprocessing
# =============================================================================

BASIC_PIPELINE = """
[General]
name = "basic_pipeline"
description = "Basic preprocessing: denoise + brain extraction + info"
version = "1.0.0"
author = "DIPY Developers"

[io]
dwi = ""
bvals = ""
bvecs = ""
t1w = ""
bids_folder = ""
out_dir = "."

[[pipeline]]
name = "denoise"
cli = "dipy_denoise_nlmeans"
input_files = "${io.dwi}"

[[pipeline]]
name = "brain_extraction"
cli = "dipy_median_otsu"
input_files = "${denoise.out_denoised}"
save_masked = true
vol_idx = "0, 1"

[[pipeline]]
name = "mask"
cli = "dipy_mask"
input_files = "${denoise.out_denoised}"
lb = 15

[[pipeline]]
name = "info"
cli = "dipy_info"
input_files = "${mask.out_mask}"
"""

# =============================================================================
# Preprocessing Pipeline - Complete preprocessing chain
# =============================================================================

PREPROCESSING_PIPELINE = """
[General]
name = "preprocessing_pipeline"
description = "Full preprocessing: b0 → brain → Gibbs → motion → denoise"
version = "1.0.0"
author = "DIPY Developers"

[io]
dwi = ""
bvals = ""
bvecs = ""
t1w = ""
bids_folder = ""
out_dir = "."

[[pipeline]]
name = "b0_extraction"
cli = "dipy_extract_b0"
input_files = "${io.dwi}"
bvalues_files = "${io.bvals}"
b0_threshold = 50

[[pipeline]]
name = "brain_mask"
cli = "dipy_median_otsu"
input_files = "${b0_extraction.out_b0}"
median_radius = 2
numpass = 5

[[pipeline]]
name = "gibbs"
cli = "dipy_gibbs_ringing"
input_files = "${io.dwi}"
slice_axis = 2
n_points = 3

[[pipeline]]
name = "motion_correction"
cli = "dipy_correct_motion"
input_files = "${gibbs.out_unring}"
bvalues_files = "${io.bvals}"
bvectors_files = "${io.bvecs}"

[[pipeline]]
name = "denoise"
cli = "dipy_denoise_mppca"
input_files = "${motion_correction.out_moved}"
"""

# =============================================================================
# DTI Pipeline - DTI reconstruction
# =============================================================================

DTI_ONLY_PIPELINE = """
[General]
name = "dti_pipeline"
description = "Preprocessing + DTI reconstruction with tensor metrics"
version = "1.0.0"
author = "DIPY Developers"

[io]
dwi = ""
bvals = ""
bvecs = ""
t1w = ""
bids_folder = ""
out_dir = "."

[[pipeline]]
name = "denoise"
cli = "dipy_denoise_nlmeans"
input_files = "${io.dwi}"

[[pipeline]]
name = "brain_mask"
cli = "dipy_median_otsu"
input_files = "${denoise.out_denoised}"
median_radius = 2
numpass = 5

[[pipeline]]
name = "dti_fit"
cli = "dipy_fit_dti"
input_files = "${denoise.out_denoised}"
bvalues_files = "${io.bvals}"
bvectors_files = "${io.bvecs}"
mask_files = "${brain_mask.out_mask}"
fit_method = "WLS"
b0_threshold = 50
"""

# =============================================================================
# Multi-shell Pipeline - Multiple reconstruction methods
# =============================================================================

MULTI_SHELL_PIPELINE = """
[General]
name = "multi_shell_pipeline"
description = "Preprocessing + multi-method (DTI, DKI, CSD, CSA, GQI, MAPMRI)"
version = "1.0.0"
author = "DIPY Developers"

[io]
dwi = ""
bvals = ""
bvecs = ""
t1w = ""
bids_folder = ""
out_dir = "."

# Preprocessing stages
[[pipeline]]
name = "denoise"
cli = "dipy_denoise_mppca"
input_files = "${io.dwi}"

[[pipeline]]
name = "brain_mask"
cli = "dipy_median_otsu"
input_files = "${denoise.out_denoised}"
median_radius = 2
numpass = 5

# Reconstruction stages - run in parallel (no dependencies between them)
[[pipeline]]
name = "dti_fit"
cli = "dipy_fit_dti"
input_files = "${denoise.out_denoised}"
bvalues_files = "${io.bvals}"
bvectors_files = "${io.bvecs}"
mask_files = "${brain_mask.out_mask}"
fit_method = "WLS"
b0_threshold = 50

[[pipeline]]
name = "dki_fit"
cli = "dipy_fit_dki"
input_files = "${denoise.out_denoised}"
bvalues_files = "${io.bvals}"
bvectors_files = "${io.bvecs}"
mask_files = "${brain_mask.out_mask}"
b0_threshold = 50

[[pipeline]]
name = "csd_fit"
cli = "dipy_fit_csd"
input_files = "${denoise.out_denoised}"
bvalues_files = "${io.bvals}"
bvectors_files = "${io.bvecs}"
mask_files = "${brain_mask.out_mask}"
b0_threshold = 50

[[pipeline]]
name = "csa_fit"
cli = "dipy_fit_csa"
input_files = "${denoise.out_denoised}"
bvalues_files = "${io.bvals}"
bvectors_files = "${io.bvecs}"
mask_files = "${brain_mask.out_mask}"
sh_order = 8

[[pipeline]]
name = "gqi_fit"
cli = "dipy_fit_gqi"
input_files = "${denoise.out_denoised}"
bvalues_files = "${io.bvals}"
bvectors_files = "${io.bvecs}"
mask_files = "${brain_mask.out_mask}"
"""

# =============================================================================
# Tractography Pipeline - Preprocessing + reconstruction + tracking
# =============================================================================

TRACTOGRAPHY_PIPELINE = """
[General]
name = "tractography_pipeline"
description = "Preprocessing + reconstruction + fiber tracking"
version = "1.0.0"
author = "DIPY Developers"

[io]
dwi = ""
bvals = ""
bvecs = ""
t1w = ""
bids_folder = ""
out_dir = "."

# Preprocessing
[[pipeline]]
name = "denoise"
cli = "dipy_denoise_mppca"
input_files = "${io.dwi}"

[[pipeline]]
name = "brain_mask"
cli = "dipy_median_otsu"
input_files = "${denoise.out_denoised}"
median_radius = 2
numpass = 5

# DTI reconstruction for FA-based stopping criterion
[[pipeline]]
name = "dti_fit"
cli = "dipy_fit_dti"
input_files = "${denoise.out_denoised}"
bvalues_files = "${io.bvals}"
bvectors_files = "${io.bvecs}"
mask_files = "${brain_mask.out_mask}"
fit_method = "WLS"
b0_threshold = 50
extract_pam_values = true

# CSD reconstruction for fiber orientation
[[pipeline]]
name = "csd_fit"
cli = "dipy_fit_csd"
input_files = "${denoise.out_denoised}"
bvalues_files = "${io.bvals}"
bvectors_files = "${io.bvecs}"
mask_files = "${brain_mask.out_mask}"
b0_threshold = 50
extract_pam_values = true

# Fiber tracking using CSD peaks
[[pipeline]]
name = "tracking"
cli = "dipy_track"
pam_files = "${csd_fit.out_pam}"
stopping_files = "${dti_fit.out_fa}"
seeding_files = "${brain_mask.out_mask}"
seed_density = 2
step_size = 0.5
"""

# =============================================================================
# Full Pipeline - Complete analysis
# =============================================================================

FULL_PIPELINE = """
[General]
name = "full_pipeline"
description = "Full pipeline: preprocessing + reconstruction + tracking + SLR + bundles"
version = "1.0.0"
author = "DIPY Developers"

[io]
dwi = ""
bvals = ""
bvecs = ""
t1w = ""
bids_folder = ""
out_dir = "."
atlas_tractogram = ""
bundle_atlas_dir = ""

# Full preprocessing chain
[[pipeline]]
name = "reslice"
cli = "dipy_reslice"
input_files = "${io.dwi}"

[[pipeline]]
name = "b0_extraction"
cli = "dipy_extract_b0"
input_files = "${reslice.out_resliced}"
bvalues_files = "${io.bvals}"
b0_threshold = 50

[[pipeline]]
name = "brain_mask"
cli = "dipy_median_otsu"
input_files = "${reslice.out_resliced}"
bvalues_files = ["${io.bvals}"]
median_radius = 2
numpass = 5
save_masked = true

[[pipeline]]
name = "gibbs"
cli = "dipy_gibbs_ringing"
input_files = "${brain_mask.out_masked}"
slice_axis = 2
num_processes = -1

# Step 3: Motion correction
[[pipeline]]
name = "motion_correction"
cli = "dipy_correct_motion"
input_files = "${gibbs.out_unring}"
bvalues_files = "${io.bvals}"
bvectors_files = "${io.bvecs}"

# Step 4: Bias field correction (using median_otsu on b0)
[[pipeline]]
name = "bias_correction"
cli = "dipy_correct_biasfield"
input_files = "${motion_correction.out_moved}"
bval = "${io.bvals}"
bvec = "${io.bvecs}"
method = "b0"

# Step 5: Denoising with Patch2Self
[[pipeline]]
name = "denoise"
cli = "dipy_denoise_patch2self"
input_files = "${bias_correction.out_corrected}"
bval_files = "${io.bvals}"
verbose = true

# Multiple reconstructions
[[pipeline]]
name = "dti_fit"
cli = "dipy_fit_dti"
input_files = "${denoise.out_denoised}"
bvalues_files = "${io.bvals}"
bvectors_files = "${io.bvecs}"
mask_files = "${brain_mask.out_mask}"
fit_method = "WLS"
extract_pam_values = true

[[pipeline]]
name = "csd_fit"
cli = "dipy_fit_csd"
input_files = "${denoise.out_denoised}"
bvalues_files = "${io.bvals}"
bvectors_files = "${io.bvecs}"
mask_files = "${brain_mask.out_mask}"
extract_pam_values = true


# Tractography
[[pipeline]]
name = "tracking"
cli = "dipy_track"
pam_files = "${csd_fit.out_pam}"
stopping_files = "${dti_fit.out_fa}"
seeding_files = "${brain_mask.out_mask}"
seed_density = 2

# Registration (SLR)
[[pipeline]]
name = "register"
cli = "dipy_slr"
moving_files = "${tracking.out_tractogram}"
static_files = "${io.atlas_tractogram}"
bbox_valid_check = false

# Bundle segmentation (RecoBundles)
[[pipeline]]
name = "segment_bundles"
cli = "dipy_recobundles"
streamline_files = "${register.out_moved}"
model_bundle_files = "${io.bundle_atlas_dir}/*.trk"
mix_names = true
"""

# =============================================================================
# Comprehensive Pipeline - Complete end-to-end analysis
# =============================================================================

COMPREHENSIVE_PIPELINE = """
[General]
name = "comprehensive_pipeline"
description = "Complete end-to-end with all preprocessing and 6 methods"
version = "1.0.0"
author = "DIPY Developers"

[io]
dwi = ""
bvals = ""
bvecs = ""
t1w = ""
bids_folder = ""
out_dir = "."
atlas_tractogram = ""
bundle_atlas_dir = ""

# Step 0: B0 extraction
[[pipeline]]
name = "b0_extraction"
cli = "dipy_extract_b0"
input_files = "${io.dwi}"
bvalues_files = "${io.bvals}"
b0_threshold = 50

# Step 1: Brain extraction
[[pipeline]]
name = "brain_extraction"
cli = "dipy_median_otsu"
input_files = "${b0_extraction.out_b0}"
median_radius = 2
numpass = 5
save_masked = true

# Apply mask to full DWI
[[pipeline]]
name = "mask_dwi"
cli = "dipy_mask"
input_files = "${io.dwi}"
mask_files = "${brain_extraction.out_mask}"

# Step 2: Gibbs ringing correction
[[pipeline]]
name = "gibbs"
cli = "dipy_gibbs_ringing"
input_files = "${mask_dwi.out_masked}"
slice_axis = 2
n_points = 3

# Step 3: Motion correction
[[pipeline]]
name = "motion_correction"
cli = "dipy_correct_motion"
input_files = "${gibbs.out_unring}"
bvalues_files = "${io.bvals}"
bvectors_files = "${io.bvecs}"

# Step 4: Bias field correction (using median_otsu on b0)
[[pipeline]]
name = "bias_correction"
cli = "dipy_median_otsu"
input_files = "${motion_correction.out_moved}"

# Step 5: Denoising with Patch2Self
[[pipeline]]
name = "denoise"
cli = "dipy_denoise_patch2self"
input_files = "${bias_correction.out_corrected}"
bvalues_files = "${io.bvals}"

# Step 6: S0 replacement using DTI prediction
[[pipeline]]
name = "s0_replacement"
cli = "dipy_fit_dti"
input_files = "${denoise.out_denoised}"
bvalues_files = "${io.bvals}"
bvectors_files = "${io.bvecs}"
mask_files = "${brain_extraction.out_mask}"
b0_threshold = 50
predict_s0 = true

# Step 7: Multi-method reconstruction
# DTI
[[pipeline]]
name = "dti_fit"
cli = "dipy_fit_dti"
input_files = "${s0_replacement.out_s0}"
bvalues_files = "${io.bvals}"
bvectors_files = "${io.bvecs}"
mask_files = "${brain_extraction.out_mask}"
fit_method = "WLS"
b0_threshold = 50
extract_pam_values = true

# DKI
[[pipeline]]
name = "dki_fit"
cli = "dipy_fit_dki"
input_files = "${s0_replacement.out_s0}"
bvalues_files = "${io.bvals}"
bvectors_files = "${io.bvecs}"
mask_files = "${brain_extraction.out_mask}"
b0_threshold = 50
extract_pam_values = true

# CSD
[[pipeline]]
name = "csd_fit"
cli = "dipy_fit_csd"
input_files = "${s0_replacement.out_s0}"
bvalues_files = "${io.bvals}"
bvectors_files = "${io.bvecs}"
mask_files = "${brain_extraction.out_mask}"
b0_threshold = 50
extract_pam_values = true

# CSA
[[pipeline]]
name = "csa_fit"
cli = "dipy_fit_csa"
input_files = "${s0_replacement.out_s0}"
bvalues_files = "${io.bvals}"
bvectors_files = "${io.bvecs}"
mask_files = "${brain_extraction.out_mask}"
sh_order = 8
extract_pam_values = true

# GQI
[[pipeline]]
name = "gqi_fit"
cli = "dipy_fit_gqi"
input_files = "${s0_replacement.out_s0}"
bvalues_files = "${io.bvals}"
bvectors_files = "${io.bvecs}"
mask_files = "${brain_extraction.out_mask}"
extract_pam_values = true

# MAPMRI
[[pipeline]]
name = "mapmri_fit"
cli = "dipy_fit_mapmri"
input_files = "${s0_replacement.out_s0}"
bvalues_files = "${io.bvals}"
bvectors_files = "${io.bvecs}"
mask_files = "${brain_extraction.out_mask}"
extract_pam_values = true

# Step 8: Multi-method fiber tracking
[[pipeline]]
name = "track_dti"
cli = "dipy_track"
pam_files = "${dti_fit.out_pam}"
stopping_files = "${dti_fit.out_fa}"
seeding_files = "${brain_extraction.out_mask}"
seed_density = 2
step_size = 0.5

[[pipeline]]
name = "track_dki"
cli = "dipy_track"
pam_files = "${dki_fit.out_pam}"
stopping_files = "${dti_fit.out_fa}"
seeding_files = "${brain_extraction.out_mask}"
seed_density = 2
step_size = 0.5

[[pipeline]]
name = "track_csd"
cli = "dipy_track"
pam_files = "${csd_fit.out_pam}"
stopping_files = "${dti_fit.out_fa}"
seeding_files = "${brain_extraction.out_mask}"
seed_density = 2
step_size = 0.5

[[pipeline]]
name = "track_csa"
cli = "dipy_track"
pam_files = "${csa_fit.out_pam}"
stopping_files = "${dti_fit.out_fa}"
seeding_files = "${brain_extraction.out_mask}"
seed_density = 2
step_size = 0.5

[[pipeline]]
name = "track_gqi"
cli = "dipy_track"
pam_files = "${gqi_fit.out_pam}"
stopping_files = "${dti_fit.out_fa}"
seeding_files = "${brain_extraction.out_mask}"
seed_density = 2
step_size = 0.5

[[pipeline]]
name = "track_mapmri"
cli = "dipy_track"
pam_files = "${mapmri_fit.out_pam}"
stopping_files = "${dti_fit.out_fa}"
seeding_files = "${brain_extraction.out_mask}"
seed_density = 2
step_size = 0.5

# Step 9: Multi-method registration (SLR to atlas)
[[pipeline]]
name = "register_dti"
cli = "dipy_slr"
moving_files = "${track_dti.out_tractogram}"
static_files = "${io.atlas_tractogram}"

[[pipeline]]
name = "register_dki"
cli = "dipy_slr"
moving_files = "${track_dki.out_tractogram}"
static_files = "${io.atlas_tractogram}"

[[pipeline]]
name = "register_csd"
cli = "dipy_slr"
moving_files = "${track_csd.out_tractogram}"
static_files = "${io.atlas_tractogram}"

[[pipeline]]
name = "register_csa"
cli = "dipy_slr"
moving_files = "${track_csa.out_tractogram}"
static_files = "${io.atlas_tractogram}"

[[pipeline]]
name = "register_gqi"
cli = "dipy_slr"
moving_files = "${track_gqi.out_tractogram}"
static_files = "${io.atlas_tractogram}"

[[pipeline]]
name = "register_mapmri"
cli = "dipy_slr"
moving_files = "${track_mapmri.out_tractogram}"
static_files = "${io.atlas_tractogram}"

# Step 10: Multi-method segmentation (Recobundles)
[[pipeline]]
name = "segment_dti"
cli = "dipy_recobundles"
tractogram_files = "${register_dti.out_moved}"
atlas_dir = "${io.bundle_atlas_dir}"

[[pipeline]]
name = "segment_dki"
cli = "dipy_recobundles"
tractogram_files = "${register_dki.out_moved}"
atlas_dir = "${io.bundle_atlas_dir}"

[[pipeline]]
name = "segment_csd"
cli = "dipy_recobundles"
tractogram_files = "${register_csd.out_moved}"
atlas_dir = "${io.bundle_atlas_dir}"

[[pipeline]]
name = "segment_csa"
cli = "dipy_recobundles"
tractogram_files = "${register_csa.out_moved}"
atlas_dir = "${io.bundle_atlas_dir}"

[[pipeline]]
name = "segment_gqi"
cli = "dipy_recobundles"
tractogram_files = "${register_gqi.out_moved}"
atlas_dir = "${io.bundle_atlas_dir}"

[[pipeline]]
name = "segment_mapmri"
cli = "dipy_recobundles"
tractogram_files = "${register_mapmri.out_moved}"
atlas_dir = "${io.bundle_atlas_dir}"
"""


# =============================================================================
# Pipeline Dictionary and Helper Functions
# =============================================================================

PREDEFINED_PIPELINES = {
    "denoise": {
        "description": "Simple denoising: nlmeans + mask + info",
        "config": DENOISE_PIPELINE,
    },
    "basic": {
        "description": "Basic preprocessing: denoise + brain extraction + mask + info",
        "config": BASIC_PIPELINE,
    },
    "preprocessing": {
        "description": "Full preprocessing: b0 → brain → Gibbs → motion → denoise",
        "config": PREPROCESSING_PIPELINE,
    },
    "dti_only": {
        "description": "Preprocessing + DTI reconstruction with tensor metrics",
        "config": DTI_ONLY_PIPELINE,
    },
    "multi_shell": {
        "description": "Multi-method: DTI, DKI, CSD, CSA, GQI (parallel)",
        "config": MULTI_SHELL_PIPELINE,
    },
    "tractography": {
        "description": "Preprocessing + DTI + CSD + fiber tracking",
        "config": TRACTOGRAPHY_PIPELINE,
    },
    "full": {
        "description": (
            "Full pipeline: preprocessing + reconstruction + tracking + "
            "SLR + bundles"
        ),
        "config": FULL_PIPELINE,
    },
    "comprehensive": {
        "description": "Complete: preprocessing + 6 methods + tracking + SLR + bundles",
        "config": COMPREHENSIVE_PIPELINE,
    },
}


def get_predefined_pipeline(*, pipeline_name):
    """Get predefined pipeline configuration by name.

    Parameters
    ----------
    pipeline_name : str
        Name of the predefined pipeline.

    Returns
    -------
    str
        TOML configuration string with [[pipeline]] sections.

    Raises
    ------
    KeyError
        If pipeline_name is not found.
    """
    if pipeline_name not in PREDEFINED_PIPELINES:
        available = ", ".join(list_predefined_pipelines())
        raise KeyError(
            f"Pipeline '{pipeline_name}' not found. "
            f"Available pipelines: {available}"
        )
    return PREDEFINED_PIPELINES[pipeline_name]["config"]


def get_pipeline_description(*, pipeline_name):
    """Get description for a predefined pipeline.

    Parameters
    ----------
    pipeline_name : str
        Name of the predefined pipeline.

    Returns
    -------
    str
        Description of the pipeline.

    Raises
    ------
    KeyError
        If pipeline_name is not found.
    """
    if pipeline_name not in PREDEFINED_PIPELINES:
        available = ", ".join(list_predefined_pipelines())
        raise KeyError(
            f"Pipeline '{pipeline_name}' not found. "
            f"Available pipelines: {available}"
        )
    return PREDEFINED_PIPELINES[pipeline_name]["description"]


def list_predefined_pipelines(*, log_level=None):
    """List all available predefined pipeline names.

    Parameters
    ----------
    log_level : int, optional
        Logging level. If not DEBUG (10), only returns "full" and allows
        "interactive" mode. If DEBUG or None, returns all pipelines.

    Returns
    -------
    list
        List of pipeline names.
    """
    import logging

    all_pipelines = list(PREDEFINED_PIPELINES.keys())

    # If log_level is provided and it's not DEBUG, filter to show only "full"
    # Note: "interactive" is handled separately as it's not a predefined pipeline
    if log_level is not None and log_level != logging.DEBUG:
        return ["full"]

    return all_pipelines


def list_pipelines_with_descriptions(*, log_level=None):
    """List all predefined pipelines with descriptions.

    Parameters
    ----------
    log_level : int, optional
        Logging level. If not DEBUG (10), only shows "full" pipeline.
        If DEBUG or None, shows all pipelines.

    Returns
    -------
    str
        Formatted list of pipelines with descriptions.
    """
    lines = ["Available Semantic Pipelines (using [[pipeline]] sections):", "=" * 70]
    for name in list_predefined_pipelines(log_level=log_level):
        desc = PREDEFINED_PIPELINES[name]["description"]
        lines.append(f"  {name:<15} - {desc}")
    lines.append("=" * 70)
    lines.append(
        "\nEach pipeline uses semantic stage names with automatic " "DAG-based wiring."
    )
    lines.append("Stages are executed in topological order based on dependencies.")

    return "\n".join(lines)
