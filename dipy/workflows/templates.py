"""Predefined semantic pipeline templates for dipy_auto workflow.

This module provides TOML templates using the new semantic pipeline
specification with [[pipeline]] sections and automatic DAG-based wiring.
"""

import logging

# =============================================================================
# Preprocessing Pipeline - Complete preprocessing chain
# =============================================================================

PREPROCESSING_PIPELINE = """
[General]
name = "preprocessing_pipeline"
description = "Full preprocessing: denoise → Gibbs → motion → bias → reslice → mask"
version = "1.0.0"
author = "DIPY Developers"

[io]
dwi = ""
bvals = ""
bvecs = ""
t1w = ""
bids_folder = ""
dwi_opp_phase = ""
pe_dir = ""
total_readout_time = ""
effective_echo_spacing = ""
echo_time = ""
repetition_time = ""
out_dir = "."

# Same chain, in the same order, as the preprocessing of the full pipeline;
# see the comments there for the constraints that fix this order.
[[pipeline]]
name = "denoise"
cli = "dipy_denoise"
input_files = "${io.dwi}"
bvalues_files = "${io.bvals}"

[[pipeline]]
name = "gibbs"
cli = "dipy_gibbs_ringing"
input_files = "${denoise.out_denoised}"
slice_axis = 2
num_processes = -1

[[pipeline]]
name = "motion_correction"
cli = "dipy_correct_motion"
input_files = "${gibbs.out_unring}"
bvalues_files = "${io.bvals}"
bvectors_files = "${io.bvecs}"

[[pipeline]]
name = "bias_correction"
cli = "dipy_correct_biasfield"
input_files = "${motion_correction.out_moved}"
bval = "${io.bvals}"
bvec = "${motion_correction.out_bvecs}"
method = "bspline"

[[pipeline]]
name = "reslice"
cli = "dipy_reslice"
input_files = "${bias_correction.out_corrected}"
order = "lanczos2"

[[pipeline]]
name = "b0_extraction"
cli = "dipy_extract_b0"
input_files = "${reslice.out_resliced}"
bvalues_files = "${io.bvals}"
b0_threshold = 50

[[pipeline]]
name = "brain_mask"
cli = "dipy_brain_mask"
input_files = "${reslice.out_resliced}"
bvalues_files = ["${io.bvals}"]
median_radius = 2
numpass = 5
finalize_mask = true
save_masked = true
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
dwi_opp_phase = ""
pe_dir = ""
total_readout_time = ""
effective_echo_spacing = ""
echo_time = ""
repetition_time = ""
out_dir = "."

[[pipeline]]
name = "denoise"
cli = "dipy_denoise"
input_files = "${io.dwi}"
bvalues_files = "${io.bvals}"

[[pipeline]]
name = "brain_mask"
cli = "dipy_median_otsu"
input_files = "${denoise.out_denoised}"
bvalues_files = "${io.bvals}"
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
dwi_opp_phase = ""
pe_dir = ""
total_readout_time = ""
effective_echo_spacing = ""
echo_time = ""
repetition_time = ""
out_dir = "."

# Preprocessing
[[pipeline]]
name = "denoise"
cli = "dipy_denoise"
input_files = "${io.dwi}"
bvalues_files = "${io.bvals}"

[[pipeline]]
name = "brain_mask"
cli = "dipy_median_otsu"
input_files = "${denoise.out_denoised}"
bvalues_files = "${io.bvals}"
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
description = "Full pipeline: preprocessing with motion correction + T1 registration + reconstruction + tracking + SLR + bundles"
version = "1.0.0"
author = "DIPY Developers"

[io]
dwi = ""
bvals = ""
bvecs = ""
t1w = ""
bids_folder = ""
dwi_opp_phase = ""
pe_dir = ""
total_readout_time = ""
effective_echo_spacing = ""
echo_time = ""
repetition_time = ""
out_dir = "."
atlas_tractogram = ""
bundle_atlas_dir = ""

# Full preprocessing chain.
#
# Stage order is constrained, in decreasing strictness:
#   1. Denoising and Gibbs unringing both exploit properties of the data as it
#      came off the scanner, so every stage that interpolates (reslice, motion
#      and eddy correction, registration) has to run after them.
#   2. Denoising comes first. Gibbs unringing selects its subvoxel shift by
#      minimising local total variation, and noise corrupts that choice. The
#      ordering is a hard requirement for MP-PCA, whose noise-level estimate
#      assumes spatially white noise that unringing would correlate.
#   3. Gibbs unringing is FFT-based and needs the full field of view. Handing
#      it a skull-stripped volume puts a hard edge at the mask boundary, which
#      is itself a source of ringing.
#   4. The brain mask is thresholded on intensity, so it is computed last: on
#      denoised, unrung, bias-corrected data, and on the final grid.
#   5. Motion correction rotates the b-vectors; every later stage reads the
#      rotated ones.
[[pipeline]]
name = "denoise"
cli = "dipy_denoise"
input_files = "${io.dwi}"
bvalues_files = "${io.bvals}"

[[pipeline]]
name = "gibbs"
cli = "dipy_gibbs_ringing"
input_files = "${denoise.out_denoised}"
slice_axis = 2
num_processes = -1

[[pipeline]]
name = "motion_correction"
cli = "dipy_correct_motion"
input_files = "${gibbs.out_unring}"
bvalues_files = "${io.bvals}"
bvectors_files = "${io.bvecs}"

[[pipeline]]
name = "bias_correction"
cli = "dipy_correct_biasfield"
input_files = "${motion_correction.out_moved}"
bval = "${io.bvals}"
bvec = "${motion_correction.out_bvecs}"
method = "bspline"

# Resampling. Everything below this point works on the resliced grid.
[[pipeline]]
name = "reslice"
cli = "dipy_reslice"
input_files = "${bias_correction.out_corrected}"
order = "lanczos2"

[[pipeline]]
name = "b0_extraction"
cli = "dipy_extract_b0"
input_files = "${reslice.out_resliced}"
bvalues_files = "${io.bvals}"
b0_threshold = 50

[[pipeline]]
name = "brain_mask"
cli = "dipy_brain_mask"
input_files = "${reslice.out_resliced}"
bvalues_files = ["${io.bvals}"]
median_radius = 2
numpass = 5
finalize_mask = true
save_masked = true

# T1 to diffusion space, skipped when no T1 image is given.
[[pipeline]]
name = "t1_registration"
cli = "dipy_align_affine"
static_image_files = "${b0_extraction.out_b0}"
moving_image_files = "${io.t1w}"
transform = "rigid"
out_moved = "t1w_dwispace.nii.gz"
out_affine = "t1w_to_dwi_affine.txt"
optional = true

# Multiple reconstructions
[[pipeline]]
name = "dti_fit"
cli = "dipy_fit_dti"
input_files = "${reslice.out_resliced}"
bvalues_files = "${io.bvals}"
bvectors_files = "${motion_correction.out_bvecs}"
mask_files = "${brain_mask.out_mask}"
fit_method = "WLS"
extract_pam_values = true
out_dir = "${io.out_dir}/dti"

[[pipeline]]
name = "force_fit"
cli = "dipy_fit_force"
input_files = "${reslice.out_resliced}"
bvalues_files = "${io.bvals}"
bvectors_files = "${motion_correction.out_bvecs}"
mask_files = "${brain_mask.out_mask}"
compute_kurtosis = true
engine = "ray"
out_dir = "${io.out_dir}/force"

# Tractography
[[pipeline]]
name = "tracking"
cli = "dipy_track"
pam_files = "${force_fit.out_pam}"
stopping_files = "${force_fit.out_fa}"
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
out_dir = "${io.out_dir}/rec_bundles"

# Label bundles
[[pipeline]]
name = "label_bundles"
cli = "dipy_labelsbundles"
streamline_files = "${tracking.out_tractogram}"
labels_files = "${segment_bundles.out_dir}/*_labels.npy"
mix_names = true
out_dir = "${io.out_dir}/org_bundles"

# Buan profiles
[[pipeline]]
name = "buan_profiles"
cli = "dipy_buan_profiles"
model_bundle_folder = "${io.bundle_atlas_dir}"
subject_folder = "${io.out_dir}"
metric_folder= "${dti_fit.out_dir}"
orig_bundle_folder= "${label_bundles.out_dir}"
out_dir = "${io.out_dir}/buan_profiles"
"""

# =============================================================================
# Pipeline Dictionary and Helper Functions
# =============================================================================

PREDEFINED_PIPELINES = {
    "preprocessing": {
        "description": "Full preprocessing: denoise → Gibbs → motion → bias → reslice → mask",
        "config": PREPROCESSING_PIPELINE,
    },
    "dti_only": {
        "description": "Preprocessing + DTI reconstruction with tensor metrics",
        "config": DTI_ONLY_PIPELINE,
    },
    "tractography": {
        "description": "Preprocessing + DTI + CSD + fiber tracking",
        "config": TRACTOGRAPHY_PIPELINE,
    },
    "full": {
        "description": (
            "Full pipeline: preprocessing with motion correction + T1 registration"
            " + reconstruction + tracking + SLR + bundles"
        ),
        "config": FULL_PIPELINE,
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
            f"Pipeline '{pipeline_name}' not found. Available pipelines: {available}"
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
            f"Pipeline '{pipeline_name}' not found. Available pipelines: {available}"
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
        "\nEach pipeline uses semantic stage names with automatic DAG-based wiring."
    )
    lines.append("Stages are executed in topological order based on dependencies.")

    return "\n".join(lines)
