#!python
import logging
import os
import sys

from dipy.utils.optpkg import optional_package
from dipy.workflows.flow_runner import run_flow

cli_flows = {
    "dipy_align_affine": ("dipy.workflows.align", "ImageRegistrationFlow"),
    "dipy_align_syn": ("dipy.workflows.align", "SynRegistrationFlow"),
    "dipy_apply_transform": ("dipy.workflows.align", "ApplyTransformFlow"),
    "dipy_buan_lmm": ("dipy.workflows.stats", "LinearMixedModelsFlow"),
    "dipy_buan_shapes": ("dipy.workflows.stats", "BundleShapeAnalysis"),
    "dipy_buan_profiles": ("dipy.workflows.stats", "BundleAnalysisTractometryFlow"),
    "dipy_bundlewarp": ("dipy.workflows.align", "BundleWarpFlow"),
    "dipy_classify_tissue": ("dipy.workflows.segment", "ClassifyTissueFlow"),
    "dipy_correct_motion": ("dipy.workflows.align", "MotionCorrectionFlow"),
    "dipy_correct_biasfield": ("dipy.workflows.nn", "BiasFieldCorrectionFlow"),
    "dipy_concatenate_tractograms": ("dipy.workflows.io", "ConcatenateTractogramFlow"),
    "dipy_convert_tractogram": ("dipy.workflows.io", "ConvertTractogramFlow"),
    "dipy_convert_tensors": ("dipy.workflows.io", "ConvertTensorsFlow"),
    "dipy_convert_sh": ("dipy.workflows.io", "ConvertSHFlow"),
    "dipy_denoise_nlmeans": ("dipy.workflows.denoise", "NLMeansFlow"),
    "dipy_denoise_lpca": ("dipy.workflows.denoise", "LPCAFlow"),
    "dipy_denoise_mppca": ("dipy.workflows.denoise", "MPPCAFlow"),
    "dipy_denoise_patch2self": ("dipy.workflows.denoise", "Patch2SelfFlow"),
    "dipy_evac_plus": ("dipy.workflows.nn", "EVACPlusFlow"),
    "dipy_fetch": ("dipy.workflows.io", "FetchFlow"),
    "dipy_fit_csa": ("dipy.workflows.reconst", "ReconstQBallBaseFlow"),
    "dipy_fit_csd": ("dipy.workflows.reconst", "ReconstCSDFlow"),
    "dipy_fit_dki": ("dipy.workflows.reconst", "ReconstDkiFlow"),
    "dipy_fit_dti": ("dipy.workflows.reconst", "ReconstDtiFlow"),
    "dipy_fit_dsi": ("dipy.workflows.reconst", "ReconstDsiFlow"),
    "dipy_fit_dsid": ("dipy.workflows.reconst", "ReconstDsiFlow"),
    "dipy_fit_forecast": ("dipy.workflows.reconst", "ReconstForecastFlow"),
    "dipy_fit_gqi": ("dipy.workflows.reconst", "ReconstGQIFlow"),
    "dipy_fit_ivim": ("dipy.workflows.reconst", "ReconstIvimFlow"),
    "dipy_fit_mapmri": ("dipy.workflows.reconst", "ReconstMAPMRIFlow"),
    "dipy_fit_opdt": ("dipy.workflows.reconst", "ReconstQBallBaseFlow"),
    "dipy_fit_qball": ("dipy.workflows.reconst", "ReconstQBallBaseFlow"),
    "dipy_fit_sdt": ("dipy.workflows.reconst", "ReconstSDTFlow"),
    "dipy_fit_sfm": ("dipy.workflows.reconst", "ReconstSFMFlow"),
    "dipy_gibbs_ringing": ("dipy.workflows.denoise", "GibbsRingingFlow"),
    "dipy_horizon": ("dipy.workflows.viz", "HorizonFlow"),
    "dipy_info": ("dipy.workflows.io", "IoInfoFlow"),
    "dipy_extract_b0": ("dipy.workflows.io", "ExtractB0Flow"),
    "dipy_extract_shell": ("dipy.workflows.io", "ExtractShellFlow"),
    "dipy_extract_volume": ("dipy.workflows.io", "ExtractVolumeFlow"),
    "dipy_labelsbundles": ("dipy.workflows.segment", "LabelsBundlesFlow"),
    "dipy_math": ("dipy.workflows.io", "MathFlow"),
    "dipy_mask": ("dipy.workflows.mask", "MaskFlow"),
    "dipy_median_otsu": ("dipy.workflows.segment", "MedianOtsuFlow"),
    "dipy_nifti2pam": ("dipy.workflows.io", "NiftisToPamFlow"),
    "dipy_pam2nifti": ("dipy.workflows.io", "PamToNiftisFlow"),
    "dipy_recobundles": ("dipy.workflows.segment", "RecoBundlesFlow"),
    "dipy_reslice": ("dipy.workflows.align", "ResliceFlow"),
    "dipy_sh_convert_mrtrix": ("dipy.workflows.io", "ConvertSHFlow"),
    "dipy_slr": ("dipy.workflows.align", "SlrWithQbxFlow"),
    "dipy_snr_in_cc": ("dipy.workflows.stats", "SNRinCCFlow"),
    "dipy_split": ("dipy.workflows.io", "SplitFlow"),
    "dipy_tensor2pam": ("dipy.workflows.io", "TensorToPamFlow"),
    "dipy_track": ("dipy.workflows.tracking", "LocalFiberTrackingPAMFlow"),
    "dipy_track_pft": ("dipy.workflows.tracking", "PFTrackingPAMFlow"),
}


def run():
    """Run scripts located in pyproject.toml."""
    script_name = os.path.basename(sys.argv[0])
    mod_name, flow_name = cli_flows.get(script_name, (None, None))
    if mod_name is None:
        print(f"Flow: {script_name} not Found in DIPY")
        print(f"Available flows: {', '.join(cli_flows.keys())}")
        sys.exit(1)
    mod, _, _ = optional_package(mod_name)

    if script_name in ["dipy_sh_convert_mrtrix"]:
        logging.warning(
            "`dipy_sh_convert_mrtrix` CLI is deprecated since DIPY 1.11.0. It will be "
            "removed on later release. Please use the `dipy_convert_sh` CLI instead.",
        )

    extra_args = {}
    if script_name == "dipy_fit_dsid":
        extra_args = {
            "remove_convolution": {
                "dest": "remove_convolution",
                "action": "store_true",
                "default": True,
            }
        }
    elif script_name == "dipy_fit_csa":
        extra_args = {
            "method": {
                "action": "store",
                "dest": "method",
                "metavar": "string",
                "default": "csa",
            }
        }
    elif script_name == "dipy_fit_opdt":
        extra_args = {
            "method": {
                "action": "store",
                "dest": "method",
                "metavar": "string",
                "default": "opdt",
            }
        }
    elif script_name == "dipy_fit_qball":
        extra_args = {
            "method": {
                "action": "store",
                "dest": "method",
                "metavar": "string",
                "default": "qball",
            }
        }
    run_flow(getattr(mod, flow_name)(), extra_args=extra_args)
