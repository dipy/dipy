#!python
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
    "dipy_buan_profiles": ("dipy.workflows.stats",
                           "BundleAnalysisTractometryFlow"),
    "dipy_bundlewarp": ("dipy.workflows.align", "BundleWarpFlow"),
    "dipy_correct_motion": ("dipy.workflows.align", "MotionCorrectionFlow"),
    "dipy_convert_tractogram": ("dipy.workflows.io", "ConvertTractogramFlow"),
    "dipy_convert_tensors": ("dipy.workflows.io", "ConvertTensorsFlow"),
    "dipy_denoise_nlmeans": ("dipy.workflows.denoise", "NLMeansFlow"),
    "dipy_denoise_lpca": ("dipy.workflows.denoise", "LPCAFlow"),
    "dipy_denoise_mppca": ("dipy.workflows.denoise", "MPPCAFlow"),
    "dipy_denoise_patch2self": ("dipy.workflows.denoise", "Patch2SelfFlow"),
    "dipy_evac_plus": ("dipy.workflows.nn", "EVACPlusFlow"),
    "dipy_fetch": ("dipy.workflows.io", "FetchFlow"),
    "dipy_fit_csa": ("dipy.workflows.reconst", "ReconstCSAFlow"),
    "dipy_fit_csd": ("dipy.workflows.reconst", "ReconstCSDFlow"),
    "dipy_fit_dki": ("dipy.workflows.reconst", "ReconstDkiFlow"),
    "dipy_fit_dti": ("dipy.workflows.reconst", "ReconstDtiFlow"),
    "dipy_fit_dsi": ("dipy.workflows.reconst", "ReconstDsiFlow"),
    "dipy_fit_ivim": ("dipy.workflows.reconst", "ReconstIvimFlow"),
    "dipy_fit_mapmri": ("dipy.workflows.reconst", "ReconstMAPMRIFlow"),
    "dipy_mask": ("dipy.workflows.mask", "MaskFlow"),
    "dipy_gibbs_ringing": ("dipy.workflows.denoise", "GibbsRingingFlow"),
    "dipy_horizon": ("dipy.workflows.viz", "HorizonFlow"),
    "dipy_info": ("dipy.workflows.io", "IoInfoFlow"),
    "dipy_concatenate_tractograms": ("dipy.workflows.io",
                                     "ConcatenateTractogramFlow"),
    "dipy_labelsbundles": ("dipy.workflows.segment", "LabelsBundlesFlow"),
    "dipy_median_otsu": ("dipy.workflows.segment", "MedianOtsuFlow"),
    "dipy_recobundles": ("dipy.workflows.segment", "RecoBundlesFlow"),
    "dipy_reslice": ("dipy.workflows.align", "ResliceFlow"),
    "dipy_sh_convert_mrtrix": ("dipy.workflows.io", "ConvertSHFlow"),
    "dipy_snr_in_cc": ("dipy.workflows.stats", "SNRinCCFlow"),
    "dipy_split": ("dipy.workflows.io", "SplitFlow"),
    "dipy_track": ("dipy.workflows.tracking", "LocalFiberTrackingPAMFlow"),
    "dipy_track_pft": ("dipy.workflows.tracking", "PFTrackingPAMFlow"),
    "dipy_slr": ("dipy.workflows.align", "SlrWithQbxFlow"),
}


def run():
    """Run scripts located in pyproject.toml."""
    script_name = os.path.basename(sys.argv[0])
    mod_name, flow_name = cli_flows.get(script_name, (None, None))
    if mod_name is None:
        print(f"Flow: {script_name} not Found in DIPY")
        print("Available flows: %s" % ", ".join(cli_flows.keys()))
        sys.exit(1)
    mod, _, _ = optional_package(mod_name)
    run_flow(getattr(mod, flow_name)())
