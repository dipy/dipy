#!python

from dipy.workflows.flow_runner import run_flow
from dipy.workflows import (align, denoise, io, mask, nn, reconst, segment,
                            stats, tracking, viz)


def dipy_align_affine():
    run_flow(align.ImageRegistrationFlow())


def dipy_align_syn():
    run_flow(align.SynRegistrationFlow())


def dipy_apply_transform():
    run_flow(align.ApplyTransformFlow())


def dipy_buan_lmm():
    run_flow(stats.LinearMixedModelsFlow())


def dipy_buan_shapes():
    run_flow(stats.BundleShapeAnalysis())


def dipy_buan_profiles():
    run_flow(stats.BundleAnalysisTractometryFlow())


def dipy_bundlewarp():
    run_flow(align.BundleWarpFlow())


def dipy_correct_motion():
    run_flow(align.MotionCorrectionFlow())


def dipy_denoise_nlmeans():
    run_flow(denoise.NLMeansFlow())


def dipy_denoise_lpca():
    run_flow(denoise.LPCAFlow())


def dipy_denoise_mppca():
    run_flow(denoise.MPPCAFlow())


def dipy_denoise_patch2self():
    run_flow(denoise.Patch2SelfFlow())


def dipy_evac_plus():
    run_flow(nn.EVACPlusFlow())


def dipy_fetch():
    run_flow(io.FetchFlow())


def dipy_fit_csa():
    run_flow(reconst.ReconstCSAFlow())


def dipy_fit_csd():
    run_flow(reconst.ReconstCSDFlow())


def dipy_fit_dki():
    run_flow(reconst.ReconstDkiFlow())


def dipy_fit_dti():
    run_flow(reconst.ReconstDtiFlow())


def dipy_fit_ivim():
    run_flow(reconst.ReconstIvimFlow())


def dipy_fit_mapmri():
    run_flow(reconst.ReconstMAPMRIFlow())


def dipy_mask():
    run_flow(mask.MaskFlow())


def dipy_gibbs_ringing():
    run_flow(denoise.GibbsRingingFlow())


def dipy_horizon():
    run_flow(viz.HorizonFlow())


def dipy_info():
    run_flow(io.IoInfoFlow())


def dipy_labelsbundles():
    run_flow(segment.LabelsBundlesFlow())


def dipy_median_otsu():
    run_flow(segment.MedianOtsuFlow())


def dipy_recobundles():
    run_flow(segment.RecoBundlesFlow())


def dipy_reslice():
    run_flow(align.ResliceFlow())


def dipy_snr_in_cc():
    run_flow(stats.SNRinCCFlow())


def dipy_split():
    run_flow(io.SplitFlow())


def dipy_track():
    run_flow(tracking.LocalFiberTrackingPAMFlow())


def dipy_track_pft():
    run_flow(tracking.PFTrackingPAMFlow())


def dipy_slr():
    run_flow(align.StreamlineLinearRegistrationFlow())
