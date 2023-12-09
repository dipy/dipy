#!python

from dipy.utils.optpkg import optional_package
from dipy.workflows.flow_runner import run_flow


def dipy_align_affine():
    align, _, _ = optional_package('dipy.workflows.align')
    run_flow(align.ImageRegistrationFlow())


def dipy_align_syn():
    align, _, _ = optional_package('dipy.workflows.align')
    run_flow(align.SynRegistrationFlow())


def dipy_apply_transform():
    align, _, _ = optional_package('dipy.workflows.align')
    run_flow(align.ApplyTransformFlow())


def dipy_buan_lmm():
    stats, _, _ = optional_package('dipy.workflows.stats')
    run_flow(stats.LinearMixedModelsFlow())


def dipy_buan_shapes():
    stats, _, _ = optional_package('dipy.workflows.stats')
    run_flow(stats.BundleShapeAnalysis())


def dipy_buan_profiles():
    stats, _, _ = optional_package('dipy.workflows.stats')
    run_flow(stats.BundleAnalysisTractometryFlow())


def dipy_bundlewarp():
    align, _, _ = optional_package('dipy.workflows.align')
    run_flow(align.BundleWarpFlow())


def dipy_correct_motion():
    align, _, _ = optional_package('dipy.workflows.align')
    run_flow(align.MotionCorrectionFlow())


def dipy_denoise_nlmeans():
    denoise, _, _ = optional_package('dipy.workflows.denoise')
    run_flow(denoise.NLMeansFlow())


def dipy_denoise_lpca():
    denoise, _, _ = optional_package('dipy.workflows.denoise')
    run_flow(denoise.LPCAFlow())


def dipy_denoise_mppca():
    denoise, _, _ = optional_package('dipy.workflows.denoise')
    run_flow(denoise.MPPCAFlow())


def dipy_denoise_patch2self():
    denoise, _, _ = optional_package('dipy.workflows.denoise')
    run_flow(denoise.Patch2SelfFlow())


def dipy_evac_plus():
    nn, _, _ = optional_package('dipy.workflows.nn')
    run_flow(nn.EVACPlusFlow())


def dipy_fetch():
    io, _, _ = optional_package('dipy.workflows.io')
    run_flow(io.FetchFlow())


def dipy_fit_csa():
    reconst, _, _ = optional_package('dipy.workflows.reconst')
    run_flow(reconst.ReconstCSAFlow())


def dipy_fit_csd():
    reconst, _, _ = optional_package('dipy.workflows.reconst')
    run_flow(reconst.ReconstCSDFlow())


def dipy_fit_dki():
    reconst, _, _ = optional_package('dipy.workflows.reconst')
    run_flow(reconst.ReconstDkiFlow())


def dipy_fit_dti():
    reconst, _, _ = optional_package('dipy.workflows.reconst')
    run_flow(reconst.ReconstDtiFlow())


def dipy_fit_ivim():
    reconst, _, _ = optional_package('dipy.workflows.reconst')
    run_flow(reconst.ReconstIvimFlow())


def dipy_fit_mapmri():
    reconst, _, _ = optional_package('dipy.workflows.reconst')
    run_flow(reconst.ReconstMAPMRIFlow())


def dipy_mask():
    mask, _, _ = optional_package('dipy.workflows.mask')
    run_flow(mask.MaskFlow())


def dipy_gibbs_ringing():
    denoise, _, _ = optional_package('dipy.workflows.denoise')
    run_flow(denoise.GibbsRingingFlow())


def dipy_horizon():
    viz, _, _ = optional_package('dipy.workflows.viz')
    run_flow(viz.HorizonFlow())


def dipy_info():
    io, _, _ = optional_package('dipy.workflows.io')
    run_flow(io.IoInfoFlow())


def dipy_labelsbundles():
    segment, _, _ = optional_package('dipy.workflows.segment')
    run_flow(segment.LabelsBundlesFlow())


def dipy_median_otsu():
    segment, _, _ = optional_package('dipy.workflows.segment')
    run_flow(segment.MedianOtsuFlow())


def dipy_recobundles():
    segment, _, _ = optional_package('dipy.workflows.segment')
    run_flow(segment.RecoBundlesFlow())


def dipy_reslice():
    align, _, _ = optional_package('dipy.workflows.align')
    run_flow(align.ResliceFlow())


def dipy_snr_in_cc():
    stats, _, _ = optional_package('dipy.workflows.stats')
    run_flow(stats.SNRinCCFlow())


def dipy_split():
    io, _, _ = optional_package('dipy.workflows.io')
    run_flow(io.SplitFlow())


def dipy_track():
    tracking, _, _ = optional_package('dipy.workflows.tracking')
    run_flow(tracking.LocalFiberTrackingPAMFlow())


def dipy_track_pft():
    tracking, _, _ = optional_package('dipy.workflows.tracking')
    run_flow(tracking.PFTrackingPAMFlow())


def dipy_slr():
    align, _, _ = optional_package('dipy.workflows.align')
    run_flow(align.StreamlineLinearRegistrationFlow())
