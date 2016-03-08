from __future__ import division, print_function, absolute_import
from os.path import join as pjoin
import os
from glob import glob
import logging

from dipy.workflows.utils import choose_create_out_dir
from dipy.workflows.segment import median_otsu_flow
from dipy.workflows.denoise import nlmeans_flow
from dipy.workflows.reconst import dti_metrics_flow
from dipy.workflows.fodf import fodf_flow
from dipy.workflows.tracking import deterministic_tracking_flow
from dipy.workflows.track_metrics import track_density_flow

def simple_pipeline_flow(input_files, bvalues, bvectors, work_dir='',
                         resume=False):
    """ A simple dwi processing pipeline with the following steps:
        -Denoising
        -Masking
        -DTI reconstruction
        -HARDI recontructio
        -Deterministic tracking
        -Tracts metrics

    Parameters
    ----------
    input_files : string
        Path to the dwi volumes. This path may contain wildcards to process
        multiple inputs at once.
    bvalues : string
        Path to the bvalues files. This path may contain wildcards to use
        multiple bvalues files at once.
    bvectors : string
        Path to the bvalues files. This path may contain wildcards to use
        multiple bvalues files at once.
    work_dir : string, optional
        Working directory (default input file directory)
    resume : bool, optional
        If enabled, the pipeline will not run tasks if the output exists.

    Outputs
    -------
    """
    for dwi, bval, bvec in zip(glob(input_files),
                                     glob(bvalues),
                                     glob(bvectors)):

        basename = os.path.basename(dwi)
        while '.' in basename:
            basename = os.path.splitext(basename)[0]

        nifti_basename = basename + '_{0}.nii.gz'
        work_dir = choose_create_out_dir(work_dir, dwi)

        mask_filename = pjoin(work_dir, nifti_basename.format('mask'))
        if os.path.exists(mask_filename) is False or resume is False:
            median_otsu_flow(dwi, out_dir=work_dir, mask=mask_filename)
        else:
            logging.info('Skipped median otsu segmentation')

        denoised_dwi = pjoin(work_dir, nifti_basename.format('nlmeans'))
        if os.path.exists(denoised_dwi) is False or resume is False:
            nlmeans_flow(dwi, out_dir=work_dir, denoised=denoised_dwi)
        else:
            logging.info('Skipped nlmeans denoise')

        metrics_dir = pjoin(work_dir, 'metrics')
        fa_path = pjoin(metrics_dir, 'fa.nii.gz')
        if os.path.exists(fa_path) is False or resume is False:
            dti_metrics_flow(denoised_dwi, bval, bvec,
                             out_dir=metrics_dir, mask_files=mask_filename)
        else:
            logging.info('Skipped dti metrics')

        peaks_dir = pjoin(work_dir, 'peaks')
        if os.path.exists(peaks_dir) is False or resume is False:
            fodf_flow(denoised_dwi, bval, bvec, out_dir=peaks_dir,
                      mask_files=mask_filename)
        else:
            logging.info('Skipped fodf')

        tractograms_dir = pjoin(work_dir, 'tractograms')
        tractogram = 'deterministic_tractogram.trk'

        tractogram_path = pjoin(tractograms_dir, tractogram)
        if os.path.exists(tractogram_path) is False or resume is False:
            deterministic_tracking_flow(denoised_dwi, mask_filename, bval, bvec,
                                        out_dir=tractograms_dir,
                                        tractogram=tractogram)
        else:
            logging.info('Skipped deterministic tracking')

        tdi = os.path.join(metrics_dir, 'tdi.nii.gz')
        if os.path.exists(tdi) is False or resume is False:
            track_density_flow(tractogram_path, fa_path, out_dir=metrics_dir, tdi=tdi)
        else:
            logging.info('Skipped track density')
