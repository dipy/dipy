from __future__ import division, print_function, absolute_import
from os.path import join as pjoin
from glob import glob

import nibabel as nib
import numpy as np
from dipy.workflows.utils import choose_create_out_dir
from dipy.workflows.segment import median_otsu_flow
from dipy.workflows.denoise import nlmeans_flow
from dipy.workflows.reconst import dti_metrics_flow
from dipy.workflows.fodf import fodf_flow
from dipy.workflows.tracking import deterministic_tracking_flow
from dipy.workflows.tract_metrics import tract_density_flow

def simple_pipeline_flow(input_files, bvalues, bvectors, work_dir=''):
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
    """

    for dwi, bval, bvec in zip(glob(input_files),
                                     glob(bvalues),
                                     glob(bvectors)):

        work_dir = choose_create_out_dir(work_dir, dwi)

        mask_filename = pjoin(work_dir, 'brain_mask.nii.gz')
        median_otsu_flow(dwi, out_dir=work_dir, mask=mask_filename)

        denoised_dwi = pjoin(work_dir, 'dwi_2x2x2_nlmeans.nii.gz')
        nlmeans_flow(dwi, out_dir=work_dir, denoised=denoised_dwi)


        #if not params.resume or not all_files_exist(denoised_dwi):
        #    print 'path does not exists {0}'.format(denoised_dwi)
        #    nlmeans_flow(params.input, out_dir='work')

        #mask_fname = fname + '_nlmeans_mask' + ext
        #mask_path = os.path.join(work_dir, mask_fname)

        #if not params.resume or not all_files_exist(mask_path):
        #    median_otsu_flow(denoised_dwi, '', 'True', '4', '4')

        metrics_dir = pjoin(work_dir, 'metrics')
        fa_path = pjoin(metrics_dir, 'fa.nii.gz')
        dti_metrics_flow(denoised_dwi, mask_filename, bval, bvec,
                         out_dir=metrics_dir)

        #fodf_path = pjoin(metrics_dir, 'fodf.nii.gz')
        #if not params.resume or not all_files_exist(fodf_path):
        peaks_dir = pjoin(work_dir, 'peaks')
        fodf_flow(denoised_dwi, mask_filename, bval, bvec, out_dir=peaks_dir)

        tractograms_dir = pjoin(work_dir, 'tractograms')
        tractogram = 'deterministic_tractogram.trk'
        deterministic_tracking_flow(denoised_dwi, mask_filename, bval, bvec,
                                    out_dir=tractograms_dir,
                                    tractogram=tractogram)

        tract_density_flow(tractogram, fa_path, out_dir=metrics_dir)