from __future__ import division, print_function, absolute_import
from os.path import join as pjoin
import os
from glob import glob
import logging
import inspect

from dipy.workflows.utils import choose_create_out_dir
from dipy.workflows.multi_io import io_iterator_
from dipy.workflows.segment import median_otsu_flow
from dipy.workflows.denoise import nlmeans_flow
from dipy.workflows.reconst import dti_metrics_flow
from dipy.workflows.fodf import fodf_flow
from dipy.workflows.tracking import EuDX_tracking_flow
from dipy.workflows.track_metrics import track_density_flow

def simple_pipeline_flow(input_files, bvalues, bvectors, resume=False,
                         out_dir=''):
    """ A simple dwi processing pipeline with the following steps:
        -Denoising
        -Masking
        -DTI reconstruction
        -HARDI recontruction
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
    resume : bool, optional
        If enabled, the pipeline will not run tasks if the output exists.
    out_dir : string, optional
        Working directory (default input file directory)
    """
    io_it = io_iterator_(inspect.currentframe(), simple_pipeline_flow)
    for dwi, bval, bvec in io_it:

        basename = os.path.basename(dwi)
        while '.' in basename:
            basename = os.path.splitext(basename)[0]

        nifti_basename = basename + '_{0}.nii.gz'
        #out_dir = choose_create_out_dir(out_dir, dwi)

        #mask_filename = pjoin(out_dir, nifti_basename.format('mask'))
        #if os.path.exists(mask_filename) is False or resume is False:
        #    median_otsu_flow(dwi, out_dir=out_dir, out_mask=mask_filename)
        #else:
        #    logging.info('Skipped median otsu segmentation')

        mask_filename = nifti_basename.format('mask')
        if os.path.exists(mask_filename) is False or resume is False:
            median_otsu_flow(dwi, out_dir=out_dir, out_mask=mask_filename)
        else:
            logging.info('Skipped median otsu segmentation')

        denoised_dwi = pjoin(out_dir, nifti_basename.format('nlmeans'))
        if os.path.exists(denoised_dwi) is False or resume is False:
            nlmeans_flow(dwi, out_dir=out_dir, out_denoised=denoised_dwi)
        else:
            logging.info('Skipped nlmeans denoise')

        metrics_dir = pjoin(out_dir, 'metrics')
        fa_path = pjoin(metrics_dir, 'fa.nii.gz')
        if os.path.exists(fa_path) is False or resume is False:
            dti_metrics_flow(denoised_dwi, bval, bvec, mask_files=mask_filename,
                             out_dir=metrics_dir)
        else:
            logging.info('Skipped dti metrics')

        peaks_dir = pjoin(out_dir, 'peaks')
        peaks_values = pjoin(peaks_dir, 'peaks_values.nii.gz')
        peaks_indices = pjoin(peaks_dir, 'peaks_indices.nii.gz')

        if os.path.exists(peaks_dir) is False or resume is False:
            fodf_flow(denoised_dwi, bval, bvec, mask_files=mask_filename,
                      out_dir=peaks_dir, out_peaks_values=peaks_values,
                      out_peaks_indices=peaks_indices)
        else:
            logging.info('Skipped fodf')

        tractograms_dir = pjoin(out_dir, 'tractograms')
        tractogram = 'eudx_tractogram.trk'
        tractogram_path = pjoin(tractograms_dir, tractogram)
        if os.path.exists(tractogram_path) is False or resume is False:
            EuDX_tracking_flow(peaks_values, peaks_indices,
                               out_dir=tractograms_dir, out_tractogram=tractogram)
        else:
           logging.info('Skipped deterministic tracking')

        tdi = os.path.join(metrics_dir, 'tdi.nii.gz')
        tdi_2x = os.path.join(metrics_dir, 'tdi_2x.nii.gz')
        if os.path.exists(tdi) is False or resume is False:
            track_density_flow(tractogram_path, fa_path, out_dir=metrics_dir,
                               out_tdi=tdi)

            track_density_flow(tractogram_path, fa_path, out_dir=metrics_dir,
                               out_tdi=tdi_2x, up_factor=2.0)
        else:
            logging.info('Skipped track density')
