import logging
from glob import glob
import os

import numpy as np
import nibabel as nib

from dipy.reconst.peaks import peaks_from_model, reshape_peaks_for_visualization
from dipy.data import get_sphere
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response

from dipy.workflows.utils import choose_create_out_dir

def fodf_flow(input_files, mask_files, bvalues, bvectors, out_dir='',
              b0_threshold=0.0, fodf='fodf.nii.gz', peaks='peaks.nii.gz',
              peaks_values='peaks_values.nii.gz',
              peaks_indices='peaks_indices.nii.gz'):
    """ Workflow for peaks computation. Peaks computation is done by 'globing'
        ``input_files`` and saves the peaks in a directory specified by
        ``out_dir``.

    Parameters
    ----------
    input_files : string
        Path to the input volumes. This path may contain wildcards to process
        multiple inputs at once.
    mask_files : string
        Path to the input masks. This path may contain wildcards to use
        multiple masks at once.
    bvalues : string
        Path to the bvalues files. This path may contain wildcards to use
        multiple bvalues files at once.
    bvectors : string
        Path to the bvalues files. This path may contain wildcards to use
        multiple bvalues files at once.
    out_dir : string, optional
        Output directory (default input file directory)
    b0_threshold : float, optional
        Threshold used to find b=0 directions
    fodf : string, optional
        Name of the fodf volume to be saved (default 'fodf.nii.gz')
    peaks : string, optional
        Name of the peaks volume to be saved (default 'peaks.nii.gz')
    peaks_values : string, optional
        Name of the peaks_values volume to be saved (default 'peaks_values.nii.gz')
    peaks_indices : string, optional
        Name of the peaks_indices volume to be saved (default 'peaks_indices.nii.gz')
    Outputs
    -------
    fodf : Nifti file
        Fodf volume
    peaks : Nifti file
        Peaks volume
    peaks_values : Nifti file
        Peaks_values volume
    peaks_indices : string, optional
        Peaks_indices volume
    """

    for dwi, maskfile, bval, bvec in zip(glob(input_files),
                                         glob(mask_files),
                                         glob(bvalues),
                                         glob(bvectors)):

        logging.info('Computing fiber odfs for {0}'.format(dwi))
        vol = nib.load(dwi)
        data = vol.get_data()
        affine = vol.get_affine()

        bvals, bvecs = read_bvals_bvecs(bval, bvec)
        gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)
        mask_vol = nib.load(maskfile).get_data().astype(np.bool)

        sh_order = 8
        if data.shape[-1] < 15:
            raise ValueError('You need at least 15 unique DWI volumes to '
                             'compute fiber odfs. You currently have: {0}'
                             ' DWI volumes.'.format(data.shape[-1]))

        elif data.shape[-1] < 30:
            sh_order = 6

        response, ratio = auto_response(gtab, data)
        response = list(response)

        logging.info('Eigenvalues for the frf of the input data are : {0}'.format(response[0]))
        logging.info('Ratio for smallest to largest eigen value is {0}'.format(ratio))

        peaks_sphere = get_sphere('symmetric362')

        csd_model = ConstrainedSphericalDeconvModel(gtab, response,
                                                    sh_order=sh_order)

        peaks_csd = peaks_from_model(model=csd_model,
                                     data=data,
                                     sphere=peaks_sphere,
                                     relative_peak_threshold=.5,
                                     min_separation_angle=25,
                                     mask=mask_vol,
                                     return_sh=True,
                                     sh_order=sh_order,
                                     normalize_peaks=True,
                                     parallel=False)

        out_dir_path = choose_create_out_dir(out_dir, dwi)

        nib.save(nib.Nifti1Image(peaks_csd.shm_coeff.astype(np.float32),
                                 affine),
                 os.path.join(out_dir_path, fodf))

        nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_csd),
                                 affine),
                 os.path.join(out_dir_path, peaks))

        nib.save(nib.Nifti1Image(peaks_csd.peak_values.astype(np.float32),
                                 affine),
                 os.path.join(out_dir_path, peaks_values))

        nib.save(nib.Nifti1Image(peaks_csd.peak_indices, affine),
                 os.path.join(out_dir_path, peaks_indices))

        logging.info('Finished computing fiber odfs.')
