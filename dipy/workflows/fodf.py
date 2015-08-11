from glob import glob
import os

import numpy as np
import nibabel as nib
import pickle

from dipy.reconst.peaks import peaks_from_model, reshape_peaks_for_visualization
from dipy.data import get_sphere
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response


def compute_fodf(input, mask, bvalues, bvectors, out_dir):
    print glob(input)
    print glob(mask)
    print glob(bvalues)
    print glob(bvectors)

    for dwi, maskfile, bval, bvec in zip(glob(input),
                                         glob(mask),
                                         glob(bvalues),
                                         glob(bvectors)):
        vol = nib.load(dwi)
        data = vol.get_data()
        affine = vol.get_affine()

        bvals, bvecs = read_bvals_bvecs(bval, bvec)
        if bvals.min() != 0:
            if bvals.min() > 20:
                raise ValueError('The minimal bvalue is greater than 20. ' +
                                 'This is highly suspicious. Please check ' +
                                 'your data to ensure everything is correct.\n' +
                                 'Value found: {0}'.format(bvals.min()))

            gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())
        else:
            gtab = gradient_table(bvals, bvecs)

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

        print("Eigenvalues for the frf of the input data are :", response[0])
        print("Ratio for smallest to largest eigen value is", ratio)

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

        if out_dir == '':
            out_dir_path = os.path.dirname(dwi)
        elif not os.path.isabs(out_dir):
            out_dir_path = os.path.join(os.path.dirname(dwi), out_dir)
            if not os.path.exists(out_dir_path):
                os.makedirs(out_dir_path)
        else:
            out_dir_path = out_dir

        nib.save(nib.Nifti1Image(peaks_csd.shm_coeff.astype(np.float32),
                                 affine),
                 os.path.join(out_dir_path, 'fodf.nii.gz'))

        nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_csd),
                                 affine),
                 os.path.join(out_dir_path, 'peaks.nii.gz'))

        nib.save(nib.Nifti1Image(peaks_csd.peak_values.astype(np.float32),
                                 affine),
                 os.path.join(out_dir_path, 'peaks_values.nii.gz'))

        nib.save(nib.Nifti1Image(peaks_csd.peak_indices, affine),
                 os.path.join(out_dir_path, 'peaks_indices.nii.gz'))
