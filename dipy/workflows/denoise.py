from glob import glob
import os

import nibabel as nib

from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import piesno, estimate_sigma
from dipy.workflows.utils import choose_create_out_dir

def nlmeans_flow(dwis, sigma=0, out_dir=''):

    for dwi_path in glob(dwis):
        image = nib.load(dwi_path)

        data = image.get_data()

        if sigma == 0:
            sigma = estimate_sigma(data)
            print 'Found sigma {0}'.format(sigma)

        out_dir_path = choose_create_out_dir(out_dir, dwi_path)

        denoised_data = nlmeans(data, sigma)
        denoised_image = nib.Nifti1Image(
            denoised_data, image.get_affine(), image.get_header())

        denoised_image.to_filename(
            os.path.join(out_dir_path, 'dwi_2x2x2_nlmeans.nii.gz'))
