from glob import glob
import os

import nibabel as nib

from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import piesno, estimate_sigma
from dipy.workflows.utils import choose_create_out_dir


def nlmeans_flow(input_files, out_dir='',
                 out_filename='dwi_2x2x2_nlmeans.nii.gz', sigma=0):
    """ Workflow wrapping the nlmeans denoising method.

    It applies nlmeans denoise on each file found by 'globing'
    ``input_files`` and saves the results in a directory specified by
    ``out_dir``.

    Parameters
    ----------
    input_files : string
        Path to the input volumes. This path may contain wildcards to process
        multiple inputs at once.
    out_dir : string, optional
        Output directory (default input file directory)
    out_filename : string, optional
        Name of the resuting denoised volume (default dwi_2x2x2_nlmeans.nii.gz)
    sigma : float,
        Radius (in voxels) of the applied median filter(default 4)

    Outputs
    -------
    denoised : Nifti File
           The denoised volume.
    """
    for fpath in glob(input_files):
        print ''
        print('Denoising {0}'.format(fpath))

        image = nib.load(fpath)
        data = image.get_data()

        if sigma == 0:
            sigma = estimate_sigma(data)
            print 'Found sigma {0}'.format(sigma)

        denoised_data = nlmeans(data, sigma)
        denoised_image = nib.Nifti1Image(
            denoised_data, image.get_affine(), image.get_header())

        out_dir_path = choose_create_out_dir(out_dir, fpath)
        out_file_path = os.path.join(out_dir_path, out_filename)

        denoised_image.to_filename(out_file_path)
        print('Denoised volume saved as {0}'.format(out_file_path))

