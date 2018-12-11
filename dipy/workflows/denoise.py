from __future__ import division, print_function, absolute_import

import logging
import shutil

from dipy.io.image import load_nifti, save_nifti
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.workflows.workflow import Workflow


class NLMeansFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'nlmeans'

    def run(self, input_files, sigma=0, out_dir='',
            out_denoised='dwi_nlmeans.nii.gz'):
        """ Workflow wrapping the nlmeans denoising method.

        It applies nlmeans denoise on each file found by 'globing'
        ``input_files`` and saves the results in a directory specified by
        ``out_dir``.

        Parameters
        ----------
        input_files : string
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        sigma : float, optional
            Sigma parameter to pass to the nlmeans algorithm
            (default: auto estimation).
        out_dir : string, optional
            Output directory (default input file directory)
        out_denoised : string, optional
            Name of the resuting denoised volume (default: dwi_nlmeans.nii.gz)
        """
        io_it = self.get_io_iterator()
        for fpath, odenoised in io_it:
            if self._skip:
                shutil.copy(fpath, odenoised)
                logging.warning('Denoising skipped for now.')
            else:
                logging.info('Denoising {0}'.format(fpath))
                data, affine, image = load_nifti(fpath, return_img=True)

                if sigma == 0:
                    logging.info('Estimating sigma')
                    sigma = estimate_sigma(data)
                    logging.debug('Found sigma {0}'.format(sigma))

                denoised_data = nlmeans(data, sigma)
                save_nifti(odenoised, denoised_data, affine, image.header)

                logging.info('Denoised volume saved as {0}'.format(odenoised))
