import logging
import shutil
import numpy as np

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.denoise.patch2self import patch2self
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.localpca import localpca, mppca
from dipy.denoise.gibbs import gibbs_removal
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.denoise.pca_noise_estimate import pca_noise_estimate
from dipy.workflows.workflow import Workflow


class Patch2SelfFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'patch2self'

    def run(self, input_files, bval_files, model='ols',
            b0_threshold=50, alpha=1.0, verbose=False, patch_radius=0,
            b0_denoising=True, clip_negative_vals=False, shift_intensity=True,
            out_dir='', out_denoised='dwi_patch2self.nii.gz'):
        """Workflow for Patch2Self denoising method.

        It applies patch2self denoising on each file found by 'globing'
        ``input_file`` and ``bval_file``. It saves the results in a directory
        specified by ``out_dir``.

        Parameters
        ----------
        input_files : string
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        bval_files : string
            bval file associated with the diffusion data.
        model : string, or initialized linear model object.
            This will determine the algorithm used to solve the set of linear
            equations underlying this model. If it is a string it needs to be
            one of the following: {'ols', 'ridge', 'lasso'}. Otherwise,
            it can be an object that inherits from
            `dipy.optimize.SKLearnLinearSolver` or an object with a similar
            interface from Scikit-Learn:
            `sklearn.linear_model.LinearRegression`,
            `sklearn.linear_model.Lasso` or `sklearn.linear_model.Ridge`
            and other objects that inherit from `sklearn.base.RegressorMixin`.
            Default: 'ols'.
        b0_threshold : int, optional
            Threshold for considering volumes as b0.
        alpha : float, optional
            Regularization parameter only for ridge regression model.
        verbose : bool, optional
            Show progress of Patch2Self and time taken.
        patch_radius : variable int, optional
            The radius of the local patch to be taken around each voxel
        b0_denoising : bool, optional
            Skips denoising b0 volumes if set to False.
        clip_negative_vals : bool, optional
            Sets negative values after denoising to 0 using `np.clip`.
        shift_intensity : bool, optional
            Shifts the distribution of intensities per volume to give
            non-negative values
        out_dir : string, optional
            Output directory (default current directory)
        out_denoised : string, optional
            Name of the resulting denoised volume
            (default: dwi_patch2self.nii.gz)

        References
        ----------
        .. [Fadnavis20] S. Fadnavis, J. Batson, E. Garyfallidis, Patch2Self:
                    Denoising Diffusion MRI with Self-supervised Learning,
                    Advances in Neural Information Processing Systems 33 (2020)

        """
        io_it = self.get_io_iterator()
        if isinstance(patch_radius, list) and len(patch_radius) == 1:
            patch_radius = int(patch_radius[0])
        for fpath, bvalpath, odenoised in io_it:
            if self._skip:
                shutil.copy(fpath, odenoised)
                logging.warning('Denoising skipped for now.')
            else:
                logging.info('Denoising %s', fpath)
                data, affine, image = load_nifti(fpath, return_img=True)
                bvals = np.loadtxt(bvalpath)

                denoised_data = patch2self(
                    data, bvals, model=model, b0_threshold=b0_threshold,
                    alpha=alpha, verbose=verbose, patch_radius=patch_radius,
                    b0_denoising=b0_denoising,
                    clip_negative_vals=clip_negative_vals,
                    shift_intensity=shift_intensity,
                )
                save_nifti(odenoised, denoised_data, affine, image.header)

                logging.info('Denoised volumes saved as %s', odenoised)


class NLMeansFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'nlmeans'

    def run(self, input_files, sigma=0, patch_radius=1, block_radius=5,
            rician=True, out_dir='', out_denoised='dwi_nlmeans.nii.gz'):
        """Workflow wrapping the nlmeans denoising method.

        It applies nlmeans denoise on each file found by 'globing'
        ``input_files`` and saves the results in a directory specified by
        ``out_dir``.

        Parameters
        ----------
        input_files : string
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        sigma : float, optional
            Sigma parameter to pass to the nlmeans algorithm.
        patch_radius : int, optional
            patch size is ``2 x patch_radius + 1``.
        block_radius : int, optional
            block size is ``2 x block_radius + 1``.
        rician : bool, optional
            If True the noise is estimated as Rician, otherwise Gaussian noise
            is assumed.
        out_dir : string, optional
            Output directory. (default current directory)
        out_denoised : string, optional
            Name of the resulting denoised volume.

        References
        ----------
        .. [Descoteaux08] Descoteaux, Maxime and Wiest-Daesslé, Nicolas and
        Prima, Sylvain and Barillot, Christian and Deriche, Rachid.
        Impact of Rician Adapted Non-Local Means Filtering on
        HARDI, MICCAI 2008

        """
        io_it = self.get_io_iterator()
        for fpath, odenoised in io_it:
            if self._skip:
                shutil.copy(fpath, odenoised)
                logging.warning('Denoising skipped for now.')
            else:
                logging.info('Denoising %s', fpath)
                data, affine, image = load_nifti(fpath, return_img=True)

                if sigma == 0:
                    logging.info('Estimating sigma')
                    sigma = estimate_sigma(data)
                    logging.debug('Found sigma {0}'.format(sigma))

                denoised_data = nlmeans(data, sigma=sigma,
                                        patch_radius=patch_radius,
                                        block_radius=block_radius,
                                        rician=rician)
                save_nifti(odenoised, denoised_data, affine, image.header)

                logging.info('Denoised volume saved as %s', odenoised)


class LPCAFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'lpca'

    def run(self, input_files, bvalues_files, bvectors_files, sigma=0,
            b0_threshold=50, bvecs_tol=0.01, patch_radius=2, pca_method='eig',
            tau_factor=2.3, out_dir='', out_denoised='dwi_lpca.nii.gz'):
        r"""Workflow wrapping LPCA denoising method.

        Parameters
        ----------
        input_files : string
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        bvalues_files : string
            Path to the bvalues files. This path may contain wildcards to use
            multiple bvalues files at once.
        bvectors_files : string
            Path to the bvectors files. This path may contain wildcards to use
            multiple bvectors files at once.
        sigma : float, optional
            Standard deviation of the noise estimated from the data.
            Default 0: it means sigma value estimation with the Manjon2013
            algorithm [3]_.
        b0_threshold : float, optional
            Threshold used to find b0 volumes.
        bvecs_tol : float, optional
            Threshold used to check that norm(bvec) = 1 +/- bvecs_tol
            b-vectors are unit vectors.
        patch_radius : int, optional
            The radius of the local patch to be taken around each voxel (in
            voxels) For example, for a patch radius with value 2, and assuming
            the input image is a 3D image, the denoising will take place in
            blocks of 5x5x5 voxels.
        pca_method : string, optional
            Use either eigenvalue decomposition ('eig') or singular value
            decomposition ('svd') for principal component analysis. The default
            method is 'eig' which is faster. However, occasionally 'svd' might
            be more accurate.
        tau_factor : float, optional
            Thresholding of PCA eigenvalues is done by nulling out eigenvalues
            that are smaller than:

            .. math ::

                    \tau = (\tau_{factor} \sigma)^2

            \tau_{factor} can be change to adjust the relationship between the
            noise standard deviation and the threshold \tau. If \tau_{factor}
            is set to None, it will be automatically calculated using the
            Marcenko-Pastur distribution [2]_.
        out_dir : string, optional
            Output directory. (default current directory)
        out_denoised : string, optional
            Name of the resulting denoised volume.

        References
        ----------
        .. [1] Veraart J, Novikov DS, Christiaens D, Ades-aron B, Sijbers,
        Fieremans E, 2016. Denoising of Diffusion MRI using random
        matrix theory. Neuroimage 142:394-406.
        doi: 10.1016/j.neuroimage.2016.08.016

        .. [2] Veraart J, Fieremans E, Novikov DS. 2016. Diffusion MRI noise
        mapping using random matrix theory. Magnetic Resonance in
        Medicine.
        doi: 10.1002/mrm.26059.

        .. [3] Manjon JV, Coupe P, Concha L, Buades A, Collins DL (2013)
        Diffusion Weighted Image Denoising Using Overcomplete Local
        PCA. PLoS ONE 8(9): e73021.
        https://doi.org/10.1371/journal.pone.0073021

        """
        io_it = self.get_io_iterator()
        if isinstance(patch_radius, list) and len(patch_radius) == 1:
            patch_radius = int(patch_radius[0])
        for dwi, bval, bvec, odenoised in io_it:
            logging.info('Denoising %s', dwi)
            data, affine, image = load_nifti(dwi, return_img=True)

            if not sigma:
                logging.info('Estimating sigma')
                bvals, bvecs = read_bvals_bvecs(bval, bvec)
                gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold,
                                      atol=bvecs_tol)
                sigma = pca_noise_estimate(data, gtab, correct_bias=True,
                                           smooth=3)
                logging.debug('Found sigma %s', sigma)

            denoised_data = localpca(data, sigma=sigma,
                                     patch_radius=patch_radius,
                                     pca_method=pca_method,
                                     tau_factor=tau_factor)
            save_nifti(odenoised, denoised_data, affine, image.header)

            logging.info('Denoised volume saved as %s', odenoised)


class MPPCAFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'mppca'

    def run(self, input_files, patch_radius=2, pca_method='eig',
            return_sigma=False, out_dir='', out_denoised='dwi_mppca.nii.gz',
            out_sigma='dwi_sigma.nii.gz'):
        r"""Workflow wrapping Marcenko-Pastur PCA denoising method.

        Parameters
        ----------
        input_files : string
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        patch_radius : variable int, optional
            The radius of the local patch to be taken around each voxel (in
            voxels) For example, for a patch radius with value 2, and assuming
            the input image is a 3D image, the denoising will take place in
            blocks of 5x5x5 voxels.
        pca_method : string, optional
            Use either eigenvalue decomposition ('eig') or singular value
            decomposition ('svd') for principal component analysis. The default
            method is 'eig' which is faster. However, occasionally 'svd' might
            be more accurate.
        return_sigma : bool, optional
            If true, a noise standard deviation estimate based on the
            Marcenko-Pastur distribution is returned [2]_.
        out_dir : string, optional
            Output directory. (default current directory)
        out_denoised : string, optional
            Name of the resulting denoised volume.
        out_sigma : string, optional
            Name of the resulting sigma volume.

        References
        ----------
        .. [1] Veraart J, Novikov DS, Christiaens D, Ades-aron B, Sijbers,
        Fieremans E, 2016. Denoising of Diffusion MRI using random matrix
        theory. Neuroimage 142:394-406.
        doi: 10.1016/j.neuroimage.2016.08.016

        .. [2] Veraart J, Fieremans E, Novikov DS. 2016. Diffusion MRI noise
        mapping using random matrix theory. Magnetic Resonance in Medicine.
        doi: 10.1002/mrm.26059.

        """
        io_it = self.get_io_iterator()
        if isinstance(patch_radius, list) and len(patch_radius) == 1:
            patch_radius = int(patch_radius[0])

        for dwi, odenoised, osigma in io_it:
            logging.info('Denoising %s', dwi)
            data, affine, image = load_nifti(dwi, return_img=True)

            denoised_data, sigma = mppca(data, patch_radius=patch_radius,
                                         pca_method=pca_method,
                                         return_sigma=True)

            save_nifti(odenoised, denoised_data, affine, image.header)
            logging.info('Denoised volume saved as %s', odenoised)
            if return_sigma:
                save_nifti(osigma, sigma, affine, image.header)
                logging.info('Sigma volume saved as %s', osigma)


class GibbsRingingFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'gibbs_ringing'

    def run(self, input_files, slice_axis=2, n_points=3, num_processes=1,
            out_dir='', out_unring='dwi_unring.nii.gz'):
        r"""Workflow for applying Gibbs Ringing method.

        Parameters
        ----------
        input_files : string
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        slice_axis : int, optional
            Data axis corresponding to the number of acquired slices.
            Could be (0, 1, or 2): for example, a value of 2 would mean the
            third axis.
        n_points : int, optional
            Number of neighbour points to access local TV (see note).
        num_processes : int or None, optional
            Split the calculation to a pool of children processes. Only
            applies to 3D or 4D `data` arrays. Default is 1. If < 0 the maximal
            number of cores minus ``num_processes + 1`` is used (enter -1 to
            use as many cores as possible). 0 raises an error.
        out_dir : string, optional
            Output directory. (default current directory)
        out_unring : string, optional
            Name of the resulting denoised volume.

        References
        ----------
        .. [1] Neto Henriques, R., 2018. Advanced Methods for Diffusion MRI
        Data Analysis and their Application to the Healthy Ageing Brain
        (Doctoral thesis). https://doi.org/10.17863/CAM.29356

        .. [2] Kellner E, Dhital B, Kiselev VG, Reisert M. Gibbs-ringing
        artifact removal based on local subvoxel-shifts. Magn Reson Med. 2016
        doi: 10.1002/mrm.26054.

        """
        io_it = self.get_io_iterator()
        for dwi, ounring in io_it:
            logging.info('Unringing %s', dwi)
            data, affine, image = load_nifti(dwi, return_img=True)

            unring_data = gibbs_removal(data, slice_axis=slice_axis,
                                        n_points=n_points,
                                        num_processes=num_processes)

            save_nifti(ounring, unring_data, affine, image.header)
            logging.info('Denoised volume saved as %s', ounring)
